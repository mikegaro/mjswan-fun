/**
 * A companion policy: a second robot's policy, stepped every control step beside the
 * selected one (format 11, `add_policy(companion=True)`).
 *
 * It brings its own MDP and so its own term state: commands, observation runner, action
 * terms, terminations and the graph sessions and slot reader they read through. What it
 * shares with the active policy is the model and data, the physics substeps its action
 * terms join, and the scene. A termination of its own resets only its robot: the qpos
 * and qvel of every joint under its entity prefix, back to keyframe 0. The selected
 * policy's own termination resets the scene but leaves a companion's robot where it is,
 * its episode running; a reset someone asks for (the reset button, a load) resets it too.
 *
 * Not carried: events (refused at load, so a companion's robot only ever starts from the
 * keyframe), structured sensors (raycast, contact, camera), motions and plugins.
 */
import type { MainModule, MjData, MjModel } from 'mujoco';
import type * as THREE from 'three';
import type * as ort from 'onnxruntime-web/wasm';
import { ActionManager, type ManagedActionTerm } from '../action/ActionManager';
import { clampActions, readClipActions } from '../action/applyAction';
import { CommandManager, type CommandsConfig } from '../command';
import type { ModelFieldDefaults } from '../event/modelWrite';
import { getOrt } from '../onnx/ortEnv';
import { OnnxSessionCache, type SlotReader } from '../onnx/session';
import { createSlotReader } from '../onnx/slotReader';
import { BufferBudget, reserveTermFeeds } from '../policy/bufferBudget';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyRunner } from '../policy/PolicyRunner';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { PolicyConfig } from '../policy/types';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import type { SeededRng } from '../rng';
import { TerminationManager } from '../termination/TerminationManager';

/** A companion as the runtime resolved it: config plus bytes. */
export type CompanionInput = {
  config: PolicyConfig;
  onnx: ArrayBuffer;
  graphs?: Array<{ name: string; data: ArrayBuffer }>;
};

/** What a companion borrows from the runtime that hosts it. */
export interface CompanionHost {
  readonly mujoco: MainModule;
  readonly mjModel: MjModel;
  readonly mjData: MjData;
  readonly scene: THREE.Scene;
  readonly bodies: Record<number, THREE.Group> | null;
  readonly mujocoRoot: THREE.Group | null;
  readonly modelDefaults: ModelFieldDefaults | null;
  readonly rng: SeededRng;
  /** Seconds per control step. */
  readonly stepDt: number;
  /** The runtime's own action-manager builder, handed the companion's dependencies. */
  buildActionManager(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder,
    budget: BufferBudget,
    deps: { commandManager: CommandManager; graphs: OnnxSessionCache; readSlot: SlotReader },
  ): ActionManager | null;
  /** The joint-bias map the runtime derives from a policy config. */
  jointBias(config: PolicyConfig): Map<string, number>;
}

const EMPTY_ACTIONS = new Float32Array(0);

/** The entity prefix (`g1/`) of a policy's joints, which its reset is scoped to. */
export function entityPrefix(jointNames: readonly string[]): string {
  const first = jointNames[0] ?? '';
  const slash = first.indexOf('/');
  if (slash <= 0) {
    throw new Error(
      `companion: policy joint "${first}" carries no entity prefix ("<entity>/<joint>"), ` +
        'so its robot cannot be told apart from the scene\'s.',
    );
  }
  const prefix = first.slice(0, slash + 1);
  const outside = jointNames.filter((name) => !name.startsWith(prefix));
  if (outside.length > 0) {
    throw new Error(
      `companion: its joints span more than one entity: ${prefix} and ${outside.join(', ')}.`,
    );
  }
  return prefix;
}

/** A set of joints' qpos and qvel, to carry across a reset. */
export type JointSnapshot = { qpos: number[]; qvel: number[]; qposValues: Float64Array; qvelValues: Float64Array };

/** Take the values at `joints`' addresses. */
export function snapshotJoints(mjData: MjData, joints: { qpos: number[]; qvel: number[] }): JointSnapshot {
  return {
    ...joints,
    qposValues: Float64Array.from(joints.qpos, (adr) => mjData.qpos[adr]),
    qvelValues: Float64Array.from(joints.qvel, (adr) => mjData.qvel[adr]),
  };
}

/** Put back what {@link snapshotJoints} took. */
export function restoreJoints(mjData: MjData, snapshot: JointSnapshot): void {
  const { qpos, qvel } = mjData;
  snapshot.qpos.forEach((adr, i) => (qpos[adr] = snapshot.qposValues[i]));
  snapshot.qvel.forEach((adr, i) => (qvel[adr] = snapshot.qvelValues[i]));
}

/** qpos and qvel addresses of every joint whose name starts with `prefix`. */
export function entityJointAddresses(
  mjModel: MjModel,
  prefix: string,
): { qpos: number[]; qvel: number[] } {
  const names = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const qpos: number[] = [];
  const qvel: number[] = [];
  // Width of each joint type in qpos and qvel: free, ball, slide, hinge.
  const qposWidth = [7, 4, 1, 1];
  const qvelWidth = [6, 3, 1, 1];
  for (let j = 0; j < mjModel.njnt; j++) {
    const start = mjModel.name_jntadr[j];
    let end = start;
    while (end < names.length && names[end] !== 0) end++;
    if (!decoder.decode(names.subarray(start, end)).startsWith(prefix)) continue;
    const type = mjModel.jnt_type[j];
    for (let k = 0; k < qposWidth[type]; k++) qpos.push(mjModel.jnt_qposadr[j] + k);
    for (let k = 0; k < qvelWidth[type]; k++) qvel.push(mjModel.jnt_dofadr[j] + k);
  }
  return { qpos, qvel };
}

export class CompanionPolicy {
  readonly name: string;
  private readonly graphs = new OnnxSessionCache();
  private readonly commands = new CommandManager();
  private readonly readSlot: SlotReader;
  private runner: PolicyRunner | null = null;
  private stateBuilder: PolicyStateBuilder | null = null;
  private actions: ActionManager | null = null;
  private module: OnnxModule | null = null;
  private inputDict: Record<string, ort.Tensor> | null = null;
  private terminations: TerminationManager | null = null;
  private jointBias = new Map<string, number>();
  private clipActions: number | null = null;
  private episodeLength = 0;
  private inferencing = false;
  private lastError: string | null = null;
  private joints: { qpos: number[]; qvel: number[] } = { qpos: [], qvel: [] };

  private constructor(
    private readonly host: CompanionHost,
    private readonly input: CompanionInput,
  ) {
    this.name = String(input.config.name ?? 'companion');
    this.readSlot = createSlotReader(
      () => ({
        mujoco: host.mujoco,
        mjModel: host.mjModel,
        mjData: host.mjData,
        commandManager: this.commands,
        episodeLength: this.episodeLength,
        lastActions: () => this.runner?.getLastActions() ?? null,
        actionTermField: (term, field) => this.actions?.getStateField(term, field) ?? null,
        modelDefaults: host.modelDefaults,
      }),
      { jointBias: (name) => this.jointBias.get(name) ?? 0 },
    );
  }

  /** Build every term against the host's model, then run the first reset. */
  static async load(host: CompanionHost, input: CompanionInput): Promise<CompanionPolicy> {
    const companion = new CompanionPolicy(host, input);
    try {
      await companion.init();
    } catch (error) {
      await companion.dispose();
      throw error;
    }
    return companion;
  }

  private async init(): Promise<void> {
    const { host } = this;
    const config = this.input.config;
    if (config.events && config.events.length > 0) {
      throw new Error(
        `companion "${this.name}": carries events, which a companion does not run; ` +
          'its robot starts from the keyframe. Build it with events={}.',
      );
    }
    const jointNames = config.policy_joint_names ?? [];
    if (jointNames.length === 0) {
      throw new Error(`companion "${this.name}": its config has no policy_joint_names.`);
    }
    this.joints = entityJointAddresses(host.mjModel, entityPrefix(jointNames));

    const budget = new BufferBudget();
    reserveTermFeeds(config, budget, { raycast: {}, camera: {} });
    await this.graphs.load(this.input.graphs ?? []);
    this.jointBias = host.jointBias(config);
    this.clipActions = readClipActions(config.clip_actions);

    if (config.commands && typeof config.commands === 'object') {
      this.commands.initialize(config.commands as CommandsConfig, {
        mujoco: host.mujoco,
        mjModel: host.mjModel,
        mjData: host.mjData,
        scene: host.scene,
        bodies: host.bodies,
        mujocoRoot: host.mujocoRoot,
        // Its own reset, not the scene's: the selected policy's episode carries on.
        requestReset: () => void this.resetOwnRobot(),
        rng: host.rng,
        onnxSessions: this.graphs,
        readOnnxSlot: this.readSlot,
      });
    }

    const runner = new PolicyRunner(config, {
      policyModules: { tracking: TrackingPolicy, locomotion: LocomotionPolicy },
      observations: {},
      motions: [],
      onnxSessions: this.graphs,
      readOnnxSlot: this.readSlot,
      budget,
    });
    await runner.init({
      mujoco: host.mujoco,
      mjModel: host.mjModel,
      mjData: host.mjData,
      scene: host.scene,
      commandManager: this.commands,
    });
    await runner.preloadAll();
    this.runner = runner;
    this.stateBuilder = new PolicyStateBuilder(
      host.mujoco,
      host.mjModel,
      host.mjData,
      runner.getPolicyJointNames(),
    );
    this.actions = host.buildActionManager(config, runner, this.stateBuilder, budget, {
      commandManager: this.commands,
      graphs: this.graphs,
      readSlot: this.readSlot,
    });

    if (config.terminations && Object.keys(config.terminations).length > 0) {
      this.terminations = new TerminationManager(config.terminations, {}, runner, {
        onnxSessions: this.graphs,
        readOnnxSlot: this.readSlot,
        episodeLength: () => this.episodeLength,
        stepDt: host.stepDt,
      });
    }

    const module = new OnnxModule(this.input.onnx, {
      in_keys: config.in_keys,
      out_keys: config.out_keys,
    });
    await module.init();
    this.module = module;
    this.inputDict = module.initInput();
    await this.resetTerms();
    console.log(`[companion] "${this.name}" loaded`, {
      obsSize: runner.getObservationSize(),
      joints: jointNames.length,
    });
  }

  /** Its action terms, which the physics substeps apply with the active policy's. */
  get actionTerms(): ManagedActionTerm['term'][] {
    return this.actions?.terms ?? [];
  }

  /** The part of a control step before the physics: observe, infer, process the action. */
  async beginStep(): Promise<void> {
    const { runner, stateBuilder } = this;
    if (!runner || !stateBuilder) return;
    await this.commands.settled();
    const obs = await runner.collectObservationsByKey(stateBuilder.build());
    const fresh = await this.infer(obs);
    const actions = this.actions;
    if (!actions) return;
    try {
      await actions.process(
        runner.getLastActions() ?? EMPTY_ACTIONS,
        { episodeLength: this.episodeLength, stepDt: this.host.stepDt },
        { fresh },
      );
    } catch (error) {
      console.error(`[companion] "${this.name}": processing the action failed:`, error);
    }
  }

  /**
   * Its terminations, after the physics and before the step's forward. True when they
   * fired: the caller then calls {@link resetOwnRobot} unless the whole scene resets.
   */
  async terminated(): Promise<boolean> {
    this.episodeLength += 1;
    if (!this.terminations || !this.stateBuilder) return false;
    return (await this.terminations.evaluate(this.stateBuilder.build())).done;
  }

  /** After the step's forward: commands advance, as for the active policy. */
  endStep(afterReset: boolean): void {
    this.commands.update(this.host.stepDt, afterReset);
  }

  /** Its robot's qpos and qvel, to carry across a scene reset that is not its own. */
  snapshot(): JointSnapshot {
    return snapshotJoints(this.host.mjData, this.joints);
  }

  /** Put back what {@link snapshot} took. */
  restore(snapshot: JointSnapshot): void {
    restoreJoints(this.host.mjData, snapshot);
  }

  /** Put its robot alone back to keyframe 0 and reset its terms. Before the forward. */
  async resetOwnRobot(): Promise<void> {
    console.info(`[companion] "${this.name}": reset after ${this.episodeLength} step(s)`);
    const { mjModel, mjData } = this.host;
    const key = mjModel.nkey > 0 ? mjModel.key_qpos : mjModel.qpos0;
    const qpos = mjData.qpos;
    for (const adr of this.joints.qpos) qpos[adr] = key[adr];
    const qvel = mjData.qvel;
    for (const adr of this.joints.qvel) qvel[adr] = 0;
    await this.resetTerms();
  }

  /** mjlab's reset order for its terms: observations, actions, commands, terminations. */
  async resetTerms(): Promise<void> {
    this.episodeLength = 0;
    if (this.module) this.inputDict = this.module.initInput();
    if (this.runner && this.stateBuilder) this.runner.reset(this.stateBuilder.build());
    try {
      await this.actions?.reset();
      await this.commands.resetTerms();
    } catch (error) {
      console.warn(`[companion] "${this.name}": reset terms failed:`, error);
    }
    this.terminations?.reset();
  }

  private async infer(obs: Record<string, Float32Array>): Promise<boolean> {
    const { module, runner } = this;
    if (!module || !runner || this.inferencing) return false;
    this.inferencing = true;
    try {
      const { Tensor } = getOrt();
      const input: Record<string, ort.Tensor> = { ...(this.inputDict ?? module.initInput()) };
      for (const [key, value] of Object.entries(obs)) {
        input[key] = new Tensor('float32', value, module.inputDims(key, value.length));
      }
      if (module.inKeys.some((key) => !input[key])) {
        console.warn(`[companion] "${this.name}": missing ONNX input`, {
          wanted: module.inKeys,
          available: Object.keys(input),
        });
        return false;
      }
      const [result, carry] = await module.runInference(input);
      if (Object.keys(carry).length > 0) this.inputDict = { ...this.inputDict, ...carry };
      const outKey = module.outKeys[0];
      const tensor = result.action ?? (outKey ? result[outKey] : null) ?? null;
      if (!tensor) return false;
      const raw = tensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      if (this.actions && action.length !== runner.getNumActions()) {
        console.warn(`[companion] "${this.name}": action size mismatch`, {
          expected: runner.getNumActions(),
          got: action.length,
        });
        return false;
      }
      clampActions(action, this.clipActions);
      runner.setLastActions(action);
      this.lastError = null;
      return true;
    } catch (error) {
      const message = String(error);
      if (message !== this.lastError) {
        this.lastError = message;
        console.error(`[companion] "${this.name}": ONNX inference failed:`, error);
      }
      return false;
    } finally {
      this.inferencing = false;
    }
  }

  async dispose(): Promise<void> {
    const module = this.module;
    this.module = null;
    this.runner = null;
    this.stateBuilder = null;
    this.actions = null;
    this.terminations = null;
    this.commands.dispose();
    await module?.dispose();
    await this.graphs.clear();
  }
}
