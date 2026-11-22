import { describe, expect, it } from 'vitest';
import type { MjModel } from 'mujoco';
import type { MjData } from 'mujoco';
import { entityJointAddresses, entityPrefix, restoreJoints, snapshotJoints } from './companion';

/** A model holding just the joint tables `entityJointAddresses` reads. */
function fakeModel(joints: Array<{ name: string; type: number; qpos: number; dof: number }>): MjModel {
  const encoder = new TextEncoder();
  const bytes: number[] = [];
  const nameAdr: number[] = [];
  for (const joint of joints) {
    nameAdr.push(bytes.length);
    bytes.push(...encoder.encode(joint.name), 0);
  }
  return {
    njnt: joints.length,
    names: Uint8Array.from(bytes).buffer,
    name_jntadr: Int32Array.from(nameAdr),
    jnt_type: Int32Array.from(joints.map((j) => j.type)),
    jnt_qposadr: Int32Array.from(joints.map((j) => j.qpos)),
    jnt_dofadr: Int32Array.from(joints.map((j) => j.dof)),
  } as unknown as MjModel;
}

describe('entityPrefix', () => {
  it('is the entity name its joints share', () => {
    expect(entityPrefix(['g1/left_hip_pitch_joint', 'g1/right_knee_joint'])).toBe('g1/');
  });

  it('refuses joints with no entity, or spread over two', () => {
    expect(() => entityPrefix(['left_hip_pitch_joint'])).toThrow(/no entity prefix/);
    expect(() => entityPrefix(['g1/a', 'robot/b'])).toThrow(/more than one entity/);
  });
});

describe('entityJointAddresses', () => {
  it('covers every qpos and qvel slot of the entity\'s joints, free joint included', () => {
    const model = fakeModel([
      { name: 'robot/floating_base_joint', type: 0, qpos: 0, dof: 0 },
      { name: 'robot/knee', type: 3, qpos: 7, dof: 6 },
      { name: 'g1/floating_base_joint', type: 0, qpos: 8, dof: 7 },
      { name: 'g1/knee', type: 3, qpos: 15, dof: 13 },
      { name: 'ball/ball_joint', type: 0, qpos: 16, dof: 14 },
    ]);
    expect(entityJointAddresses(model, 'g1/')).toEqual({
      qpos: [8, 9, 10, 11, 12, 13, 14, 15],
      qvel: [7, 8, 9, 10, 11, 12, 13],
    });
  });
});

describe('snapshotJoints / restoreJoints', () => {
  it('carry only the given joints\' values across a reset', () => {
    const data = { qpos: Float64Array.from([1, 2, 3, 4]), qvel: Float64Array.from([5, 6, 7]) } as unknown as MjData;
    const snapshot = snapshotJoints(data, { qpos: [1, 2], qvel: [0] });
    data.qpos.fill(0);
    data.qvel.fill(0);
    restoreJoints(data, snapshot);
    expect(Array.from(data.qpos)).toEqual([0, 2, 3, 0]);
    expect(Array.from(data.qvel)).toEqual([5, 0, 0]);
  });
});
