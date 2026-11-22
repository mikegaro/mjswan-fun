"""LIRA logo kick

The Booster K1 kick policy from ``k1_kick.py`` in front of the LIRA logo from
``lira_logo.py``. The letters stand across the K1's path; the gear starts above them,
tilted, and falls onto them once, when the scene loads.

When a kick is over (the ball at rest) or the episode runs out of time, the K1 and the
gear stay where they are (``scene.keep_on_reset``) and only a new ball and kick target
are drawn, from the kick task's own ranges, around the K1 as it stands: the ball 2 m
away, anywhere in the half in front of it; the target 2-4 m past the ball, within about
57 degrees either side of the K1-to-ball line. Nothing keeps a target off the letters.
The K1 falling over resets the whole scene, gear included, which falls again; so does
the reset button.

The logo is turned to face the K1, which always starts at the origin facing +x, and
placed so the A's opening is straight ahead; in this logo the gear sits over the A.
The policy was trained on open ground, so a target behind the letters is reached only
through the A's opening or round the ends; the letters are solid, so a miss bounces off
them.

A Unitree G1 walks a square around the logo at the same time, with the playground's G1
velocity policy run as a *companion* (``add_policy(companion=True)``): a second policy
the browser steps beside the K1's, on its own robot and its own MDP. Its command comes
from ``SquareWalkCommand``, which steers it from corner to corner of a square of
half-side ``G1_HALF_SIDE`` round the logo's middle: straight along each side, turning
on the spot at each corner. It falling over resets the G1 alone. Nothing makes it give
way to the K1 or the ball, and the square's front side crosses the K1's path.

The only spotlight is the logo's (``lira_logo.letters_spec``): the G1's model brings
one of its own, on its pelvis, which is removed.

    uv run lira-kick

Needs what both demos need: ``pip install 'mjswan[wandb,mjlab]'``, ``PLAYGROUND_SRC``
on the ``kick-k1-task`` branch of mjlab-biped-playground, a W&B login, and ``LIRA_LOGO``;
and a G1 checkpoint, ``G1_CHECKPOINT`` (default: the playground's own
``g1_velocity_da`` run).
"""

import copy
import dataclasses
import inspect
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg

import mjswan
from examples.demo import k1_kick, lira_logo
from mjswan.mjlab.runner import create_pt_onnx_export_context, export_checkpoint

#: Where the letters' front face is (m along +x). The ball spawns at most 2 m from the
#: K1, at the origin, so always in front of them.
LOGO_FRONT_X = 3.0
#: The K1's episode ends that leave it, and the gear, where they are: the kick is over,
#: or out of time. Falling over (``fell_over``) resets the scene.
KEEP_ON = ("kick_done", "time_out")
#: Extra contact slots for the ball, robot and gear against the logo's convex pieces.
LOGO_CONTACTS = 120

#: The G1 velocity task the companion policy was trained on, and its checkpoint.
G1_TASK = "Pumas-Velocity-Flat-Unitree-G1-PPO-DA"
G1_CHECKPOINT = Path(
    os.getenv(
        "G1_CHECKPOINT",
        k1_kick.PLAYGROUND_SRC.parent
        / "logs/rsl_rl/g1_velocity_da/wandb_checkpoints/nhlap6tn/model_14999.pt",
    )
)
#: The G1's scene entity, and so the prefix of its joints, bodies and sensors.
G1_ENTITY = "g1"
#: The G1's square round the logo: half its side (m), centred on the logo's middle, and
#: how close (m) to a corner counts as reaching it.
G1_HALF_SIDE = 2.6
G1_CORNER_REACHED = 0.4
#: The G1's forward speed (m/s) along a side, and its steering: turn rate (rad/s) per
#: radian of heading error, up to the most it was trained with.
G1_SPEED = 0.5
G1_TURN_GAIN = 1.5
G1_MAX_TURN = 0.7
#: Extra contact slots for the G1.
G1_CONTACTS = 60

#: -90 degrees about z: logo x runs along world -y, so the K1 reads "LIRA" left to
#: right, and the letters' front (logo -y) faces the K1 (world -x).
LOGO_ROT = (float(np.cos(-np.pi / 4)), 0.0, 0.0, float(np.sin(-np.pi / 4)))


def _gear_start_pose(path: Path) -> tuple[float, ...]:
    """World pose (x, y, z, quat) of the gear's centre at reset, above the letters."""
    shapes = lira_logo.trace_logo(path)
    lo, hi = lira_logo.a_opening_x(shapes)
    gx, gz = lira_logo.gear_centre(shapes)
    gear_low = min(t[3] for t in shapes[4].visual)
    x = LOGO_FRONT_X + lira_logo.DEPTH / 2
    y = -(gx - (lo + hi) / 2)  # logo x runs along world -y
    z = lira_logo.CAP_HEIGHT + lira_logo.GEAR_LIFT + (gz - gear_low)
    quat = np.zeros(4)
    tilt = np.zeros(4)
    mujoco.mju_euler2Quat(tilt, np.radians(lira_logo.GEAR_TILT), "xyz")
    mujoco.mju_mulQuat(quat, np.array(LOGO_ROT), tilt)
    return (float(x), float(y), float(z), *map(float, quat))


class LogoKickCommand(k1_kick.BrowserKickCommand):
    """The browser kick command, with the ball and target drawn around the K1 as it is.

    The scene keeps the K1 where it is when a kick ends (`keep_on_reset`), so a new ball
    has to be drawn around wherever it stands now, not around the origin
    `BrowserKickCommand` assumes. The task's ranges, in the K1's frame: the ball
    ``ball_distance_range`` away at ``ball_bearing_range`` off its heading, the target
    ``target_radius_range`` past the ball, ``target_angle_range`` off that line.

    The K1's pose is read off ``self.robot``, the one entity the trace serves the term.
    The browser serves it from the last forward, which is the K1's pose at the end of
    the kick when the scene keeps it; after a fall, the scene resets it to the start,
    and the ball is drawn around where it fell.
    """

    def __init__(self, cfg: "LogoKickCommandCfg", env: Any):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        cfg = self.cfg
        n = len(env_ids)
        data = self.robot.data
        origin = data.root_link_pos_w[:, :2]
        heading = data.heading_w
        dist = k1_kick._uniform(n, cfg.ball_distance_range)
        bearing = heading + k1_kick._uniform(n, cfg.ball_bearing_range)

        ball = origin + dist[:, None] * torch.stack([bearing.cos(), bearing.sin()], -1)
        self.ball_spawn_xy = ball
        # The target anywhere within `target_max_radius` of the logo's middle,
        # evenly over the disc (hence the square root).
        radius = cfg.target_max_radius * k1_kick._uniform(n, (0.0, 1.0)).sqrt()
        angle = k1_kick._uniform(n, (-math.pi, math.pi))
        centre = torch.tensor(cfg.logo_xy, dtype=torch.float32)
        self.target_xy = centre + radius[:, None] * torch.stack(
            [angle.cos(), angle.sin()], -1
        )
        pose = torch.cat(
            [
                ball,
                torch.full((n, 1), k1_kick.BALL_RADIUS),
                torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4),
            ],
            dim=-1,
        )
        ball_entity = self._env.scene["ball"]
        ball_entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        ball_entity.write_root_link_velocity_to_sim(torch.zeros(n, 6), env_ids=env_ids)


@dataclass(kw_only=True)
class LogoKickCommandCfg(k1_kick.BrowserKickCommandCfg):
    #: The scene entity the browser trace keys ``self.robot`` by.
    entity_name: str = "robot"
    #: The logo's middle on the floor, and how far from it a target can be.
    logo_xy: tuple[float, float] = (0.0, 0.0)
    target_max_radius: float = 3.0

    def build(self, env: Any) -> LogoKickCommand:
        return LogoKickCommand(self, env)


def _target_viz(cfg: Any) -> list[dict[str, Any]]:
    """The target marker, on the floor."""
    del cfg
    return [
        {
            "shape": "sphere",
            "radius": 0.1,
            "color": [1.0, 0.5, 0.0, 0.7],
            "origin": {"state": "target_xy", "components": [0, 1, None]},
        }
    ]


mjswan.register_command(
    "LogoKickCommandCfg",
    mjswan.CommandBinding(
        state_fields=["target_xy", "ball_spawn_xy"],
        command_field="target_xy",
        viz=_target_viz,
    ),
)


def _logo_centre_xy(path: Path) -> tuple[float, float]:
    """The logo's middle on the floor, gear included: the centre of the G1's square."""
    shapes = lira_logo.trace_logo(path)
    lo, hi = lira_logo.a_opening_x(shapes)
    xs = [x for s in shapes for t in s.visual for x in (t[1], t[2], t[4], t[5])]
    mid = (min(xs) + max(xs)) / 2
    return LOGO_FRONT_X + lira_logo.DEPTH / 2, -(mid - (lo + hi) / 2)


class SquareWalkCommand(CommandTerm):
    """A velocity command that walks the robot round a square, corner to corner.

    It heads for the current corner: turning toward it at ``turn_gain`` per radian of
    heading error, and walking at ``speed`` scaled down by how far off it is facing, so
    at a corner it turns on the spot. Once within ``corner_reached`` the next corner
    takes over. Every reset starts again from the first corner, which the robot's start
    pose faces.

    Written with tensor ops only (no Python branching on state) so the browser trace
    replays it: the corner is a one-hot pick, its advance a ``where``.
    """

    cfg: "SquareWalkCommandCfg"

    def __init__(self, cfg: "SquareWalkCommandCfg", env: Any):
        super().__init__(cfg, env)
        n = env.num_envs
        # The one entity held on the term, so the browser trace serves its state.
        self.robot = env.scene[cfg.entity_name]
        self.corners = torch.tensor(cfg.corners, dtype=torch.float32)
        self.corner = torch.zeros(n)
        self.target_xy = self.corners[0].expand(n, 2).clone()
        self.vel_command_b = torch.zeros(n, 3)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.corner = torch.zeros_like(self.corner)
        self._steer()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        self._steer()

    def _steer(self) -> None:
        cfg = self.cfg
        pos = self.robot.data.root_link_pos_w[:, :2]
        pick = lambda corner: (
            (  # noqa: E731
                torch.arange(len(self.corners))[None] == corner[:, None]
            ).float()
            @ self.corners
        )
        reached = (pick(self.corner) - pos).norm(dim=-1) < cfg.corner_reached
        self.corner = torch.where(
            reached, torch.remainder(self.corner + 1, len(self.corners)), self.corner
        )
        self.target_xy = pick(self.corner)
        to = self.target_xy - pos
        error = torch.atan2(to[:, 1], to[:, 0]) - self.robot.data.heading_w
        error = torch.atan2(torch.sin(error), torch.cos(error))
        self.vel_command_b = torch.stack(
            [
                cfg.speed * torch.cos(error).clamp(min=0.0),
                torch.zeros_like(error),
                (cfg.turn_gain * error).clamp(-cfg.max_turn, cfg.max_turn),
            ],
            dim=-1,
        )


@dataclass(kw_only=True)
class SquareWalkCommandCfg(CommandTermCfg):
    #: The scene entity it steers, which the browser trace keys ``self.robot`` by.
    entity_name: str = G1_ENTITY
    #: The square's corners (x, y) in walking order.
    corners: tuple[tuple[float, float], ...] = ()
    corner_reached: float = G1_CORNER_REACHED
    speed: float = G1_SPEED
    turn_gain: float = G1_TURN_GAIN
    max_turn: float = G1_MAX_TURN
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: Any) -> SquareWalkCommand:
        return SquareWalkCommand(self, env)


mjswan.register_command(
    "SquareWalkCommandCfg",
    mjswan.CommandBinding(
        state_fields=["vel_command_b", "corner", "target_xy"],
        command_field="vel_command_b",
    ),
)


def _soccer_ball(spec_fn: Any) -> Any:
    """`spec_fn` with its ball dressed as a soccer ball: white, with 12 black patches.

    The patches sit at an icosahedron's vertices, as a soccer ball's pentagons do. They
    are thin discs, mostly inside the sphere, with no mass and no collisions, so the
    ball rolls and is kicked as before.
    """

    def build() -> mujoco.MjSpec:
        spec = spec_fn()
        geom = spec.geom("ball_geom")
        geom.rgba = (0.95, 0.95, 0.95, 1.0)
        radius = geom.size[0]
        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            v
            for a, b in ((1.0, phi), (-1.0, phi), (1.0, -phi), (-1.0, -phi))
            for v in ((0.0, a, b), (a, b, 0.0), (b, 0.0, a))
        ]
        for i, v in enumerate(vertices):
            axis = np.asarray(v) / np.linalg.norm(v)
            quat = np.zeros(4)
            mujoco.mju_quatZ2Vec(quat, axis)
            spec.body("ball").add_geom(
                name=f"ball_patch_{i}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(0.33 * radius, 0.004, 0.0),
                pos=(radius - 0.0035) * axis,
                quat=quat,
                rgba=(0.05, 0.05, 0.05, 1.0),
                contype=0,
                conaffinity=0,
                density=0.0,
            )
        return spec

    return build


def _without_lights(spec_fn: Any) -> Any:
    """``spec_fn``, its lights removed: the G1's model carries a spotlight on its pelvis."""

    def build() -> mujoco.MjSpec:
        spec = spec_fn()
        for light in list(spec.lights):
            spec.delete(light)
        return spec

    return build


def _retarget(node: Any, seen: set[int] | None = None) -> None:
    """Point every reference to the task's ``robot`` entity at ``G1_ENTITY``, in place.

    The G1 task names its robot three ways: an ``entity_name`` field, a sensor name
    with its ``robot/`` prefix, and a ``SceneEntityCfg`` in a term's ``params``,
    including the ones a term function only takes as a default argument, which are
    made explicit here so they can be renamed too.
    """
    seen = set() if seen is None else seen
    if id(node) in seen or node is None or isinstance(node, (str, int, float, bool)):
        return
    seen.add(id(node))
    if isinstance(node, SceneEntityCfg):
        if node.name == "robot":
            node.name = G1_ENTITY
        return
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, str) and value.startswith("robot/"):
                node[key] = G1_ENTITY + value[len("robot") :]
            else:
                _retarget(value, seen)
        return
    if isinstance(node, (list, tuple)):
        for value in node:
            _retarget(value, seen)
        return
    if not dataclasses.is_dataclass(node):
        return
    if getattr(node, "entity_name", None) == "robot":
        node.entity_name = G1_ENTITY
    func = getattr(node, "func", None)
    params = getattr(node, "params", None)
    if callable(func) and isinstance(params, dict):
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            signature = None
        for name, parameter in (signature.parameters if signature else {}).items():
            default = parameter.default
            if name not in params and isinstance(default, SceneEntityCfg):
                params[name] = copy.deepcopy(default)
    for f in fields(node):
        _retarget(getattr(node, f.name), seen)


def _g1_env_cfg(path: Path) -> tuple[Any, EntityCfg]:
    """The G1 velocity task, retargeted to ``G1_ENTITY``, and its robot entity.

    The config is the companion's own env, the G1 alone, which its terms are traced
    against; the entity is the same G1 for the kick scene.

    The command is ``SquareWalkCommand`` round the logo; the task's events (random
    pushes and spawn poses) are dropped; ``fell_over`` is mjlab's deterministic
    ``bad_orientation``, since the task's stochastic one draws with ``torch.rand``,
    which a trace cannot replay.
    """
    cfg = load_env_cfg(G1_TASK, play=True)
    # Counter-clockwise from above, starting at the corner ahead of the start pose.
    cx, cy = _logo_centre_xy(path)
    h = G1_HALF_SIDE
    # Under the task's name for it, which its `command` observation reads.
    cfg.commands = {
        "twist": SquareWalkCommandCfg(
            corners=(
                (cx + h, cy + h),
                (cx - h, cy + h),
                (cx - h, cy - h),
                (cx + h, cy - h),
            )
        )
    }
    cfg.terminations = {
        "fell_over": TerminationTermCfg(
            func=term_fns.bad_orientation,
            params={"limit_angle": cfg.terminations["fell_over"].params["limit_angle"]},
        ),
    }
    cfg.events = {}
    _retarget(cfg.observations)
    _retarget(cfg.actions)
    _retarget(cfg.commands)
    _retarget(cfg.terminations)

    # Starts halfway along the square's far side (+x), facing +y, toward the first corner.
    robot = copy.deepcopy(cfg.scene.entities["robot"])
    robot.spec_fn = _without_lights(robot.spec_fn)
    z = robot.init_state.pos[2]
    robot.init_state = dataclasses.replace(
        robot.init_state,
        pos=(cx + h, cy, z),
        rot=(float(np.cos(np.pi / 4)), 0.0, 0.0, float(np.sin(np.pi / 4))),
    )
    # The companion's own trace env: the G1 alone, under its scene name, with only what
    # the actor reads. The critic, rewards and curriculum read the task's scene sensors,
    # which the kick scene does not carry.
    cfg.scene.entities = {G1_ENTITY: robot}
    cfg.scene.sensors = ()
    cfg.observations = {"actor": cfg.observations["actor"]}
    cfg.rewards = {}
    cfg.curriculum = {}
    return cfg, robot


def _add_g1(scene: mjswan.SceneHandle, g1_cfg: Any) -> None:
    """The G1 checkpoint as the scene's companion policy."""
    context = create_pt_onnx_export_context(
        G1_TASK, env_cfg=load_env_cfg(G1_TASK, play=True)
    )
    try:
        model = export_checkpoint(context, G1_CHECKPOINT)
        prefix = lambda name: G1_ENTITY + name[len("robot") :]  # noqa: E731
        scene.add_policy(
            name="G1 Walk",
            policy=model,
            env_cfg=g1_cfg,
            task_id=G1_TASK,
            policy_joint_names=[prefix(n) for n in context.joint_names],
            default_joint_pos=context.default_joint_pos or None,
            encoder_bias=context.encoder_bias or None,
            companion=True,
        )
    finally:
        context.close()


def _lira_kick_env_cfg():
    cfg = k1_kick._kick_env_cfg()
    logo_path = lira_logo.LOGO_PATH
    ball = cfg.scene.entities["ball"]
    ball.spec_fn = _soccer_ball(ball.spec_fn)
    cfg.scene.entities["letters"] = EntityCfg(
        spec_fn=lambda: lira_logo.letters_spec(logo_path),
        init_state=EntityCfg.InitialStateCfg(
            pos=(LOGO_FRONT_X + lira_logo.DEPTH / 2, 0.0, 0.0), rot=LOGO_ROT
        ),
    )
    start = _gear_start_pose(logo_path)
    cfg.scene.entities["gear"] = EntityCfg(
        spec_fn=lambda: lira_logo.gear_spec(logo_path),
        init_state=EntityCfg.InitialStateCfg(pos=start[:3], rot=start[3:]),
    )
    # The ball at random around the K1; the target at random around the logo.
    kick = cfg.commands["kick"]
    cfg.commands["kick"] = LogoKickCommandCfg(
        **{f.name: getattr(kick, f.name) for f in fields(kick)},
        logo_xy=_logo_centre_xy(logo_path),
    )
    assert cfg.sim.nconmax is not None
    cfg.sim.nconmax += LOGO_CONTACTS
    return cfg


def _with_g1(cfg: Any, robot: EntityCfg) -> Any:
    """The kick scene with the G1 standing in it."""
    cfg.scene.entities[G1_ENTITY] = robot
    assert cfg.sim.nconmax is not None
    cfg.sim.nconmax += G1_CONTACTS
    return cfg


def add_projects(builder: mjswan.Builder) -> None:
    """Add the "LIRA Kick" project to ``builder``."""
    project = builder.add_project(name="LIRA Kick")
    g1_cfg, g1_robot = _g1_env_cfg(lira_logo.LOGO_PATH)
    scene = project.add_scene_mjlab(
        k1_kick.TASK_ID,
        name="LIRA Kick",
        env_cfg=_with_g1(_lira_kick_env_cfg(), g1_robot),
    )
    scene.set_viewer(
        mjswan.ViewerConfig(
            # From the K1's side, a little to its right, looking on past the logo
            # (the azimuth is the direction the camera looks).
            lookat=(3.0, 0.5, 0.5),
            distance=9.0,
            elevation=-25.0,
            azimuth=30.0,
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
        )
    )
    scene.add_policy_wandb(k1_kick.WANDB_RUN)
    # A kick ending (or running out of time) moves on to the next ball from where the
    # K1 is, the gear staying where it landed; a fall resets everything.
    scene.keep_on_reset(["robot", "gear"], terminations=KEEP_ON)
    _add_g1(scene, g1_cfg)


def main() -> None:
    """Build and launch the demo.

    Environment variables:
        LIRA_LOGO: The logo PNG to trace.
        PLAYGROUND_SRC: mjlab-biped-playground's src/ (kick-k1-task branch).
        MJSWAN_BASE_PATH: Base path for deployment (default: '/').
        MJSWAN_OUT_DIR: Where to build (default: the shared examples/demo/dist).
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser.
    """
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    add_projects(builder)
    out_dir = os.getenv("MJSWAN_OUT_DIR")
    app = builder.build(output_dir=Path(out_dir)) if out_dir else builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
