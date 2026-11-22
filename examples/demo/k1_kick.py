"""Booster K1 Kick

The K1 walks up to a ball and kicks it toward a target (the orange marker), from the
`mjlab-biped-playground` kick task (`Pumas-Kick-Flat-Booster-K1-PPO-DA`) and the
checkpoints of its W&B run. Each episode ends once the kicked ball comes to rest, and a
new ball and target are drawn.

The task is not on PyPI: point `PLAYGROUND_SRC` at a checkout's `src/` (default
`~/mjlab-biped-playground/src`). It is put on `sys.path` rather than installed because
the playground pins Python 3.13 and mjswan's environment runs 3.12. The W&B run is
private, so a build needs a W&B login.

    uv run python -m examples.demo.k1_kick

Needs `pip install 'mjswan[wandb,mjlab]'`.
"""

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg

import mjswan

PLAYGROUND_SRC = Path(
    os.getenv("PLAYGROUND_SRC", "~/mjlab-biped-playground/src")
).expanduser()
TASK_ID = "Pumas-Kick-Flat-Booster-K1-PPO-DA"
WANDB_RUN = "Pumanoids/k1_kick/b5vd8mjc"

#: The kick task's episode length in training.
EPISODE_LENGTH_S = 15.0
#: Ball speed (m/s) below which a kicked ball counts as at rest, as in the task.
REST_SPEED = 0.5
#: How far (m) the ball must leave its spawn point to count as kicked.
KICKED_DISTANCE = 0.3


def _import_playground() -> None:
    """Register the playground's tasks with mjlab's registry."""
    if not (PLAYGROUND_SRC / "playground").is_dir():
        raise FileNotFoundError(
            f"No playground package under {PLAYGROUND_SRC}. Set PLAYGROUND_SRC to a "
            "mjlab-biped-playground checkout's src/ directory."
        )
    sys.path.insert(0, str(PLAYGROUND_SRC))
    import playground  # noqa: F401


_import_playground()

from playground.tasks.kick.mdp.commands import (  # noqa: E402
    BALL_RADIUS,
    KickCommand,
    KickCommandCfg,
)


def _uniform(n: int, rng: tuple[float, float]) -> torch.Tensor:
    # `Tensor.uniform_` is a draw the tracer replays; the task's `torch.rand` is not.
    return torch.empty(n).uniform_(*rng)


class BrowserKickCommand(KickCommand):
    """`KickCommand` reduced to what the browser can trace.

    The robot always resets at the origin facing +x (see `_kick_env_cfg`), so the ball
    and target are drawn in the world frame from the task's ranges: the browser serves
    the robot's pose from `xpos`, which is stale between the reset events and the
    forward, and has no `qpos` reader the task's own reset-time read could use. Contact
    tracking feeds only the critic and the training terminations, so it is dropped.
    """

    # mjlab's reset (metrics, timer, resample) is what the graph and runtime already
    # run; the task's addition reads the termination manager, which a trace refuses.
    reset = CommandTerm.reset

    def __init__(self, cfg: KickCommandCfg, env: Any):
        super().__init__(cfg, env)
        # A trace keys every entity held on the term by one name; the ball is reached
        # through `env.scene` instead.
        del self.robot, self.ball
        self.ball_spawn_xy = torch.zeros_like(self.target_xy)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        cfg = self.cfg
        n = len(env_ids)
        dist = _uniform(n, cfg.ball_distance_range)
        bearing = _uniform(n, cfg.ball_bearing_range)
        radius = _uniform(n, cfg.target_radius_range)
        angle = _uniform(n, cfg.target_angle_range)

        ball = dist[:, None] * torch.stack([bearing.cos(), bearing.sin()], dim=-1)
        heading = bearing + angle
        target = ball + radius[:, None] * torch.stack(
            [heading.cos(), heading.sin()], dim=-1
        )
        self.target_xy = target
        self.ball_spawn_xy = ball

        pose = torch.cat(
            [
                ball,
                torch.full((n, 1), BALL_RADIUS),
                torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4),
            ],
            dim=-1,
        )
        ball_entity = self._env.scene["ball"]
        ball_entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        ball_entity.write_root_link_velocity_to_sim(torch.zeros(n, 6), env_ids=env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class BrowserKickCommandCfg(KickCommandCfg):
    def build(self, env: Any) -> BrowserKickCommand:
        return BrowserKickCommand(self, env)


def _kick_viz(cfg: Any) -> list[dict[str, Any]]:
    """The task's target marker, on the ground."""
    del cfg
    return [
        {
            "shape": "sphere",
            "radius": 0.1,
            "color": [1.0, 0.5, 0.0, 0.5],
            "origin": {"state": "target_xy", "components": [0, 1, None]},
        }
    ]


mjswan.register_command(
    "BrowserKickCommandCfg",
    mjswan.CommandBinding(
        state_fields=["target_xy", "ball_spawn_xy"],
        command_field="target_xy",
        viz=_kick_viz,
    ),
)


def kicked_ball_at_rest(env: Any, command_name: str = "kick") -> torch.Tensor:
    """The ball has left its spawn point and stopped: this kick is over."""
    cmd = env.command_manager.get_term(command_name)
    ball = env.scene["ball"].data
    moved = (ball.root_link_pos_w[:, :2] - cmd.ball_spawn_xy).norm(dim=-1)
    speed = ball.root_link_lin_vel_w[:, :2].norm(dim=-1)
    return (moved > KICKED_DISTANCE) & (speed < REST_SPEED)


def _kick_env_cfg() -> Any:
    """The task's play config, trimmed to what a browser viewer needs."""
    cfg = load_env_cfg(TASK_ID, play=True)
    kick = cfg.commands["kick"]
    cfg.commands["kick"] = BrowserKickCommandCfg(
        **{f.name: getattr(kick, f.name) for f in fields(kick)}
    )
    # Domain randomization is for training; show the nominal robot and ball.
    cfg.events = {
        name: term
        for name, term in cfg.events.items()
        if term.mode != "startup" and name != "randomize_terrain"
    }
    # A fixed spawn the ball and target are drawn around (`_BrowserKickCommand`).
    cfg.events["reset_base"].params["pose_range"] = {"z": (0.01, 0.05)}
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True),
        # The task's `stochastic_bad_orientation` without its random draw.
        "fell_over": TerminationTermCfg(
            func=term_fns.bad_orientation,
            params={"limit_angle": cfg.terminations["fell_over"].params["limit_angle"]},
        ),
        "kick_done": TerminationTermCfg(func=kicked_ball_at_rest),
    }
    return cfg


def add_projects(builder: mjswan.Builder) -> None:
    """Add the "Booster K1 Kick" project to ``builder``."""
    project = builder.add_project(name="Booster K1 Kick")

    scene = project.add_scene_mjlab(TASK_ID, env_cfg=_kick_env_cfg())
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=4.0,
            elevation=-25.0,
            azimuth=180.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="Trunk",
        )
    )

    # Every logged checkpoint, oldest first, so the viewer shows training progress.
    scene.add_policy_wandb(WANDB_RUN)


def setup_builder() -> mjswan.Builder:
    """Return the configured builder, not yet built."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    add_projects(builder)
    return builder


def main():
    """Main entry point for the K1 kick demo.

    Environment variables:
        PLAYGROUND_SRC: mjlab-biped-playground's src/ (default: ~/mjlab-biped-playground/src)
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
