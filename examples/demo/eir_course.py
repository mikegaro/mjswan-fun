"""EIR Course Demo

The ten examples of the EIR course on reinforcement learning
(https://github.com/LIRA-UNAM/eir-course-reinforcement-learning), each driven by the
policy the course ships for it in ``models/``:

- **Furuta Pendulum**: example1 (balance) and example2 (swing-up), learned with tabular
  Q-learning. The Q-table is turned into an ONNX policy here: discretize, look up, argmax.
- **K1 Velocity**: example3-example9 add two reward terms each on flat ground, and
  example10 is the full rough-terrain task.

The course is public, so it is cloned at a pinned commit into ``examples/demo/.cache/``
(or set ``EIR_COURSE_DIR`` to a checkout). It is put on ``sys.path`` rather than
installed, because it pins Python 3.13 and mjswan's environment runs 3.12.

    uv run python -m examples.demo.eir_course

Needs `pip install 'mjswan[mjlab]'`.
"""

import copy
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg

import mjswan
from mjswan.mjlab.runner import create_pt_onnx_export_context, export_checkpoint

COURSE_REPO = "https://github.com/LIRA-UNAM/eir-course-reinforcement-learning.git"
COURSE_COMMIT = "4fee8829061ceedccc1df006b52c2a0544311775"
CACHE_DIR = Path(__file__).parent / ".cache"

FURUTA_EXAMPLES = {
    "example1": "example1: balance",
    "example2": "example2: swing-up",
}
#: Each K1 example by the reward terms it adds, as the course README lists them.
K1_EXAMPLES = {
    "example3": "example3: velocity tracking",
    "example4": "example4: upright, pose",
    "example5": "example5: action rate, joint limits",
    "example6": "example6: foot clearance, swing height",
    "example7": "example7: foot slip, soft landing",
    "example8": "example8: air time, self-collisions",
    "example9": "example9: body angular velocity, angular momentum",
    "example10": "example10: rough terrain",
}


# ── The course ───────────────────────────────────────────────


def _course_dir() -> Path:
    """The course checkout: ``EIR_COURSE_DIR``, else a clone at ``COURSE_COMMIT``."""
    if env_dir := os.getenv("EIR_COURSE_DIR"):
        return Path(env_dir).expanduser()
    target = CACHE_DIR / f"eir-course-{COURSE_COMMIT[:7]}"
    if target.is_dir():
        return target
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Cloned beside the target and renamed, so an interrupted clone is never reused.
    staging = Path(tempfile.mkdtemp(dir=CACHE_DIR, prefix=".clone-"))
    try:
        for args in (
            ["init", "-q"],
            ["fetch", "-q", "--depth", "1", COURSE_REPO, COURSE_COMMIT],
            ["checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", str(staging), *args], check=True)
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _import_course(course: Path) -> None:
    """Register the course's tasks (``example1`` ... ``example10``) with mjlab."""
    sys.path.insert(0, str(course / "src"))
    import eir_course  # noqa: F401


# ── Furuta: Q-table → ONNX ───────────────────────────────────


class GreedyQPolicy(torch.nn.Module):
    """The course runner's greedy policy, ``levels[argmax_a Q(s, a)]``, as a graph.

    A bin is the number of interior edges at or below the value, which is what the
    course's ``torch.bucketize(..., right=True)`` returns, written as a comparison so
    it exports. Values outside the range land in the first or last bin, as in training.
    """

    def __init__(self, checkpoint: dict[str, Any]):
        super().__init__()
        from eir_course.tasks.furuta.discretize import bin_edges

        bins = checkpoint["bins"]
        edges = [
            bin_edges(b, lo, hi, sq)
            for b, lo, hi, sq in zip(
                bins,
                checkpoint["state_low"],
                checkpoint["state_high"],
                checkpoint["sqrt_spacing"],
                strict=True,
            )
        ]
        # One row of edges per dimension, padded with +inf, which no value reaches.
        width = max(len(e) for e in edges)
        padded = torch.full((len(edges), width), torch.inf)
        for i, e in enumerate(edges):
            padded[i, : len(e)] = e
        strides = [1] * len(bins)
        for i in range(len(bins) - 2, -1, -1):
            strides[i] = strides[i + 1] * bins[i + 1]

        self.register_buffer("edges", padded)
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.long))
        self.register_buffer("q", checkpoint["q"].float())
        self.register_buffer(
            "levels", torch.linspace(-1.0, 1.0, checkpoint["num_actions"])
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        bins = (obs[..., :, None] >= self.edges).sum(-1)
        state = (bins * self.strides).sum(-1)
        return self.levels[self.q[state].argmax(-1)][..., None]


def _check_greedy(policy: GreedyQPolicy, checkpoint: dict[str, Any]) -> None:
    """Fail the build unless the graph acts exactly as the course's own discretizer."""
    from eir_course.tasks.furuta.discretize import Discretizer

    disc = Discretizer(
        checkpoint["bins"],
        checkpoint["state_low"],
        checkpoint["state_high"],
        checkpoint["sqrt_spacing"],
    )
    low = torch.tensor(checkpoint["state_low"])
    high = torch.tensor(checkpoint["state_high"])
    # 20 % past each end, to cover the clamping bins too.
    states = low + (high - low) * (torch.rand(4096, len(low)) * 1.4 - 0.2)
    expected = policy.levels[policy.q[disc.index(states)].argmax(-1)][:, None]
    if not torch.equal(policy(states), expected):
        raise AssertionError("Q-table graph disagrees with the course's Discretizer.")


def _q_table_onnx(pt_path: Path) -> onnx.ModelProto:
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    policy = GreedyQPolicy(checkpoint).eval()
    _check_greedy(policy, checkpoint)
    buffer = io.BytesIO()
    torch.onnx.export(
        policy,
        (torch.zeros(1, len(checkpoint["bins"])),),
        buffer,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )
    return onnx.load_from_string(buffer.getvalue())


def _add_furuta(builder: mjswan.Builder, course: Path) -> None:
    project = builder.add_project(name="Furuta Pendulum")
    for task_id, name in FURUTA_EXAMPLES.items():
        env_cfg = load_env_cfg(task_id, play=True)
        scene = project.add_scene_mjlab(task_id, name=name, env_cfg=env_cfg)
        scene.set_viewer(
            mjswan.ViewerConfig(
                lookat=(0.0, 0.0, 0.6),
                distance=1.6,
                elevation=-20.0,
                azimuth=135.0,
                origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            )
        )
        scene.add_policy(
            name="q_table",
            policy=_q_table_onnx(course / "models" / f"{task_id}.pt"),
            env_cfg=env_cfg,
            task_id=task_id,
            # mjlab compiles each entity under its name: `robot/arm`.
            policy_joint_names=["robot/arm"],
            default_joint_pos=[0.0],
            default=True,
        )


# ── K1 velocity ──────────────────────────────────────────────


def _k1_env_cfg(task_id: str) -> Any:
    """The example's play config, with what the browser cannot trace replaced."""
    cfg = load_env_cfg(task_id, play=True)
    # `stochastic_bad_orientation` draws with `torch.rand`, which a trace cannot replay.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=term_fns.bad_orientation,
        params={"limit_angle": cfg.terminations["fell_over"].params["limit_angle"]},
    )
    # example10's domain randomization: `pseudo_inertia` does not export to ONNX, and a
    # viewer shows the nominal robot anyway. example3-example9 have none.
    cfg.events = {n: t for n, t in cfg.events.items() if t.mode != "startup"}
    return cfg


def _add_k1(builder: mjswan.Builder, course: Path) -> None:
    project = builder.add_project(name="K1 Velocity")
    for task_id, name in K1_EXAMPLES.items():
        env_cfg = _k1_env_cfg(task_id)
        scene = project.add_scene_mjlab(task_id, name=name, env_cfg=env_cfg)
        scene.set_viewer(
            mjswan.ViewerConfig(
                lookat=(0.0, 0.0, 0.0),
                distance=3.0,
                elevation=-20.0,
                azimuth=180.0,
                origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
                body_name="Trunk",
            )
        )
        # What `add_policy_wandb` does with a run's checkpoints, for the shipped one.
        context = create_pt_onnx_export_context(task_id, env_cfg=copy.deepcopy(env_cfg))
        try:
            model = export_checkpoint(context, course / "models" / f"{task_id}.pt")
            scene.add_policy(
                name=task_id,
                policy=model,
                env_cfg=env_cfg,
                task_id=task_id,
                policy_joint_names=context.joint_names or None,
                default_joint_pos=context.default_joint_pos or None,
                encoder_bias=context.encoder_bias or None,
                default=True,
            )
        finally:
            context.close()


def add_projects(builder: mjswan.Builder) -> None:
    """Add the course's two projects, Furuta Pendulum and K1 Velocity, to ``builder``."""
    course = _course_dir()
    _import_course(course)
    _add_furuta(builder, course)
    _add_k1(builder, course)


def setup_builder() -> mjswan.Builder:
    """Return the configured builder, not yet built."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    add_projects(builder)
    return builder


def main():
    """Main entry point for the EIR course demo.

    Environment variables:
        EIR_COURSE_DIR: A course checkout to use instead of the pinned clone
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
