# LIRA logo drop: an mjswan demo (design)

## Context
This is a new mjswan demo with no robot and no policy: a 3D version of the LIRA lab logo
(`/home/miguel/website_v1/images/logo.png`, 512x316 RGBA: gold "LIRA" letters and a navy partial gear).
The gold letters stand on the ground. The gear starts above them and falls onto them, the same way the
ragdoll lands on the "MuJoCo" letters on mujoco.org. For now it is only an mjswan demo page.
It also ships written instructions for moving it to the `website_v1` site later.
(The earlier WASM-vs-native question on PumaSim is answered and closed. This is a new task.)

## Goals / non-goals
- Goal: the extruded meshes match the logo shapes. The gear falls, tumbles and comes to rest on the
  letter tops with stable contact, and resetting replays the fall. The usual mjswan interactivity
  stays: dragging the gear and the reset button.
- Non-goals: website hero mode (auto-loop, hidden UI) now, PumaSim changes, robots/policies.
- Anti-goals: blocky "pixel staircase" visuals; the gear jittering or sinking into letters;
  new heavy dependencies (e.g. CoACD) just for one demo.

## Approaches (user picked #1)
1. **Trace PNG + extrude** (chosen): auto-trace each shape from logo.png into outlines, extrude them into
   visual meshes, and split the outlines into convex pieces for collision. This gives the exact look, is automatic,
   and needs no new deps.
2. Primitives only (boxes/cylinder segments): robust physics, but the R, the A and the gear look blocky. Rejected.
3. Hand-drawn coordinates / SVG: cleanest edges only if a vector logo exists; otherwise slow manual work.

## Design
**Where:** `/home/miguel/fun/mjswan/examples/demo/lira_logo.py`, plus a `lira-logo` entry wired like the
existing `eir` / `k1_kick` demos (`[project.scripts]` in `pyproject.toml` and a
`_run_module("examples.demo.lira_logo")` function in `src/mjswan/cli.py`, which are both already locally
modified; add to them). Leave all changes unstaged and do not commit (user rule).

**Logo source:** the path comes from the `LIRA_LOGO` env var, defaulting to
`/home/miguel/website_v1/images/logo.png`. Following the examples' convention, assets are not
vendored into the repo.

**Tracing (numpy / Pillow / scipy only):**
- Classify opaque pixels as gold or navy by RGB distance to the two dominant colours.
  Those colours are sampled from the image and reused as the material rgba.
- Upscale each mask about 4x with a slight blur and threshold it to smooth the 512px edges.
- `scipy.ndimage.label` separates the components: L, I, R, A (gold) and the gear (navy).
- Trace contours (marching squares in numpy), including holes such as the R's bowl and the A's
  counter. Simplify them with Douglas-Peucker. Units: letter cap height = 1.0 m.

**Geometry (MjSpec, built in Python):**
- The logo plane maps to the world x-z plane (upright letters), extruded along y. Depth is about 0.35 m;
  the builder can tune it.
- Visual: one extruded mesh per shape (side walls plus triangulated caps). It uses contype=0/conaffinity=0
  and the sampled gold or navy material.
- Collision: each outline (with holes) is split into convex 2D pieces, and each piece is extruded into a
  convex mesh geom (group 3, hidden). Target: 30 or fewer pieces for the gear so the falling body stays cheap.
  - The triangulation and convex merge (e.g. ear-clip, then Hertel-Mehlhorn) are the builder's choice.
  - If a hand-rolled triangulator is unreliable, adding `shapely` or `mapbox-earcut` to the `demo`
    extra only is acceptable.
- The letters are static worldbody geoms resting on the floor. The gear is one freejoint body with its
  logo x-position and the same depth plane, lifted about 1.5 m above the letter tops. A small initial tilt
  makes it tumble before landing across "R"/"A". It has a `home` keyframe for reset.
- Scene: checker floor plane, skybox, one directional light, 2 ms timestep. Density/friction are tuned so
  the gear settles without bouncing off the letters. ViewerConfig: front three-quarter view
  (lookat at the logo centre, distance ~6, elevation ~-15, azimuth facing the letter fronts).

**Website migration doc:** `examples/demo/lira_logo_website.md` covers:
- the build command (`MJSWAN_BASE_PATH=/lira-logo/ MJSWAN_NO_LAUNCH=1 uv run lira-logo` building to an output dir)
- copying `dist/` into `website_v1/lira-logo/` (Jekyll serves it as static files; check `_config.yaml`
  `exclude`/`include`)
- embedding it with an `<iframe>` in `index.md`
- what a hero version would still need: auto-reset loop, hidden panel, fixed camera. The doc should say
  whether mjswan supports these today.

**Spec file:** once approved, save this design to
`/home/miguel/fun/mjswan/docs/superpowers/specs/2026-11-21-lira-logo-demo-design.md` (unstaged).

## Left to the builder
The exact thresholds and smoothing, the convex-split algorithm, the extrusion depth, the gear drop
height/tilt and the material values. These are fine as long as the visuals match the logo and the landing is stable.

## Verification
- `uv run python -c` smoke test: build the MjSpec, compile it, and step about 3 s of sim with native `mujoco`.
  Then assert that the gear's z velocity is ~0, that its lowest point is above the letter tops, and that there
  is no NaN or large penetration (`data.contact.dist` > -5 mm).
- Run `uv run lira-logo` and check in the browser (claude-in-chrome screenshot) that the logo looks right,
  that the gear lands on the letters, and that reset replays the drop.
- Run the existing mjswan tests (`uv run pytest -q`) to check nothing in the CLI/pyproject wiring broke.
  No new tests (user rule).
