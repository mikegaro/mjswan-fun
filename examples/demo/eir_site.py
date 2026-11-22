"""EIR Course Site

The EIR course examples (`eir_course.py`) and the Booster K1 kick (`k1_kick.py`) as one
static site, the one published on the course repository's GitHub Pages:
https://lira-unam.github.io/eir-course-reinforcement-learning/

It builds into `examples/demo/site/` rather than the `dist/` the single demos share. The
kick needs the private playground checkout and a W&B login, so the site is built locally:

    MJSWAN_BASE_PATH=/eir-course-reinforcement-learning/ MJSWAN_NO_LAUNCH=1 uv run eir-site

then `site/` is pushed as the course repository's `gh-pages` branch.
"""

import os
from pathlib import Path

import mjswan

from . import eir_course, k1_kick

OUT_DIR = Path(os.getenv("MJSWAN_OUT_DIR", Path(__file__).parent / "site"))


def setup_builder() -> mjswan.Builder:
    """Return the configured builder, not yet built."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    eir_course.add_projects(builder)
    k1_kick.add_projects(builder)
    return builder


def main():
    """Main entry point for the EIR course site.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_OUT_DIR: Where to build (default: examples/demo/site)
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    app = setup_builder().build(output_dir=OUT_DIR)
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
