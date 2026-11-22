# Moving the LIRA logo demo to `website_v1`

`lira_logo.py` builds a self-contained static app. The lab website
(`LIRA-UNAM/website_v1`, Jekyll, `baseurl: "/"`) can serve it as a plain folder and
embed it with an `<iframe>`. Nothing on the website needs to be compiled.

## 1. Build for a subfolder

From the mjswan checkout:

```sh
MJSWAN_BASE_PATH=/lira-logo/ \
MJSWAN_OUT_DIR=/home/miguel/website_v1/lira-logo \
MJSWAN_NO_LAUNCH=1 \
uv run lira-logo
```

- `MJSWAN_BASE_PATH` must match the folder the site serves it from, with both slashes.
  If the site ever moves under a sub-path (a non-`/` `baseurl`), prefix it here too.
- `LIRA_LOGO=/path/to/logo.png` traces a different copy of the logo. The default is
  `website_v1/images/logo.png`, so a logo update on the website only needs a rebuild.

The output is about 30 MB, mostly the MuJoCo and ONNX Runtime WebAssembly files.
The scene itself (`lira/lira_logo/scene.mjz`) is small.

## 2. Let Jekyll copy it unchanged

Jekyll copies folders that don't start with `_` or `.` as they are, so `lira-logo/` is
published at `/lira-logo/`. Check two things:

- Nothing in `_config.yaml`'s `exclude:` matches `lira-logo`.
- The built `assets/` folder holds `.wasm` files. GitHub Pages serves them with the right
  MIME type. If the site's build workflow (`.github/workflows/build-site.yaml`) prunes or
  minifies files, keep it away from `lira-logo/`.

Commit the built folder like any other static asset. Alternatively, add a workflow step
that runs the build above, though that needs `uv` and this mjswan checkout in CI.

Check it locally with `bundle exec jekyll serve`, then open
`http://localhost:4000/lira-logo/`.

## 3. Embed it on a page

In `index.md` (or any page), where the logo should appear:

```html
<iframe
  src="{{ '/lira-logo/?panel=0' | relative_url }}"
  title="LIRA logo simulation"
  style="width: 100%; aspect-ratio: 16 / 9; border: 0;"
  loading="lazy"
  allow="fullscreen; xr-spatial-tracking">
</iframe>
```

- `?panel=0` hides the mjswan control panel. Drop it to keep the controls.
- The camera is the one `lira_logo.py` sets: front view, looking at the word.

## 4. What a landing-page hero would still need

The mujoco.org-style hero replays the fall without anyone touching it. mjswan today has:

| Need | Status in mjswan |
|---|---|
| Hide the control panel | Available: `?panel=0` |
| Fixed start camera | Available: `ViewerConfig` in `lira_logo.py` |
| Restart the drop | Manual only: the reset button replays the `home` keyframe |
| Automatic loop (reset every N seconds) | Not available. It needs either a small reset timer added to the mjswan frontend, or a page script that reloads the iframe (crude: it reloads the WASM too) |
| Locking the camera (no orbiting) | Not available as an option |

So for a hero, the remaining work is an auto-reset option in mjswan (for example an
`?autoreset=<seconds>` URL parameter that calls the same reset as the button) and,
optionally, a camera lock. Everything else above is ready.
