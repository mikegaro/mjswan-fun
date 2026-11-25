/**
 * Fetch each same-origin `.wasm` as its gzipped `.wasm.gz` sibling and unpack it in the
 * page. A plain static host (Apache with no `.htaccess`, say) sends the MuJoCo and ONNX
 * Runtime wasm uncompressed and untyped: ~18 MB, which gzip cuts to ~5 MB.
 *
 * `gzipWasmPlugin` in `vite.config.ts` writes the siblings and keeps the originals, which
 * this falls back to whenever a sibling is missing or not gzip (the dev server answers a
 * missing file with `index.html`). The unpacked response is typed `application/wasm`, so
 * `WebAssembly.instantiateStreaming` compiles it while it is still being unpacked.
 */

function wasmUrl(input: RequestInfo | URL): URL | null {
  const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
  try {
    const url = new URL(raw, window.location.href);
    return url.origin === window.location.origin && url.pathname.endsWith('.wasm') ? url : null;
  } catch {
    return null;
  }
}

export function installWasmGzipFetch(): void {
  if (typeof window === 'undefined' || typeof DecompressionStream === 'undefined') return;
  const plain = window.fetch.bind(window);
  window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = wasmUrl(input);
    if (url === null) return plain(input, init);
    try {
      const res = await plain(`${url.href}.gz`, init);
      if (!res.ok) return plain(input, init);
      const packed = new Uint8Array(await res.arrayBuffer());
      if (packed[0] !== 0x1f || packed[1] !== 0x8b) return plain(input, init);
      const body = new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
      return new Response(body, { status: 200, headers: { 'Content-Type': 'application/wasm' } });
    } catch {
      return plain(input, init);
    }
  }) as typeof fetch;
}

// Only a production build writes the siblings.
if (import.meta.env.PROD) installWasmGzipFetch();
