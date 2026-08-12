import type { NextConfig } from "next";

// Two build shapes from one config.
//
//   dev / local  : normal server build, with /api proxied to uvicorn on :8000
//   NEXT_OUTPUT  : static export, for the Hugging Face Space
//
// The Space runs a python:slim image, so there is no Node at runtime. Every page
// here is client-rendered and already prerenders as static, so exporting to
// plain files lets FastAPI serve the whole app itself - one process instead of
// two, no Node in the runtime image, and roughly 1.5 GB smaller. `rewrites` are
// unsupported under export and unnecessary there: same origin means the
// relative /api paths hit FastAPI directly.
const isExport = process.env.NEXT_OUTPUT === "export";

const nextConfig: NextConfig = isExport
  ? {
      output: "export",
      // Emits out/evidence/index.html rather than out/evidence.html, which is
      // what StaticFiles(html=True) resolves cleanly.
      trailingSlash: true,
    }
  : {
      async rewrites() {
        return [
          {
            source: "/api/:path*",
            destination: "http://127.0.0.1:8000/api/:path*",
          },
        ];
      },
    };

export default nextConfig;
