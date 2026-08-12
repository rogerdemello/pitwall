/**
 * Verify the static export actually produces a working site.
 *
 * This project ships in two shapes and both have to keep working:
 *
 *   static   NEXT_PUBLIC_STATIC=1 NEXT_OUTPUT=export -> frozen JSON under
 *            /data/**.json, served by Hugging Face's free static Space. This is
 *            the deployed one.
 *   live     NEXT_OUTPUT=export with FastAPI on the same origin, in Docker.
 *
 * `next build` succeeding is not the same as the export working. A Suspense
 * bailout from `useSearchParams`, an accidental dynamic route segment, a Route
 * Handler, or a missing `index.html` for a nested path all build clean and fail
 * only when a judge opens the page. Hugging Face's static host in particular
 * will not resolve a nested `index.html`, which is why `Nav` rewrites hrefs and
 * why every route needs its own directory here.
 *
 * So this runs the real export, asserts every route emitted a file, and then
 * serves the output and fetches each page over HTTP.
 *
 *   node scripts/check-export.mjs
 *   node scripts/check-export.mjs --skip-build   # reuse an existing out/
 */

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const OUT = join(ROOT, "out");

/**
 * Routes are discovered from `app/` rather than listed here, so the check can
 * never drift out of date with the app. A hardcoded list would have to be
 * updated by the same person who just forgot to check their new route works.
 */
async function discoverRoutes() {
  const { readdir } = await import("node:fs/promises");
  const appDir = join(ROOT, "app");
  const routes = [];
  const walk = async (dir, prefix) => {
    for (const e of await readdir(dir, { withFileTypes: true })) {
      if (!e.isDirectory()) {
        if (/^page\.(t|j)sx?$/.test(e.name)) routes.push(prefix || "/");
        continue;
      }
      // Route groups (parens) do not appear in the URL; private dirs are skipped.
      if (e.name.startsWith("_")) continue;
      const seg = e.name.startsWith("(") ? "" : `/${e.name}`;
      await walk(join(dir, e.name), prefix + seg);
    }
  };
  await walk(appDir, "");
  return [...new Set(routes)].sort();
}

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon",
  ".mp3": "audio/mpeg", ".woff2": "font/woff2", ".txt": "text/plain",
};

const log = (...a) => console.log(...a);
const fail = (msg) => { console.error(`\n  FAIL  ${msg}`); process.exitCode = 1; };

function run(cmd, args, env) {
  return new Promise((res, rej) => {
    const p = spawn(cmd, args, {
      cwd: ROOT, env: { ...process.env, ...env }, stdio: "inherit", shell: true,
    });
    p.on("close", (code) => (code === 0 ? res() : rej(new Error(`${cmd} exited ${code}`))));
  });
}

/** Where a request path lands in a `trailingSlash: true` export. */
function candidates(urlPath) {
  const clean = urlPath.split("?")[0].replace(/\/+$/, "");
  if (clean === "" || clean === "/index.html") return [join(OUT, "index.html")];
  return [
    join(OUT, clean),
    join(OUT, `${clean}.html`),
    join(OUT, clean, "index.html"),
  ];
}

function serve() {
  const server = createServer((req, res) => {
    const found = candidates(decodeURIComponent(req.url))
      .find((p) => existsSync(p) && statSync(p).isFile());
    if (!found) { res.writeHead(404).end("not found"); return; }
    res.writeHead(200, { "content-type": TYPES[extname(found)] ?? "application/octet-stream" });
    createReadStream(found).pipe(res);
  });
  return new Promise((r) => server.listen(0, () => r(server)));
}

async function main() {
  const skipBuild = process.argv.includes("--skip-build");

  if (!skipBuild) {
    log("building static export ...");
    await run("npx", ["next", "build"], {
      NEXT_PUBLIC_STATIC: "1", NEXT_OUTPUT: "export", NEXT_PUBLIC_API: "",
    });
  }

  if (!existsSync(OUT)) { fail(`no export at ${OUT}`); return; }

  // 1. Every route emitted a file.
  const ROUTES = await discoverRoutes();
  log(`\nchecking emitted files for ${ROUTES.length} discovered route(s)`);
  for (const route of ROUTES) {
    const target = route === "/" ? join(OUT, "index.html")
                                 : join(OUT, route.replace(/^\//, ""), "index.html");
    if (existsSync(target)) log(`  ok    ${route}  ->  ${target.slice(ROOT.length + 1)}`);
    else fail(`${route} produced no index.html (expected ${target})`);
  }

  // 2. Nothing that cannot work under `output: export` slipped in.
  log("\nchecking for export-incompatible output");
  const appDir = join(ROOT, "app");
  const { readdir } = await import("node:fs/promises");
  const walk = async (dir) => {
    const out = [];
    for (const e of await readdir(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) out.push(...await walk(p));
      else out.push(p);
    }
    return out;
  };
  if (existsSync(appDir)) {
    const files = await walk(appDir);
    for (const f of files) {
      const rel = f.slice(ROOT.length + 1);
      if (/\[[^\]]+\]/.test(rel)) {
        fail(`dynamic route segment ${rel} - unsupported without generateStaticParams, `
           + "and it would couple the frontend build to backend/races/");
      }
      if (/[\\/]route\.(t|j)sx?$/.test(rel)) {
        fail(`Route Handler ${rel} - unsupported under output: export`);
      }
      if (/\.(t|j)sx?$/.test(rel)) {
        const src = await readFile(f, "utf8");
        if (src.includes("useSearchParams")) {
          fail(`${rel} uses useSearchParams - forces a CSR bailout and errors the `
             + "export build without a Suspense boundary. Use lib/urlState.ts.");
        }
        if (src.includes('from "next/image"')) {
          fail(`${rel} imports next/image - its default loader is unsupported under export`);
        }
      }
    }
  }

  // 3. Serve it and fetch every route, the way the host will.
  log("\nfetching each route over HTTP");
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const route of ROUTES) {
      const res = await fetch(`${base}${route}`);
      const body = await res.text();
      if (!res.ok) { fail(`GET ${route} -> ${res.status}`); continue; }
      if (!body.includes("<html") && !body.includes("<!DOCTYPE")) {
        fail(`GET ${route} returned no HTML document`);
        continue;
      }
      // What the *static* HTML can honestly guarantee. Every page here is
      // client-rendered, so headings and content appear after hydration and
      // asserting on them belongs to the Playwright pass, not to this file.
      // What must be true of the served bytes is that the shell is intact:
      // a titled document with the navigation in it. A page whose export is an
      // empty <body> looks fine to `next build` and blank to a judge.
      const problems = [];
      if (!/<title[\s>]/.test(body)) problems.push("no <title>");
      if (!/<nav[\s>]/.test(body)) problems.push("no <nav> landmark");
      if (body.length < 1000) problems.push(`suspiciously small (${body.length} bytes)`);
      if (problems.length) fail(`${route}: ${problems.join(", ")}`);
      else log(`  ok    ${route}  ${res.status}  ${body.length} bytes`);
    }
  } finally {
    server.close();
  }

  if (process.exitCode) console.error("\nexport verification FAILED");
  else log("\nexport verification passed");
}

main().catch((e) => { console.error(e); process.exit(1); });
