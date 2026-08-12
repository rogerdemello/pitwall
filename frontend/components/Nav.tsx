"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { STATIC_MODE } from "@/lib/api";

/* Hugging Face's static host serves exact file paths only: it resolves
 * index.html at the root but NOT inside subdirectories, so a link to
 * "/evidence/" 404s even though out/evidence/index.html exists - and the
 * browser then falls back to huggingface.co, leaving the Space entirely.
 *
 * So static builds link at the file itself. The URLs are less tidy, but they
 * work, which matters more. Server builds keep the clean paths. */
const LINKS = [
  { key: "/", label: "Race Replay" },
  { key: "/evidence", label: "Evidence" },
  { key: "/live", label: "Live Analysis" },
];

function href(key: string): string {
  if (!STATIC_MODE) return key;
  return key === "/" ? "/index.html" : `${key}/index.html`;
}

export default function Nav() {
  const path = usePathname();

  return (
    <nav className="nav">
      {LINKS.map((l) => {
        // usePathname yields "/evidence" even when the URL ends in
        // /index.html, so match on the logical key rather than the href.
        const active =
          l.key === "/" ? path === "/" : path.startsWith(l.key);
        return (
          <Link key={l.key} href={href(l.key)} className={active ? "active" : ""}>
            {l.label}
          </Link>
        );
      })}
    </nav>
  );
}
