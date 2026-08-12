"use client";

/* Theme toggle.
 *
 * Dark is the default because this is a pit-wall console, but the OS preference
 * is honoured when nothing has been pinned, and an explicit choice always wins
 * over the OS in both directions (see the :not([data-theme=...]) guards in
 * globals.css).
 *
 * The stamp is applied by an inline script in the document head before paint —
 * doing it here in an effect would flash the wrong theme on every load.
 */

import { useSyncExternalStore } from "react";

type Theme = "dark" | "light" | "system";

const KEY = "pitwall-theme";

export const themeInitScript = `
(function () {
  try {
    var t = localStorage.getItem(${JSON.stringify(KEY)});
    if (t === 'dark' || t === 'light') {
      document.documentElement.setAttribute('data-theme', t);
    }
  } catch (e) {}
})();
`;

/* The current theme lives on the <html> element, which the init script already
 * stamped before React mounted. That makes it external state, so we subscribe to
 * it rather than mirroring it into React state in an effect - which would both
 * trip the set-state-in-effect rule and risk a hydration mismatch. */
function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  return () => observer.disconnect();
}

function getSnapshot(): Theme {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "dark" || t === "light" ? t : "system";
}

export default function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, () => "system" as Theme);

  function apply(next: Theme) {
    const root = document.documentElement;
    if (next === "system") {
      root.removeAttribute("data-theme");
      localStorage.removeItem(KEY);
    } else {
      root.setAttribute("data-theme", next);
      localStorage.setItem(KEY, next);
    }
  }

  const next: Theme = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  const label = theme === "system" ? "Auto" : theme === "dark" ? "Dark" : "Light";

  return (
    <button
      className="chip"
      onClick={() => apply(next)}
      title={`Theme: ${label}. Click for ${next}.`}
      aria-label={`Theme: ${label}. Switch to ${next}.`}
      style={{ padding: "5px 11px", fontSize: 12 }}
    >
      <span aria-hidden>{theme === "light" ? "☀" : theme === "dark" ? "☾" : "◐"}</span>
      <span style={{ marginLeft: 6 }}>{label}</span>
    </button>
  );
}
