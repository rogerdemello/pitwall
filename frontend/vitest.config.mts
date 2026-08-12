import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Only the pure modules. Component rendering is covered by the Playwright
    // smoke pass instead: installing jsdom and RTL to assert on three
    // presentational components is hours of config for coverage nobody reads,
    // while `tsc --noEmit` and a real export check catch more.
    include: ["lib/**/*.test.ts"],
    environment: "node",
  },
});
