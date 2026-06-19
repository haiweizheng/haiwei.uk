import { defineConfig } from "tsup"
import { readFile } from "node:fs/promises"
import { dirname } from "node:path"
import type { Plugin } from "esbuild"

const jsdomPatch: Plugin = {
  name: "jsdom-patch",
  setup(build) {
    build.onLoad({ filter: /XMLHttpRequest-impl\.js$/ }, async (args) => {
      let contents = await readFile(args.path, "utf8")
      contents = contents.replace(
        'const syncWorkerFile = require.resolve ? require.resolve("./xhr-sync-worker.js") : null;',
        "const syncWorkerFile = null;",
      )
      return { contents, loader: "js", resolveDir: dirname(args.path) }
    })
  },
}

export default defineConfig({
  entry: {
    index: "src/index.ts",
  },
  format: ["esm"],
  dts: true,
  tsconfig: "tsconfig.json",
  sourcemap: false,
  clean: true,
  minify: true,
  treeshake: true,
  target: "es2022",
  splitting: false,
  outDir: "dist",
  platform: "node",
  noExternal: [/^(?!@quartz-community\/)/],
  external: ["@quartz-community/types"],
  banner: {
    js: 'import { createRequire } from "node:module"; import { fileURLToPath } from "node:url"; import { dirname } from "node:path"; const require = createRequire(import.meta.url); const __filename = fileURLToPath(import.meta.url); const __dirname = dirname(__filename);',
  },
  esbuildPlugins: [jsdomPatch],
})
