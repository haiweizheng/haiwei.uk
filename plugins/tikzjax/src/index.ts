import { createHash } from "node:crypto"
import fs from "node:fs/promises"
import path from "node:path"
import process from "node:process"
import tikzjaxModule from "node-tikzjax"
import type { Code, Root } from "mdast"
import { visit } from "unist-util-visit"
import type { QuartzTransformerPlugin } from "@quartz-community/types"

type TexPackages = Record<string, string>
type Tex2Svg = (source: string, opts: Record<string, unknown>) => Promise<string>

export interface TikzJaxOptions {
  texPackages?: TexPackages
  tikzLibraries?: string
  addToPreamble?: string
  embedFontCss?: boolean
  fontCssUrl?: string
  disableOptimize?: boolean
}

const defaultOptions: Required<Omit<TikzJaxOptions, "addToPreamble">> &
  Pick<TikzJaxOptions, "addToPreamble"> = {
  texPackages: {
    amsmath: "intlimits",
    amssymb: "",
    amsfonts: "",
  },
  tikzLibraries: "arrows.meta,calc,positioning,matrix,decorations.pathreplacing",
  addToPreamble: undefined,
  embedFontCss: true,
  fontCssUrl: "https://cdn.jsdelivr.net/npm/node-tikzjax@1.0.5/css/fonts.css",
  disableOptimize: true,
}

let renderQueue = Promise.resolve()

function getTex2Svg(): Tex2Svg {
  const direct = tikzjaxModule as unknown
  if (typeof direct === "function") {
    return direct as Tex2Svg
  }

  const firstDefault = (direct as { default?: unknown }).default
  if (typeof firstDefault === "function") {
    return firstDefault as Tex2Svg
  }

  const secondDefault = (firstDefault as { default?: unknown } | undefined)?.default
  if (typeof secondDefault === "function") {
    return secondDefault as Tex2Svg
  }

  throw new Error("Could not find node-tikzjax renderer.")
}

const tex2svg = getTex2Svg()

function runOneAtATime<T>(fn: () => Promise<T>): Promise<T> {
  const result = renderQueue.then(fn, fn)
  renderQueue = result.then(
    () => undefined,
    () => undefined,
  )
  return result
}

function normalizeTikzSource(source: string): string {
  const trimmed = source.trim()
  if (trimmed.includes("\\begin{document}")) {
    return trimmed
  }

  return `\\begin{document}
${trimmed}
\\end{document}`
}

function svgDataUrl(svg: string): string {
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`
}

async function cachedRender(source: string, opts: TikzJaxOptions): Promise<string> {
  const cacheDir = path.join(process.cwd(), ".quartz-cache", "tikzjax")
  const hash = createHash("sha256")
    .update(source)
    .update(JSON.stringify(opts))
    .digest("hex")
  const cachePath = path.join(cacheDir, `${hash}.svg`)

  try {
    return await fs.readFile(cachePath, "utf8")
  } catch {
    const svg = await runOneAtATime(() =>
      tex2svg(source, {
        texPackages: opts.texPackages,
        tikzLibraries: opts.tikzLibraries,
        addToPreamble: opts.addToPreamble,
        embedFontCss: opts.embedFontCss,
        fontCssUrl: opts.fontCssUrl,
        disableOptimize: opts.disableOptimize,
      }),
    )

    await fs.mkdir(cacheDir, { recursive: true })
    await fs.writeFile(cachePath, svg, "utf8")
    return svg
  }
}

function tikzRemarkPlugin(opts: TikzJaxOptions) {
  return async (tree: Root) => {
    const renders: Promise<void>[] = []

    visit(tree, "code", (node: Code) => {
      if (node.lang !== "tikz") return

      renders.push(
        (async () => {
          const source = normalizeTikzSource(node.value)
          const svg = await cachedRender(source, opts)
          const html = `<figure class="tikzjax"><img class="tikzjax__image" src="${svgDataUrl(
            svg,
          )}" alt="TikZ diagram" loading="lazy" decoding="async"></figure>`

          Object.assign(node, {
            type: "html",
            value: html,
          })
        })(),
      )
    })

    await Promise.all(renders)
  }
}

export const TikzJax: QuartzTransformerPlugin<Partial<TikzJaxOptions>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "TikzJax",
    markdownPlugins() {
      return [[tikzRemarkPlugin, opts]]
    },
    externalResources() {
      return {
        css: [
          {
            content: `
figure.tikzjax {
  margin: 1.5rem auto;
  text-align: center;
  overflow-x: auto;
}

figure.tikzjax img.tikzjax__image {
  max-width: 100%;
  height: auto;
}

`,
            inline: true,
          },
        ],
      }
    },
  }
}

export default TikzJax
