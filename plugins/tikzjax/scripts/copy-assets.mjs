import fs from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const source = path.join(root, "node_modules", "node-tikzjax", "tex")
const target = path.join(root, "tex")

await fs.rm(target, { recursive: true, force: true })
await fs.cp(source, target, { recursive: true })
