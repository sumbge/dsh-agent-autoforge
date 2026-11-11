// Resolve hook: map the two host-provided peer packages to the host's own tree.
//
// lib/index.js imports '@deepseek-ai/dsh-tools' and '@deepseek-ai/cordis', and
// nothing in the plugin's own tree provides them -- the running host does. A
// test that imports the plugin must supply the same resolution or node fails
// with "Cannot find package '@deepseek-ai/dsh-tools'".
//
// The host packages live in the app's extracted tree. That location is a
// property of this machine, so it is probed, not assumed: no candidate means
// the caller reports UNPROVABLE rather than passing on weaker evidence.
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

export const CANDIDATES = [
  process.env.DSH_HOST_NODE_MODULES,
  String.raw`C:\Users\sumbger\.dsh\im\asar-extract\dsh\node_modules`,
  String.raw`C:\cc\DSH\resources\app.asar.unpacked\node_modules`,
  String.raw`C:\cc\DSH\resources\app\dsh\node_modules`,
].filter(Boolean)

export function hostNodeModules() {
  for (const dir of CANDIDATES) {
    if (fs.existsSync(path.join(dir, '@deepseek-ai', 'dsh-tools', 'lib', 'index.js'))) return dir
  }
  return null
}

const ROOTS = new Map([
  ['@deepseek-ai/dsh-tools', '@deepseek-ai/dsh-tools/lib/index.js'],
  ['@deepseek-ai/cordis', '@deepseek-ai/cordis/lib/index.js'],
])

const dir = hostNodeModules()
const base = dir === null ? null : pathToFileURL(dir + path.sep).href

export async function resolve(specifier, context, next) {
  const rel = ROOTS.get(specifier)
  if (rel !== undefined && base !== null) return next(new URL(rel, base).href, context)
  return next(specifier, context)
}
