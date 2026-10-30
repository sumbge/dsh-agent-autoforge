// 复刻 desktop 的 xdpool 模型路由：让别的 profile 也能走同一条已登录的模型通道。
//
// dsh-xdpool 没装在 dsh 安装目录里，只装在 desktop profile 下，所以按绝对
// file URL 解析；找不到就回落到裸包名（装在 profile 里时能解析）。
import { pathToFileURL } from 'node:url'
import { existsSync } from 'node:fs'

const CANDIDATES = [
  'C:/Users/32535/.dsh/profiles/desktop/node_modules/dsh-xdpool/lib/index.js',
  'C:/Users/32535/.dsh/profiles/desktop/node_modules/dsh-xdpool/index.js',
]

export const name = 'xdpool-route'
export const inject = ['tools']

/** 模块命名空间本身不是插件；有 apply 就用它，否则用 default。 */
function pick(mod) {
  // ESM 命名空间是不可扩展的 [object Module]，cordis 会判成「不是插件」。
  // 抽成普通对象：{ name, inject, apply }。
  if (mod && typeof mod === 'object' && typeof mod.apply === 'function' && Object.prototype.toString.call(mod) === '[object Module]') {
    return { name: mod.name, inject: mod.inject, apply: mod.apply }
  }
  // 有些包在 ESM 下是 default.default 双包裹，cordis 只认函数或带 apply 的对象。
  let m = mod
  for (let i = 0; i < 3; i += 1) {
    if (typeof m === 'function' || (m && typeof m.apply === 'function')) return m
    if (!m || !m.default) break
    m = m.default
  }
  return m
}

export function apply(ctx, config = {}) {
  ctx.effect(() => {
    let target = 'dsh-xdpool'
    const hit = CANDIDATES.find((candidate) => existsSync(candidate))
    if (hit) target = pathToFileURL(hit).href
    // ctx.plugin 要的是插件本身，不是 Promise：先 await 再交给它。
    let handle
    let disposed = false
    import(target)
      .then(pick)
      .then((plugin) => {
        if (disposed) return
        handle = ctx.plugin(plugin, config)
      })
      .catch((err) => {
        ctx.logger?.('xdpool-route')?.warn?.('xdpool 加载失败：', err)
      })
    return () => {
      disposed = true
      handle?.()
    }
  })
}
