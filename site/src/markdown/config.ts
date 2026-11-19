import { rehypeHeadingIds, type AstroMarkdownOptions } from '@astrojs/markdown-remark'
import type { Element, Root } from 'hast'
import rehypeKatex from 'rehype-katex'
import remarkDirective from 'remark-directive'
import remarkMath from 'remark-math'
import { SKIP, visit } from 'unist-util-visit'
import remarkMark from './remark-mark'
import remarkTheoremEnvironments from './remark-theorem-environments'

const text = (value: string) => ({ type: 'text' as const, value })

function rehypeReadingBlocks() {
  return (tree: Root) => {
    visit(tree, 'element', (node, index, parent) => {
      if (index === undefined || !parent) return
      if (node.tagName === 'table') {
        parent.children[index] = {
          type: 'element', tagName: 'div', properties: { className: ['reading-table'], tabIndex: 0, role: 'region', ariaLabel: '表格' }, children: [node]
        }
        return SKIP
      }
      if (node.tagName !== 'pre') return
      const code = node.children.find((child): child is Element => child.type === 'element' && child.tagName === 'code')
      if (!code) return
      const lang = node.properties.dataLanguage ?? (Array.isArray(code.properties.className) && code.properties.className.find((name) => typeof name === 'string' && name.startsWith('language-'))?.toString().slice(9))
      const toolbar: Element = {
        type: 'element', tagName: 'div', properties: { className: ['code-toolbar'] }, children: [
          { type: 'element', tagName: 'span', properties: {}, children: [text(String(lang || 'CODE'))] },
          { type: 'element', tagName: 'button', properties: { type: 'button', dataCopyCode: '', ariaLabel: '复制代码' }, children: [text('复制')] }
        ]
      }
      parent.children[index] = { type: 'element', tagName: 'div', properties: { className: ['code-block'] }, children: [toolbar, node] }
      return SKIP
    })
  }
}

export const markdown = {
  remarkPlugins: [remarkDirective, remarkTheoremEnvironments, remarkMath, remarkMark],
  rehypePlugins: [rehypeKatex, rehypeHeadingIds, rehypeReadingBlocks],
  smartypants: false,
  gfm: true,
  shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } }
} satisfies AstroMarkdownOptions
