import type { Parent, Root, Text } from 'mdast'
import { visit } from 'unist-util-visit'

const MARK_PATTERN = /==([^=\n](?:.*?[^=\n])?)==/g

function escapeHtml(value: string) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

export default function remarkMark() {
  return (tree: Root) => {
    visit(tree, 'text', (node: Text, index, parent: Parent | undefined) => {
      if (index === undefined || !parent || !node.value.includes('==')) return

      const children: Parent['children'] = []
      let cursor = 0

      for (const match of node.value.matchAll(MARK_PATTERN)) {
        const start = match.index
        if (start > cursor) children.push({ type: 'text', value: node.value.slice(cursor, start) })
        children.push({ type: 'html', value: `<mark>${escapeHtml(match[1])}</mark>` })
        cursor = start + match[0].length
      }

      if (cursor === 0) return
      if (cursor < node.value.length)
        children.push({ type: 'text', value: node.value.slice(cursor) })
      parent.children.splice(index, 1, ...children)
      return index + children.length
    })
  }
}
