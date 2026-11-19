import type { Paragraph, PhrasingContent, Root, Text } from 'mdast'
import type { ContainerDirective } from 'mdast-util-directive'
import { visit } from 'unist-util-visit'

const ENVIRONMENTS = {
  definition: { label: '定义', numbered: true },
  theorem: { label: '定理', numbered: true },
  lemma: { label: '引理', numbered: true },
  proof: { label: '证明', numbered: false }
} as const

type EnvironmentName = keyof typeof ENVIRONMENTS

function span(className: string, children: PhrasingContent[]): PhrasingContent {
  return {
    type: 'strong',
    data: { hName: 'span', hProperties: { className: [className] } },
    children
  }
}

function heading(label: string, title: PhrasingContent[]): Paragraph {
  const children: PhrasingContent[] = [
    span('theorem-environment__label', [{ type: 'text', value: label }])
  ]

  if (title.length) children.push(span('theorem-environment__title', title))

  return {
    type: 'paragraph',
    data: { hName: 'div', hProperties: { className: ['theorem-environment__heading'] } },
    children
  }
}

function proofEnd(): Paragraph {
  const symbol: Text = { type: 'text', value: '□' }
  return {
    type: 'paragraph',
    data: {
      hName: 'div',
      hProperties: { className: ['theorem-environment__proof-end'], ariaHidden: 'true' }
    },
    children: [symbol]
  }
}

export default function remarkTheoremEnvironments() {
  return (tree: Root) => {
    const counters: Record<EnvironmentName, number> = {
      definition: 0,
      theorem: 0,
      lemma: 0,
      proof: 0
    }

    visit(tree, 'containerDirective', (node: ContainerDirective) => {
      if (!(node.name in ENVIRONMENTS)) return

      const name = node.name as EnvironmentName
      const environment = ENVIRONMENTS[name]
      const labelNode = node.children[0]
      const title =
        labelNode?.type === 'paragraph' && labelNode.data?.directiveLabel === true
          ? labelNode.children
          : []

      if (title.length) node.children.shift()
      if (environment.numbered) counters[name] += 1

      const number = environment.numbered ? ` ${counters[name]}` : ''
      node.children.unshift(heading(`${environment.label}${number}`, title))
      if (name === 'proof') node.children.push(proofEnd())

      const data = node.data || (node.data = {})
      data.hName = 'section'
      data.hProperties = {
        className: ['theorem-environment', `theorem-environment--${name}`],
        ...(node.attributes?.id ? { id: node.attributes.id } : {})
      }
    })
  }
}
