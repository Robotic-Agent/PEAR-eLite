import { defineConfig } from 'astro/config'
import { markdown } from './src/markdown/config'

export default defineConfig({
  site: 'https://robotic-agent.github.io',
  base: '/PEAR-eLite',
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  markdown
})
