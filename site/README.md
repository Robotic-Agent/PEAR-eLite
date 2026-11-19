# PEAR-eLite 项目站

项目站独立于 Python 训练环境。页面沿用 `pure-astro-blog` 的阅读设计：低对比网格、克制的靛蓝配色、随系统变化的浅色/深色模式、中文正文与等宽信息标签、文章目录及代码工具栏。高亮与定理块 Markdown 插件从该仓库迁入，沿用 Apache-2.0 许可；项目图标直接使用 `docs/assets/` 里的 PEAR-eLite 标志。

## 本地写作

在仓库根目录执行（Node.js 22，pnpm 10）：

```sh
pnpm --dir site install --frozen-lockfile
pnpm --dir site dev
```

在本地编辑器里改 Markdown，浏览器打开对应文章页。保存文件后，Astro 自动更新正文和目录；预览就是正式读者看到的排版，不需要独立编辑页面。

| 页面 | 源文件 | 本地地址 |
| --- | --- | --- |
| 项目文章 | `site/src/content/overview.md` | [/PEAR-eLite/](http://localhost:4321/PEAR-eLite/) |
| 训练配方 | `site/src/content/recipes.md` | [/PEAR-eLite/recipes/](http://localhost:4321/PEAR-eLite/recipes/) |
| 实验结果 | `site/src/content/results.md` | [/PEAR-eLite/results/](http://localhost:4321/PEAR-eLite/results/) |

支持 GFM 表格与脚注、Shiki 代码高亮、KaTeX（`$...$` / `$$...$$`）、`==高亮==`、`:::definition` / `:::theorem` / `:::lemma` / `:::proof`。首页的三个导读卡片由 `overview.md` 的 `sections` 字段生成，`heading` 对应正文二级标题。主图位于 `site/src/assets/pear-elite-overview.png`，正文数据更新时也同步检查导读与评测页。

```sh
pnpm --dir site check
pnpm --dir site build
pnpm --dir site preview
```

## GitHub Pages

目标地址为 `https://robotic-agent.github.io/PEAR-eLite/`。当前私有仓库使用 GitHub Free；须先为仓库获得私有仓库 Pages 资格（组织 GitHub Team / Enterprise），或经仓库所有者明确批准调整公开范围。私有代码仓库的 Pages 站点不等于私密站点，启用前按站点的实际可见性审核正文；原仓库里的详细文档不会被整批渲染或上传，站内 GitHub 链接仍受仓库访问权限限制。

具备资格后：合入站点改动 → **Settings → Pages → Build and deployment → Source: GitHub Actions** → **Actions → Project website → Run workflow**，选择 `main`。站点 PR 和 `main` 的相关推送只校验、构建；手动运行 `main` 才上传 `site/dist/` 并部署，不上传整个仓库。若独立迁至另一个 Pages 仓库，同步修改 `astro.config.ts` 的 `site`、`base`。
