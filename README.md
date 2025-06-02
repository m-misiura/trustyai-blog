# TrustyAI Blog

A blog for TrustyAI covering research insights and engineering best practices for trustworthy AI systems.

## Development

Install dependencies:
```sh
npm install
```

Start the development server:
```sh
npm run dev
```

The site will be available at `http://localhost:4321`

## Build & Preview

Build the site:
```sh
npm run build
```

Preview the built site locally:
```sh
npm run preview
```

## Deployment

The site automatically deploys to GitHub Pages when you push to the `main` branch.

**Live site**: https://trustyai-explainability.github.io/trustyai-blog/

## Adding Posts

Create new `.md` files in `src/content/blog/` with frontmatter:

```yaml
---
title: 'Post Title'
description: 'Post description'
pubDate: 'Dec 01 2024'
heroImage: '/blog-placeholder-1.jpg'
track: 'research' # or 'engineering'
---
```
