# TrustyAI Blog

A blog for TrustyAI covering research insights, engineering best practices, and community discussions for trustworthy AI systems.

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
track: 'research' # or 'engineering' or 'community'
---
```

### Track Categories

- **Research** (`track: 'research'`): Theoretical foundations and novel approaches to trustworthy AI
- **Engineering** (`track: 'engineering'`): Practical guides and best practices for building trustworthy AI systems  
- **Community** (`track: 'community'`): Insights, discussions, and contributions from the TrustyAI community

Posts should be placed in the appropriate subdirectory:
- Research posts: `src/content/blog/research/`
- Engineering posts: `src/content/blog/engineering/`
- Community posts: `src/content/blog/community/`
