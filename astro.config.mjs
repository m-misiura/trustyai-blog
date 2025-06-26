// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { visit } from 'unist-util-visit';

// Determine if we're building for GitHub Pages (with subdirectory) or custom domain
const isGitHubPages = process.env.GITHUB_PAGES === 'true';
const baseUrl = isGitHubPages ? '/trustyai-blog' : '';

console.log('Build environment:', {
	GITHUB_PAGES: process.env.GITHUB_PAGES,
	isGitHubPages,
	baseUrl,
	NODE_ENV: process.env.NODE_ENV
});

// Custom remark plugin to handle base URL for images in markdown content
function remarkBaseUrl() {
	// @ts-ignore
	return (tree) => {
		visit(tree, 'image', (node) => {
			if (node.url && node.url.startsWith('/') && !node.url.startsWith('/trustyai-blog/')) {
				console.log(`Transforming image URL: ${node.url} -> ${baseUrl}${node.url}`);
				node.url = `${baseUrl}${node.url}`;
			}
		});
		// Also handle HTML img tags in MDX
		visit(tree, 'html', (node) => {
			if (node.value && node.value.includes('<img')) {
				node.value = node.value.replace(
					/src="(\/[^"]*?)"/g,
					(match, src) => {
						if (!src.startsWith('/trustyai-blog/')) {
							const newSrc = `${baseUrl}${src}`;
							console.log(`Transforming HTML img src: ${src} -> ${newSrc}`);
							return `src="${newSrc}"`;
						}
						return match;
					}
				);
			}
		});
	};
}

// https://astro.build/config
export default defineConfig({
	site: 'https://blog.trustyai.org',
	base: baseUrl,
	integrations: [mdx(), sitemap()],
	markdown: {
		remarkPlugins: [remarkBaseUrl],
	},
});
