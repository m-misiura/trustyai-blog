// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { visit } from 'unist-util-visit';

// Determine if we're building for GitHub Pages (with subdirectory) or custom domain
const isGitHubPages = process.env.GITHUB_PAGES === 'true';
const baseUrl = isGitHubPages ? '/trustyai-blog' : '';

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
			if (node.value && node.value.includes('<img') && node.value.includes('src="/') && !node.value.includes('src="/trustyai-blog/')) {
				console.log(`Transforming HTML img tag: ${node.value}`);
				node.value = node.value.replace(/src="\/([^"]+)"/g, `src="${baseUrl}/$1"`);
			}
		});
	};
}

// https://astro.build/config
export default defineConfig({
	site: 'https://blog.trustyai.org',
	base: baseUrl,
	integrations: [
		mdx({
			remarkPlugins: [remarkBaseUrl],
		}), 
		sitemap()
	],
	markdown: {
		remarkPlugins: [remarkBaseUrl],
	},
});
