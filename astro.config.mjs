// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { visit } from 'unist-util-visit';

// Custom remark plugin to handle base URL for images in markdown content
function remarkBaseUrl() {
	// @ts-ignore
	return (tree) => {
		const baseUrl = process.env.NODE_ENV === 'production' && process.env.GITHUB_PAGES 
			? '/trustyai-blog' 
			: '';
			
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
	base: process.env.NODE_ENV === 'production' && process.env.GITHUB_PAGES ? '/trustyai-blog' : '/',
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
