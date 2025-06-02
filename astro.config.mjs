// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { visit } from 'unist-util-visit';

// Custom remark plugin to handle base URL for images in markdown content
function remarkBaseUrl() {
	// @ts-ignore
	return (tree) => {
		visit(tree, 'image', (node) => {
			if (node.url && node.url.startsWith('/') && !node.url.startsWith('/trustyai-blog/')) {
				console.log(`Transforming image URL: ${node.url} -> /trustyai-blog${node.url}`);
				node.url = `/trustyai-blog${node.url}`;
			}
		});
		// Also handle HTML img tags in MDX
		visit(tree, 'html', (node) => {
			if (node.value && node.value.includes('<img') && node.value.includes('src="/') && !node.value.includes('src="/trustyai-blog/')) {
				console.log(`Transforming HTML img tag: ${node.value}`);
				node.value = node.value.replace(/src="\/([^"]+)"/g, 'src="/trustyai-blog/$1"');
			}
		});
	};
}

// https://astro.build/config
export default defineConfig({
	site: 'https://trustyai-explainability.github.io',
	base: '/trustyai-blog',
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
