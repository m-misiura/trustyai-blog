// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { visit } from 'unist-util-visit';

// Always use the full site URL for absolute links
const siteUrl = 'https://blog.trustyai.org';
const baseUrl = siteUrl;

console.log('Build environment:', {
	site: siteUrl,
	baseUrl: baseUrl,
	NODE_ENV: process.env.NODE_ENV
});

// Custom remark plugin to handle base URL for images in markdown content
function remarkBaseUrl() {
	// @ts-ignore
	return (tree) => {
		visit(tree, 'image', (node) => {
			if (node.url && node.url.startsWith('/') && !node.url.startsWith(siteUrl)) {
				console.log(`Transforming image URL: ${node.url} -> ${siteUrl}${node.url}`);
				node.url = `${siteUrl}${node.url}`;
			}
		});
		// Also handle HTML img tags in MDX
		visit(tree, 'html', (node) => {
			if (node.value && node.value.includes('<img')) {
				node.value = node.value.replace(
					/src="(\/[^"]*?)"/g,
					(match, src) => {
						if (!src.startsWith(siteUrl)) {
							const newSrc = `${siteUrl}${src}`;
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
	site: siteUrl,
	base: baseUrl,
	integrations: [mdx(), sitemap()],
	markdown: {
		remarkPlugins: [remarkBaseUrl],
	},
});
