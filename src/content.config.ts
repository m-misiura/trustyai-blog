import { glob } from 'astro/loaders';
import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
	// Load Markdown and MDX files in the `src/content/blog/` directory.
	loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
	// Type-check frontmatter using a schema
	schema: z.object({
		title: z.string(),
		description: z.string(),
		// Transform string to Date object
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		heroImage: z.string().optional(),
		track: z.enum(['research', 'engineering', 'community']),
		authors: z.array(z.string()).optional(),
	}),
});

const authors = defineCollection({
	// Load Markdown and MDX files in the `src/content/authors/` directory.
	loader: glob({ base: './src/content/authors', pattern: '**/*.{md,mdx}' }),
	// Type-check frontmatter using a schema
	schema: z.object({
		name: z.string(),
		bio: z.string().optional(),
		avatar: z.string().optional(),
		email: z.string().optional(),
		fosstodon: z.string().optional(),
		bluesky: z.string().optional(),
		linkedin: z.string().optional(),
		github: z.string().optional(),
		website: z.string().optional(),
	}),
});

export const collections = { blog, authors };
