// Utility functions for URL generation - always use blog.trustyai.org
const SITE_URL = 'https://blog.trustyai.org';

export function getImageUrl(heroImage: string): string {
	if (!heroImage) return '';
	return heroImage.startsWith('/') ? `${SITE_URL}${heroImage}` : heroImage;
}

export function getPostUrl(post: any): string {
	if (post.data.track === 'research') {
		return `${SITE_URL}/research/${post.id.replace('research/', '')}`;
	} else if (post.data.track === 'engineering') {
		return `${SITE_URL}/engineering/${post.id.replace('engineering/', '')}`;
	} else {
		// For posts not in subdirectories
		return `${SITE_URL}/blog/${post.id}`;
	}
}

export function getAuthorUrl(authorId: string): string {
	return `${SITE_URL}/authors/${authorId}/`;
}

export function getAssetUrl(path: string): string {
	return `${SITE_URL}${path}`;
}

export function getBlogPostUrl(postId: string): string {
	return `${SITE_URL}/blog/${postId}/`;
} 