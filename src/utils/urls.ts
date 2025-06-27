// Utility functions for URL generation

// when using a relative base, import.meta.env.BASE_URL will be '/./'
export const getBaseUrl = () => {
	if (import.meta.env.BASE_URL === '/./' || import.meta.env.BASE_URL === '/') {
		return '';
	}
	return import.meta.env.BASE_URL;
}; 