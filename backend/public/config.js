/**
 * Vercel-hosted version — API is on the same domain, so API_BASE is just '/api'.
 * Do NOT put secrets here. INTERVALS_API_KEY and AI_API_KEY live in Vercel env vars only.
 */
window.APP_CONFIG = {
  API_BASE: '/api',
  /** Optional site gate — must match PUBLIC_API_TOKEN in Vercel env vars (leave empty if unused). */
  PUBLIC_API_TOKEN: '',
  VERCEL_PROTECTION_BYPASS: ''
};
