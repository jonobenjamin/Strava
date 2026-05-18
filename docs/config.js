/**
 * Frontend UI config only — NO secrets for intervals.icu or OpenRouter here.
 * Set API_BASE to your Vercel API URL (backend/ project).
 */
window.APP_CONFIG = {
  API_BASE: 'https://strava-29tepzeat-jonobenjamins-projects.vercel.app/api',
  /** Same as PUBLIC_API_TOKEN on Vercel (site gate, not intervals.icu key). */
  PUBLIC_API_TOKEN: '',
  /** Only if Vercel Deployment Protection is enabled. */
  VERCEL_PROTECTION_BYPASS: ''
};
