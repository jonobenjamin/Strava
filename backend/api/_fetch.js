/** fetch with timeout so serverless handlers do not hang. */
async function fetchWithTimeout(url, options, timeoutMs) {
  const ms = timeoutMs || 45000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (e) {
    if (e && e.name === 'AbortError') {
      const err = new Error(`Upstream request timed out after ${ms}ms`);
      err.statusCode = 504;
      throw err;
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

module.exports = { fetchWithTimeout };
