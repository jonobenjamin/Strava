/**
 * Optional gate between public frontend and /api routes.
 * Set PUBLIC_API_TOKEN in Vercel env; frontend sends the same value (NOT intervals.icu keys).
 */

const { jsonError } = require('./_cors');

function readClientToken(req) {
  const auth = req.headers.authorization;
  if (typeof auth === 'string' && auth.toLowerCase().startsWith('bearer ')) {
    return auth.slice(7).trim();
  }
  const xKey = req.headers['x-api-key'];
  if (typeof xKey === 'string' && xKey.trim()) return xKey.trim();
  return '';
}

/** Returns true if request may proceed; on failure sends JSON 401 with CORS already applied. */
function assertClientAuthorized(req, res) {
  const expected = process.env.PUBLIC_API_TOKEN && String(process.env.PUBLIC_API_TOKEN).trim();
  if (!expected) return true;

  const token = readClientToken(req);
  if (token === expected) return true;

  jsonError(res, 401, 'Unauthorized');
  return false;
}

module.exports = { assertClientAuthorized, readClientToken };
