/**
 * CORS for GitHub Pages → Vercel /api routes.
 * Apply before every response (including errors) so browsers see headers on 401/500.
 */

const ALLOWED_ORIGINS = new Set([
  'https://jonobenjamin.github.io',
  'http://localhost:3000',
  'http://127.0.0.1:3000',
  'http://localhost:5173',
  'http://127.0.0.1:5173'
]);

const ALLOW_HEADERS =
  'Content-Type, Accept, Authorization, x-api-key, x-vercel-protection-bypass';

function applyCors(req, res) {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', ALLOW_HEADERS);
  res.setHeader('Access-Control-Max-Age', '86400');
  res.setHeader('Vary', 'Origin');
}

function handleOptions(req, res) {
  applyCors(req, res);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.status(200).json({ ok: true });
}

function jsonError(res, status, message) {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.status(status).json({ error: message });
}

module.exports = { applyCors, handleOptions, jsonError, ALLOWED_ORIGINS };
