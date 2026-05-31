/**
 * POST /api/chat — OpenAI-compatible proxy (secrets in env only).
 */

const { applyCors, handleOptions, jsonError } = require('./_cors');
const { assertClientAuthorized } = require('./_auth');
const { fetchWithTimeout } = require('./_fetch');

function bad(res, status, msg) {
  jsonError(res, status, msg);
}

function sanitizeMessages(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const m of raw.slice(0, 48)) {
    if (!m || typeof m !== 'object') continue;
    const role = m.role;
    if (!['system', 'user', 'assistant'].includes(role)) continue;
    let content = m.content;
    if (typeof content !== 'string') content = String(content ?? '');
    if (content.length > 48000) content = content.slice(0, 48000);
    out.push({ role, content });
  }
  return out;
}

function refererForOpenRouter() {
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}`;
  return process.env.OPENROUTER_SITE_URL || 'https://jonobenjamin.github.io';
}

async function readJsonBody(req) {
  if (req.body != null && typeof req.body === 'object' && !Buffer.isBuffer(req.body)) {
    return req.body;
  }
  const raw = typeof req.body === 'string' ? req.body : await new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
  if (!raw || String(raw).trim() === '') return {};
  return JSON.parse(raw);
}

module.exports = async function handler(req, res) {
  applyCors(req, res);

  try {
    if (req.method === 'OPTIONS') return handleOptions(req, res);
    if (req.method !== 'POST') return bad(res, 405, 'Method not allowed');
    if (!assertClientAuthorized(req, res)) return;

    const apiKey = process.env.AI_API_KEY;
    if (!apiKey || String(apiKey).trim() === '') {
      return bad(res, 401, 'Server not configured: AI_API_KEY is missing');
    }

    let body;
    try {
      body = await readJsonBody(req);
    } catch {
      return bad(res, 400, 'Invalid JSON body');
    }

    const messages = sanitizeMessages(body.messages);
    if (!messages.length) return bad(res, 400, 'Missing or invalid messages array');

    const apiUrl =
      (process.env.AI_API_URL && String(process.env.AI_API_URL).trim()) ||
      'https://openrouter.ai/api/v1/chat/completions';
    const model =
      (process.env.AI_MODEL && String(process.env.AI_MODEL).trim()) || 'openai/gpt-4o-mini';

    const headers = {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
    if (apiUrl.includes('openrouter.ai')) {
      headers['HTTP-Referer'] = refererForOpenRouter();
      headers['X-Title'] = 'Intervals Coach';
    }

    const upstream = await fetchWithTimeout(
      apiUrl,
      {
        method: 'POST',
        headers,
        body: JSON.stringify({ model, messages })
      },
      55000
    );

    const text = await upstream.text();
    if (!upstream.ok) {
      return bad(res, upstream.status >= 500 ? 502 : upstream.status, text || upstream.statusText);
    }

    let data;
    try {
      data = JSON.parse(text);
    } catch {
      return bad(res, 502, 'Invalid JSON from chat provider');
    }

    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    return res.status(200).json(data);
  } catch (e) {
    console.error('chat proxy', e);
    const code = typeof e.statusCode === 'number' ? e.statusCode : 502;
    return bad(res, code >= 400 && code < 600 ? code : 502, e.message || 'Chat proxy failed');
  }
};
