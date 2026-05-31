/**
 * GET /api/intervals/activities — proxies intervals.icu (secrets in env only).
 */

const { applyCors, handleOptions, jsonError } = require('../_cors');
const { assertClientAuthorized } = require('../_auth');
const { fetchWithTimeout } = require('../_fetch');

function bad(res, status, msg) {
  jsonError(res, status, msg);
}

function ymdToUtcDate(ymd) {
  const parts = String(ymd).split('-').map(Number);
  if (parts.length !== 3 || parts.some((x) => !Number.isFinite(x))) return null;
  const [y, m, d] = parts;
  return new Date(Date.UTC(y, m - 1, d));
}

function dateToYmd(d) {
  return d.toISOString().slice(0, 10);
}

function subDays(d, n) {
  const x = new Date(d.getTime());
  x.setUTCDate(x.getUTCDate() - n);
  return x;
}

async function fetchActivitiesChunk({ apiKey, athleteId, oldestStr, newestStr, limitNum }) {
  const upstream = new URL(`https://intervals.icu/api/v1/athlete/${encodeURIComponent(athleteId)}/activities`);
  upstream.searchParams.set('oldest', oldestStr);
  upstream.searchParams.set('newest', newestStr);
  upstream.searchParams.set('limit', String(limitNum));

  const basic = Buffer.from(`API_KEY:${apiKey}`, 'utf8').toString('base64');
  const r = await fetchWithTimeout(
    upstream.toString(),
    {
      method: 'GET',
      headers: {
        Authorization: `Basic ${basic}`,
        Accept: 'application/json'
      }
    },
    45000
  );

  const text = await r.text();
  if (!r.ok) {
    const err = new Error(text || r.statusText || 'Intervals.icu request failed');
    err.statusCode = r.status;
    throw err;
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    const err = new Error('Invalid JSON from intervals.icu');
    err.statusCode = 502;
    throw err;
  }

  if (!Array.isArray(data)) {
    const err = new Error('Unexpected response shape from intervals.icu');
    err.statusCode = 502;
    throw err;
  }

  return data;
}

module.exports = async function handler(req, res) {
  applyCors(req, res);

  try {
    if (req.method === 'OPTIONS') return handleOptions(req, res);
    if (req.method !== 'GET') return bad(res, 405, 'Method not allowed');
    if (!assertClientAuthorized(req, res)) return;

    const apiKey = process.env.INTERVALS_API_KEY;
    if (!apiKey || String(apiKey).trim() === '') {
      return bad(res, 401, 'Server not configured: INTERVALS_API_KEY is missing');
    }

    const athleteId = String(process.env.INTERVALS_ATHLETE_ID ?? '0').trim() || '0';
    const newestIn = typeof req.query.newest === 'string' ? req.query.newest.trim() : '';
    const oldestIn = typeof req.query.oldest === 'string' ? req.query.oldest.trim() : '';

    let newestDt = newestIn ? ymdToUtcDate(newestIn) : new Date();
    if (!newestDt || Number.isNaN(newestDt.getTime())) newestDt = new Date();

    let oldestDt = oldestIn ? ymdToUtcDate(oldestIn) : subDays(newestDt, Math.floor(365.25 * 25));
    if (!oldestDt || Number.isNaN(oldestDt.getTime())) {
      oldestDt = subDays(newestDt, Math.floor(365.25 * 25));
    }

    if (oldestDt > newestDt) {
      const t = oldestDt;
      oldestDt = newestDt;
      newestDt = t;
    }

    let limitNum = parseInt(process.env.INTERVALS_ACTIVITIES_FETCH_LIMIT, 10);
    if (!Number.isFinite(limitNum) || limitNum < 1) limitNum = 8000;
    limitNum = Math.min(limitNum, 50000);

    let chunkDays = parseInt(process.env.INTERVALS_ACTIVITIES_CHUNK_DAYS, 10);
    if (!Number.isFinite(chunkDays) || chunkDays < 14) chunkDays = 400;
    chunkDays = Math.min(chunkDays, 2000);

    const dedup = new Map();
    let windowEnd = new Date(newestDt.getTime());
    const floor = new Date(oldestDt.getTime());
    let safety = 0;

    while (windowEnd >= floor && safety < 260) {
      safety += 1;

      let windowStart = subDays(windowEnd, chunkDays);
      if (windowStart < floor) windowStart = new Date(floor.getTime());

      const chunk = await fetchActivitiesChunk({
        apiKey: String(apiKey).trim(),
        athleteId,
        oldestStr: dateToYmd(windowStart),
        newestStr: dateToYmd(windowEnd),
        limitNum
      });

      for (let i = 0; i < chunk.length; i++) {
        const act = chunk[i];
        if (act && act.id != null && !dedup.has(act.id)) dedup.set(act.id, act);
      }

      const hitCap = chunk.length >= limitNum;
      if (hitCap) {
        const narrowed = Math.max(14, Math.floor(chunkDays / 2));
        if (narrowed < chunkDays) {
          chunkDays = narrowed;
          continue;
        }
      }

      windowEnd = subDays(windowStart, 1);
      if (chunk.length === 0 && windowStart <= floor) break;
    }

    const merged = Array.from(dedup.values());
    merged.sort((a, b) => {
      const da = new Date(a.start_date_local || a.start_date || 0).getTime();
      const db = new Date(b.start_date_local || b.start_date || 0).getTime();
      return db - da;
    });

    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'private, max-age=60');
    res.setHeader('X-Intervals-Chunk-Iterations', String(safety));
    res.setHeader('X-Intervals-Activities-Returned', String(merged.length));
    return res.status(200).json(merged);
  } catch (e) {
    console.error('intervals proxy', e);
    const code = typeof e.statusCode === 'number' ? e.statusCode : 502;
    if (code >= 500) return bad(res, 502, e.message || 'Upstream fetch failed');
    return bad(res, code >= 400 && code < 600 ? code : 502, e.message || 'Upstream fetch failed');
  }
};

