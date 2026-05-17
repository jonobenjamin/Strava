/**
 * Proxies intervals.icu GET .../athlete/{id}/activities
 * Secrets: INTERVALS_API_KEY, INTERVALS_ATHLETE_ID (optional, default "0")
 *
 * Fetches the full client date range in time chunks and merges results (intervals.icu
 * may cap how many activities a single request returns).
 */

const { applyCors, handleOptions, jsonError } = require('../_cors');
const { assertClientAuthorized } = require('../_auth');

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
  const r = await fetch(upstream.toString(), {
    method: 'GET',
    headers: {
      Authorization: `Basic ${basic}`,
      Accept: 'application/json'
    }
  });

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
  if (req.method === 'OPTIONS') return handleOptions(req, res);
  if (req.method !== 'GET') {
    return bad(res, 405, 'Method not allowed');
  }
  if (!assertClientAuthorized(req, res)) return;

  const apiKey = process.env.INTERVALS_API_KEY;
  if (!apiKey || String(apiKey).trim() === '') {
    return bad(res, 500, 'Server not configured: set INTERVALS_API_KEY in project environment variables.');
  }

  const athleteId = String(process.env.INTERVALS_ATHLETE_ID ?? '0').trim() || '0';
  const newestIn = typeof req.query.newest === 'string' ? req.query.newest.trim() : '';
  const oldestIn = typeof req.query.oldest === 'string' ? req.query.oldest.trim() : '';

  let newestDt = newestIn ? ymdToUtcDate(newestIn) : new Date();
  if (!newestDt || Number.isNaN(newestDt.getTime())) {
    newestDt = new Date();
  }

  let oldestDt = oldestIn ? ymdToUtcDate(oldestIn) : subDays(newestDt, Math.floor(365.25 * 25));
  if (!oldestDt || Number.isNaN(oldestDt.getTime())) {
    oldestDt = subDays(newestDt, Math.floor(365.25 * 25));
  }

  if (oldestDt > newestDt) {
    const t = oldestDt;
    oldestDt = newestDt;
    newestDt = t;
  }

  const fetchLimitRaw = process.env.INTERVALS_ACTIVITIES_FETCH_LIMIT;
  let limitNum = parseInt(fetchLimitRaw, 10);
  if (!Number.isFinite(limitNum) || limitNum < 1) limitNum = 8000;
  limitNum = Math.min(limitNum, 50000);

  const chunkDaysRaw = process.env.INTERVALS_ACTIVITIES_CHUNK_DAYS;
  let chunkDays = parseInt(chunkDaysRaw, 10);
  if (!Number.isFinite(chunkDays) || chunkDays < 14) chunkDays = 400;
  chunkDays = Math.min(chunkDays, 2000);

  const dedup = new Map();

  let windowEnd = new Date(newestDt.getTime());
  const floor = new Date(oldestDt.getTime());
  let safety = 0;

  try {
    while (windowEnd >= floor && safety < 260) {
      safety += 1;

      let windowStart = subDays(windowEnd, chunkDays);
      if (windowStart < floor) windowStart = new Date(floor.getTime());

      const newestStr = dateToYmd(windowEnd);
      const oldestStr = dateToYmd(windowStart);

      const chunk = await fetchActivitiesChunk({
        apiKey,
        athleteId,
        oldestStr,
        newestStr,
        limitNum
      });

      for (let i = 0; i < chunk.length; i++) {
        const act = chunk[i];
        const id = act && act.id;
        if (id != null && !dedup.has(id)) dedup.set(id, act);
      }

      /** Suspected truncation → narrow window rather than silently lose rows */
      const hitCap = chunk.length >= limitNum;
      if (hitCap) {
        const narrowed = Math.max(14, Math.floor(chunkDays / 2));
        if (narrowed < chunkDays) {
          chunkDays = narrowed;
          continue;
        }
      }

      windowEnd = subDays(windowStart, 1);

      /** No progress safeguard */
      if (chunk.length === 0 && windowStart <= floor) {
        break;
      }
    }

    const merged = Array.from(dedup.values());
    merged.sort((a, b) => {
      const da = new Date(a.start_date_local || a.start_date || 0).getTime();
      const db = new Date(b.start_date_local || b.start_date || 0).getTime();
      return db - da;
    });

    res.setHeader('Cache-Control', 'private, max-age=60');
    res.setHeader('X-Intervals-Chunk-Iterations', String(safety));
    res.setHeader('X-Intervals-Activities-Returned', String(merged.length));

    res.status(200).json(merged);
  } catch (e) {
    const code = typeof e.statusCode === 'number' ? e.statusCode : 502;
    console.error('intervals proxy', e);
    if (code >= 500) return bad(res, 502, e.message || 'Upstream fetch failed');
    return bad(res, code >= 400 && code < 600 ? code : 502, e.message || 'Upstream fetch failed');
  }
};

