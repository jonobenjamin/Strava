# Vercel API (Node only)

Deploy this folder as the Vercel project **Root Directory**. There is no build step.

## Files

- `api/intervals/activities.js` — GET proxy for intervals.icu
- `api/chat.js` — POST proxy for AI chat
- `api/_cors.js` — CORS for `https://jonobenjamin.github.io`
- `api/_auth.js` — optional `PUBLIC_API_TOKEN` gate
- `api/_fetch.js` — fetch with timeout
- `vercel.json` — `@vercel/node` builds + CORS headers

## Env vars

See [.env.example](./.env.example).
