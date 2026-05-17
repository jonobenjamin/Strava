# Deploy: GitHub Pages (frontend) + Vercel (API)

## Why Vercel showed old/broken API

Vercel deploys from **Git**. If this folder was never pushed (or has no `.git`), Vercel only has whatever was last on GitHub — often **without** the `api/` folder or `vercel.json`.

## One-time: connect this folder to GitHub

```bash
cd "/Users/jonathanbenjamin/Documents/Web pages/Strava-main"

git init
git add .
git status   # confirm api/, vercel.json, docs/, package.json are listed (NOT public/)

git commit -m "Add Vercel API routes, CORS, and server-proxy frontend"

# Replace with your real repo URL if different:
git remote add origin https://github.com/jonobenjamin/Strava.git
git branch -M main
git push -u origin main
```

If the remote already has history:

```bash
git pull origin main --rebase
git push origin main
```

## Vercel project settings

| Setting | Value |
|--------|--------|
| Root directory | `.` (repo root) |
| Build command | `npm run build` |
| Output directory | `public` |
| Install command | (leave empty) |

**Environment variables** (Vercel → Settings → Environment Variables):

- `INTERVALS_API_KEY`
- `INTERVALS_ATHLETE_ID`
- `AI_API_KEY`
- `PUBLIC_API_TOKEN` (same string as in `docs/metricsintervals.html`)
- Optional: `AI_API_URL`, `AI_MODEL`

Redeploy after every env change.

## GitHub Pages (frontend)

Your site is `https://jonobenjamin.github.io/...` — ensure GitHub Pages publishes the **`docs/`** folder (or whatever branch/folder you use).

After push, set in `docs/metricsintervals.html`:

- `API_BASE` → your Vercel URL (production URL is better than preview `*-vercel.app`)
- `PUBLIC_API_TOKEN` → same as Vercel
- `VERCEL_PROTECTION_BYPASS` → only if Deployment Protection is on

## What must be in Git for Vercel to work

```
api/
  _cors.js
  _auth.js
  chat.js
  intervals/activities.js
vercel.json
package.json
docs/          # copied to public/ on build
```

`public/` is **gitignored** — Vercel builds it during deploy.
