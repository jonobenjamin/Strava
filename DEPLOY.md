# Deploy: GitHub Pages (frontend) + Vercel (API)

This repo is a **monorepo** with two deploy targets. Do not point Vercel at the repo root (Python + `requirements.txt` live there for GitHub Actions only).

## Repository layout

| Path | Deployed to | Contents |
|------|-------------|----------|
| `docs/` | **GitHub Pages** | Static HTML/JS/CSS, `config.js` (API URL only, no secrets) |
| `backend/` | **Vercel** | Node serverless `/api/*` only |
| `scripts/`, `requirements.txt` | **GitHub Actions** | Strava/travel fetch — never deployed to Vercel |

```
Strava-main/
├── docs/                 ← GitHub Pages
│   ├── config.js         ← API_BASE, optional PUBLIC_API_TOKEN
│   └── metricsintervals.html
├── backend/              ← Vercel (set Root Directory here)
│   ├── api/
│   ├── vercel.json
│   └── package.json
├── scripts/              ← CI only (Python)
└── requirements.txt      ← CI only — NOT in backend/
```

---

## 1. GitHub Pages (frontend)

1. Push this repo to GitHub.
2. **Settings → Pages → Build and deployment**
   - Source: **Deploy from a branch**
   - Branch: `main` (or your default)
   - Folder: **`/docs`**
3. Site URL example: `https://jonobenjamin.github.io/Strava/` (depends on repo name).

### Configure API URL

Edit `docs/config.js` (copy from `docs/config.example.js` if needed):

```javascript
window.APP_CONFIG = {
  API_BASE: 'https://YOUR-PROJECT.vercel.app/api',
  PUBLIC_API_TOKEN: '',           // match Vercel PUBLIC_API_TOKEN if set
  VERCEL_PROTECTION_BYPASS: ''    // only if Deployment Protection is on
};
```

Commit and push. The dashboard is at `metricsintervals.html` under your Pages path.

**Do not** put `INTERVALS_API_KEY` or `AI_API_KEY` in the frontend.

---

## 2. Vercel (backend API)

### Create / fix the Vercel project

Pick **one** setup (both `vercel.json` files are in the repo):

| Vercel Root Directory | Config file GitHub must have |
|----------------------|------------------------------|
| **`backend`** (recommended) | `backend/vercel.json` |
| **`.`** (repo root) | `vercel.json` at repo root + `.vercelignore` |

| Setting | Value |
|---------|--------|
| Framework Preset | Other |
| Build Command | *(empty — do not use `npm run build`)* |
| Output Directory | *(empty — do not use `public`)* |
| Install Command | *(empty)* |

**Why `vercel.json` “won’t commit”:** the API moved to `backend/`, but only the old root file was on GitHub. You must `git add` the new path and push — editing on github.com at `/vercel.json` alone does nothing if your Vercel project Root Directory is `backend`.

```bash
git add vercel.json backend/vercel.json .vercelignore backend/
git commit -m "Add working vercel.json for API routes and CORS"
git push origin main
```

### Environment variables

Vercel → Project → Settings → Environment Variables (Production + Preview):

| Variable | Required | Purpose |
|----------|----------|---------|
| `INTERVALS_API_KEY` | Yes | intervals.icu API key (server only) |
| `INTERVALS_ATHLETE_ID` | Yes | Athlete id (often `0`) |
| `AI_API_KEY` | Yes (for coach) | OpenRouter / OpenAI key |
| `PUBLIC_API_TOKEN` | Recommended | Gate between Pages and API |
| `AI_API_URL` | No | Default: OpenRouter chat completions URL |
| `AI_MODEL` | No | Default: `openai/gpt-4o-mini` |

Redeploy after changing env vars.

### API routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/intervals/activities?oldest=&newest=` | Proxies intervals.icu activities |
| `POST` | `/api/chat` | Proxies AI chat |
| `OPTIONS` | both | CORS preflight → `200` |

CORS allows origin: `https://jonobenjamin.github.io` (see `backend/vercel.json` and `backend/api/_cors.js`).

### Deployment Protection (401 on `*.vercel.app`)

If previews return **401** before your function runs, that is Vercel **Deployment Protection**, not a missing intervals key.

- Disable protection for the API project, **or**
- Set `VERCEL_PROTECTION_BYPASS` in `docs/config.js` to the automation bypass secret from Vercel.

### Verify after deploy

```bash
curl -sS -D - -o /dev/null \
  -X OPTIONS \
  -H "Origin: https://jonobenjamin.github.io" \
  -H "Access-Control-Request-Method: GET" \
  "https://YOUR-PROJECT.vercel.app/api/intervals/activities"
```

Expect `200` and `Access-Control-Allow-Origin: https://jonobenjamin.github.io`.

```bash
curl -sS "https://YOUR-PROJECT.vercel.app/api/intervals/activities?oldest=2024-01-01&newest=2024-12-31" \
  -H "Authorization: Bearer YOUR_PUBLIC_API_TOKEN"
```

Expect JSON array or a clear JSON error (`401` with message if env/token missing).

---

## 3. Push workflow

```bash
cd "/path/to/Strava-main"
git add docs/ backend/ DEPLOY.md README.md .gitignore
git status   # must NOT include root api/, root vercel.json, or secrets
git commit -m "Split frontend (docs) and Vercel API (backend)"
git push origin main
```

Vercel redeploys from `backend/` automatically. GitHub Pages updates from `docs/`.

---

## Checklist

- [ ] Vercel **Root Directory** = `backend`
- [ ] No `api/` or `vercel.json` at repo root
- [ ] `docs/config.js` → correct `API_BASE`
- [ ] `PUBLIC_API_TOKEN` matches on Vercel and in `docs/config.js` (if used)
- [ ] GitHub Pages source = `/docs`
- [ ] Env vars set on Vercel; redeployed
- [ ] OPTIONS + GET return CORS headers from production URL
