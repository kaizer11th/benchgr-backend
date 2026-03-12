# BenchGR Backend

FastAPI backend for BenchGR GPU Benchmark Leaderboard.

## Deploy to Vercel

1. Push this folder as its own GitHub repo (e.g. `benchgr-backend`)
2. Go to vercel.com → New Project → import `benchgr-backend`
3. Framework preset → Other
4. Add environment variables:
   - `DATABASE_URL` → your Supabase connection string
   - `SECRET_KEY` → any long random string
   - `ENVIRONMENT` → production
5. Deploy

Your API will be live at: https://benchgr-backend.vercel.app

## Local dev
```bash
cd api
pip install -r requirements.txt
uvicorn index:app --reload --port 8000
```
