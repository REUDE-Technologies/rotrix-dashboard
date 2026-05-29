# RotriDASH

Streamlit motor assessment dashboard (multi-parameter analysis, reports, calculators).

## Run locally

```bash
cd RotriDASH
pip install -r requirements.txt
streamlit run app.py
```

## Environment

Copy `.env.example` if present, or set:

- `AUTH_BACKEND` — `local_pg` or `supabase`
- `DATABASE_URL` — Postgres connection string (local auth)
- `SUPABASE_URL`, `SUPABASE_ANON_KEY` — when using Supabase auth

Seed a local super admin:

```bash
python seed_admin.py
```

## Feature flags (`config.py`)

- `SHOW_CALCULATORS_BUTTON` — header Calculators shortcut
- `SHOW_ANALYSIS_TYPE_SELECTOR` — Multi-Parameter vs Multi-File radio on analysis page
