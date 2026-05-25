# InsightX AI

**Explainable AI Platform** — Making ML models interpretable, trustworthy, and actionable.

## Project Structure

```
InsightX/
├── frontend/          # Next.js 15 + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── app/       # App Router pages & layouts
│   │   ├── components/  # Reusable UI components
│   │   │   ├── ui/      # Shadcn/ui primitives
│   │   │   ├── dashboard/  # Dashboard-specific components
│   │   │   ├── charts/    # Recharts visualization wrappers
│   │   │   └── layout/    # Shell, Sidebar, Nav
│   │   ├── lib/       # Firebase config, API client, utilities
│   │   ├── hooks/     # Custom React hooks
│   │   ├── types/     # TypeScript type definitions
│   │   └── styles/    # Global CSS & Tailwind extensions
│   └── public/        # Static assets
│
├── backend/           # Python FastAPI ML Engine
│   ├── app/
│   │   ├── main.py    # FastAPI entry point
│   │   ├── config.py  # Environment & path config
│   │   ├── routes/    # API endpoint definitions
│   │   │   ├── data.py      # Upload & preprocessing endpoints
│   │   │   ├── train.py     # Model training endpoints
│   │   │   ├── explain.py   # SHAP/LIME explanation endpoints
│   │   │   └── bias.py      # Fairness evaluation endpoints
│   │   ├── services/  # Business logic
│   │   │   ├── ml_service.py    # Training & prediction
│   │   │   ├── xai_service.py   # SHAP/LIME computation
│   │   │   └── bias_service.py  # Fairlearn integration
│   │   └── models/    # Pydantic schemas
│   ├── uploads/       # Uploaded CSV/JSON files (gitignored)
│   ├── trained_models/ # Serialized model artifacts (gitignored)
│   └── requirements.txt
│
└── README.md
```

## Quick Start

### Frontend
```bash
cd frontend
npm install
npm run dev          # → http://localhost:3000
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload    # → http://localhost:8000
```

## Tech Stack

| Layer          | Technology                                        |
|----------------|---------------------------------------------------|
| Frontend       | Next.js 15, TypeScript, Tailwind CSS, Framer Motion |
| UI Components  | Shadcn/ui (Radix primitives)                      |
| Charts         | Recharts                                          |
| Auth & DB      | Firebase (Spark Plan — free)                      |
| Backend        | Python 3.11+, FastAPI, Uvicorn                    |
| ML Engine      | scikit-learn, pandas, numpy                       |
| XAI            | SHAP, LIME                                        |
| Fairness       | Fairlearn                                         |

## License
MIT
# InsightX
