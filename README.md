# CourseSense — Enrollment Predictor

> A machine-learning powered tool that predicts course enrollment for the CCSU School of Business.  
> Built with **React + TypeScript** (frontend), **Flask** (backend), **PostgreSQL** (database), and **scikit-learn / TensorFlow** (ML).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture at a Glance](#architecture-at-a-glance)
3. [Prerequisites](#prerequisites)
4. [Quick Start (Docker — Recommended)](#quick-start-docker--recommended)
5. [Local Development (Without Docker)](#local-development-without-docker)
6. [Loading Enrollment Data](#loading-enrollment-data)
7. [Training the ML Models](#training-the-ml-models)
8. [Project Structure](#project-structure)
9. [Environment Variables](#environment-variables)
10. [Common Commands Reference](#common-commands-reference)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

---

## Project Overview

CourseSense lets advisors and administrators upload historical enrollment CSVs and receive predictions for upcoming semesters.  The app exposes a REST API (Flask) consumed by a React dashboard.  All enrollment data lives in a PostgreSQL database that is seeded from CSV exports pulled from the university's registration system.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────┐
│  Browser  (React + Vite + TypeScript, port 5173) │
└───────────────────────┬──────────────────────────┘
                        │ HTTP / JSON
┌───────────────────────▼──────────────────────────┐
│  Flask API  (Python 3.13, port 5000)             │
│  backend/app/app.py                              │
└───────┬───────────────────────────┬──────────────┘
        │ psycopg2                  │ scikit-learn / TF
┌───────▼───────┐         ┌─────────▼──────────────┐
│  PostgreSQL   │         │  ML Module             │
│  (port 5432)  │         │  backend/app/ml/       │
└───────────────┘         └────────────────────────┘
```

Four Docker services are defined in `docker-compose.yml`:

| Service    | Purpose                                          |
|------------|--------------------------------------------------|
| `db`       | PostgreSQL 15 database                           |
| `db-init`  | One-shot container that seeds the database       |
| `web`      | Flask REST API                                   |
| `client`   | React/Vite development server                    |

---

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop) | Latest | Must be running (green icon) |
| [Git](https://git-scm.com/) | 2.x | |
| Python | 3.13 | Only needed for local (non-Docker) development |
| Node.js | 18+ | Only needed for local frontend development |
| VS Code | Latest | Recommended; install the **Python** and **Docker** extensions |

> **Note:** SQL data files, raw CSV exports, and trained model artifacts are **not** stored in this repository. You must obtain them separately (see [Loading Enrollment Data](#loading-enrollment-data)).

---

## Quick Start (Docker — Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/AI-460-School-of-Business/Enrollment-Predictor.git
cd Enrollment-Predictor
```

### 2. Configure environment variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env   # if .env.example exists, otherwise edit .env directly
```

Open `.env` and set:

```env
POSTGRES_USER=DBUser
POSTGRES_PASSWORD=DBPassword
POSTGRES_DB=enrollprdctDB
DB_HOST=db
DB_PORT=5432
```

> **Security:** Never commit real credentials to the repository.  
> Keep `.env` local — it is already listed in `.gitignore`.

### 3. Add enrollment data (see [Loading Enrollment Data](#loading-enrollment-data))

### 4. Build and start all containers

```bash
docker-compose build
docker-compose up
```

The first run will:
1. Start the PostgreSQL database.
2. Run the `db-init` container, which imports your CSV files and seeds the database.
3. Start the Flask API on **http://localhost:5000**.
4. Start the React app on **http://localhost:5173**.

### 5. Open the app

Visit **http://localhost:5173** in your browser.

### 6. Stop the containers

```bash
# Stop without removing data volumes:
docker-compose down

# Stop AND wipe the database volume (full reset):
docker-compose down -v
```

---

## Local Development (Without Docker)

Use this workflow when you want faster iteration on a specific service.

### Backend (Flask)

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask development server
python app/app.py
```

The API will be available at **http://localhost:5000**.

> You still need a running PostgreSQL instance.  The easiest way is to start only the `db` service:
> ```bash
> docker-compose up db
> ```

### Frontend (React + Vite)

```bash
cd frontend

npm install
npm run dev
```

The app will be available at **http://localhost:5173**.

### Run both at once (convenience script)

```bash
# Linux / macOS
bash run_local.sh

# Windows PowerShell
.\run_local.ps1
```

---

## Loading Enrollment Data

Because raw enrollment data contains sensitive student information, **CSV and SQL files are excluded from this repository** via `.gitignore`.

### Where to place the files

| File type | Destination folder |
|-----------|-------------------|
| Raw CSV exports | `backend/data/csv/` |
| SQL dump (optional, faster) | `backend/data/sql/` |

### Obtaining the data

Contact the CCSU School of Business registrar or a previous project team member for the CSV exports from the Banner registration system.  Two CSV files are expected:

- `All Course Sections 2022 2025.csv`
- `Section Detail Report SBUS(Section Detail Report SBUS).csv`

Place them inside `backend/data/csv/` before running `docker-compose up`.

### How the database gets seeded

The `db-init` Docker service (`backend/app/database/db_init.py`) runs automatically on first startup and:

1. Waits for PostgreSQL to be ready.
2. Checks if the database is already populated (skip if it is).
3. Looks for `.sql` dump files in `backend/data/sql/` — uses them if present (fastest path).
4. Falls back to importing CSVs directly via `backend/app/database/read_csv.py`.
5. Generates `backend/app/database/enrollment_features_auto.json` from the live schema.

---

## Training the ML Models

The ML module lives in `backend/app/ml/`.  There are three model types:

| Flag | Algorithm | Best for |
|------|-----------|---------|
| `--model linear` | Ridge Regression | Baseline / interpretability |
| `--model tree` | Random Forest | General-purpose predictions |
| `--model neural` | Neural Network (TensorFlow) | Maximum accuracy |

### Run training inside Docker

```bash
docker-compose exec web python backend/app/ml/train_model.py --model tree --features min
```

### Run training locally

```bash
cd backend
source .venv/bin/activate
python app/ml/train_model.py --model tree --features min
```

Trained models are saved to `backend/data/prediction_models/` as `.pkl` files.  These files are excluded from the repository — they must be regenerated from your local data.

### Feature schemas

| Schema file | Description |
|-------------|-------------|
| `enrollment_features_min.json` | Minimal features: CRN, Semester, Year |
| `enrollment_features_rich.json` | Full feature set including demographics and historical context |
| `enrollment_features_auto.json` | Auto-generated from the live database schema |

---

## Project Structure

```
Enrollment-Predictor/
├── backend/
│   ├── app/
│   │   ├── app.py                  # Flask entry point & all API routes
│   │   ├── database/
│   │   │   ├── db_init.py          # Database seeding logic
│   │   │   ├── read_csv.py         # CSV → PostgreSQL importer
│   │   │   ├── data_validation.py  # Input validation helpers
│   │   │   ├── export_data.py      # SQL dump exporter
│   │   │   └── generate_auto_features.py
│   │   └── ml/
│   │       ├── train_model.py      # Main training script (all 3 models)
│   │       ├── predict.py          # Prediction logic
│   │       ├── predictor_service.py
│   │       ├── visualize.py        # Matplotlib plots
│   │       ├── data/               # Data loading & feature engineering
│   │       ├── models/             # Model class definitions
│   │       ├── feature_schema/     # JSON feature definitions
│   │       └── utils/              # DB config & evaluation helpers
│   ├── data/
│   │   ├── csv/                    # ⚠ Raw CSV exports (not in git)
│   │   ├── sql/                    # ⚠ SQL dumps (not in git)
│   │   └── prediction_models/      # ⚠ Trained .pkl files (not in git)
│   ├── requirements.txt
│   └── entrypoint.sh
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Root component & routing
│   │   ├── components/             # Reusable UI components
│   │   └── styles/
│   ├── reports/                    # ⚠ Generated reports (not in git)
│   ├── public/                     # Static assets (logos, icons)
│   ├── package.json
│   └── Dockerfile.client
├── .env                            # ⚠ Local secrets (not in git)
├── docker-compose.yml
├── Dockerfile
├── run_local.sh                    # Convenience: start backend + frontend
└── README.md
```

> Folders marked ⚠ are excluded from the repository.

---

## Environment Variables

All variables are loaded from the `.env` file at the project root.

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `DBUser` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `DBPassword` | PostgreSQL password |
| `POSTGRES_DB` | `enrollprdctDB` | Database name |
| `DB_HOST` | `db` | Hostname (use `db` inside Docker, `localhost` for local dev) |
| `DB_PORT` | `5432` | PostgreSQL port |

---

## Common Commands Reference

```bash
# --- Docker ---
docker-compose build          # Rebuild all images
docker-compose up             # Start all services (attach)
docker-compose up -d          # Start all services (detached / background)
docker-compose down           # Stop services, keep data
docker-compose down -v        # Stop services, WIPE database volume
docker-compose logs -f        # Tail all logs
docker-compose logs -f web    # Tail Flask API logs only

# --- Run a one-off command in a running container ---
docker-compose exec web bash
docker-compose exec web python backend/app/ml/train_model.py --model tree

# --- Frontend (local) ---
cd frontend && npm install && npm run dev
cd frontend && npm run build  # Production build

# --- Backend (local) ---
cd backend && pip install -r requirements.txt
python app/app.py

# --- Database (local psql) ---
psql -h localhost -U DBUser -d enrollprdctDB
```

---

## Troubleshooting

### Docker containers exit immediately on first run

The `db-init` service exits with code 1 when it cannot find data files.  Make sure your CSV files are in `backend/data/csv/` before running `docker-compose up`.

### `psycopg2.OperationalError: could not connect to server`

The Flask API is trying to reach the database before it is ready.  Run `docker-compose up` again — Docker Compose will retry.  If you are running Flask locally, make sure the `db` Docker service is running:

```bash
docker-compose up db
```

### Port conflicts

If port 5000, 5173, or 5432 is already in use on your machine, either stop the conflicting process or change the port mapping in `docker-compose.yml`.

### Predictions return no results / model not found

Trained model files (`.pkl`) are not stored in the repository.  Train the model first:

```bash
docker-compose exec web python backend/app/ml/train_model.py --model tree --features min
```

### Frontend changes not reflecting

Vite uses hot module replacement.  If changes are not appearing, try a hard refresh (`Ctrl+Shift+R`) or restart the client container:

```bash
docker-compose restart client
```

---

## Contributing

1. **Branch** off `main` for your feature or bug fix.
2. **Never commit** raw data files (CSV, SQL, XLSX) or trained model artifacts (`.pkl`).  They are excluded by `.gitignore`.
3. Keep the `.env` file local — use `.env.example` (if provided) to document new variables.
4. Open a pull request against `main` with a clear description of your changes.

For detailed documentation on the ML module see [`backend/app/ml/README.md`](backend/app/ml/README.md).  
For the database schema and seeding process see [`backend/app/database/README.md`](backend/app/database/README.md).  
For the full API reference see [`backend/API_DOCUMENTATION.md`](backend/API_DOCUMENTATION.md).


