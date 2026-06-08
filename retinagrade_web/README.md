# RetinaGrade — Django Web Application

Full-featured Django 4.2 web application for automated Diabetic Retinopathy grading. Wraps the existing FastAPI/PyTorch Lightning inference backend with a production-quality UI.

---

## Prerequisites

- Python 3.10+
- The FastAPI backend (`api.py`) from the parent project

---

## Installation

```bash
# 1. Enter the Django project directory
cd retinagrade_web

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply database migrations
python manage.py migrate

# 5. Seed demo users and sample predictions
python manage.py seed_demo

# 6. (Optional) Create your own superuser
python manage.py createsuperuser
```

---

## Starting the FastAPI Backend

```bash
# From the project root (DR-Classification/)
source .venv/bin/activate          # or activate the correct venv
python api.py
# Listens on http://localhost:8000
```

---

## Starting the Django App

```bash
# In retinagrade_web/ with .venv active
python manage.py runserver
# Listens on http://127.0.0.1:8001
```

> Run Django on port 8001 if FastAPI is already on 8000:
> `python manage.py runserver 8001`

---

## Demo Credentials

| Username    | Password    | Role       |
|-------------|-------------|------------|
| `admin`     | `admin123`  | Admin      |
| `clinician` | `clinic123` | Clinician  |

The `seed_demo` command also creates 5 sample Prediction objects for the clinician user so the app looks populated even without a running FastAPI backend.

---

## Pages

| URL                       | Description                                          |
|---------------------------|------------------------------------------------------|
| `/`                       | **Dashboard** — stats, grade distribution chart, recent predictions |
| `/analyse/`               | **New Analysis** — drag-and-drop upload, spinner, results |
| `/prediction/<id>/`       | **Detail** — Grad-CAM, probability bars, clinical notes, PDF download |
| `/history/`               | **History** — paginated table, grade/date filters, bulk delete |
| `/admin-dashboard/`       | **Admin** — user list, grade donut chart, backend health (admin role only) |
| `/accounts/login/`        | Sign in                                              |
| `/accounts/register/`     | Create account                                       |
| `/accounts/password-change/` | Change password                                   |
| `/django-admin/`          | Django admin panel                                   |

---

## Environment Variables

| Variable             | Default                    | Purpose                          |
|----------------------|----------------------------|----------------------------------|
| `DJANGO_SECRET_KEY`  | insecure dev key           | Set a strong random key in prod  |
| `FASTAPI_BASE_URL`   | `http://localhost:8000`    | FastAPI inference service URL    |

---

## Production Checklist

- Set `DEBUG = False` in `settings.py`
- Set a strong `DJANGO_SECRET_KEY` via env var
- Configure `ALLOWED_HOSTS`
- Run `python manage.py collectstatic`
- Use gunicorn/nginx in front of Django
- Serve `MEDIA_ROOT` via nginx, not Django

---

## Project Structure

```
retinagrade_web/
├── manage.py
├── requirements.txt
├── retinagrade/          # Django project config (settings, urls, wsgi)
├── accounts/             # Custom user model, auth views & templates
├── core/                 # Main app: models, views, services, pdf, templates
│   ├── services.py       # PredictionService — calls FastAPI
│   ├── pdf.py            # ReportLab PDF generation
│   └── management/commands/seed_demo.py
├── static/
│   ├── css/style.css     # Full dark-theme design system
│   └── js/main.js        # Upload zone, Chart.js helpers, bulk select
└── media/                # Uploaded fundus images (git-ignored)
```
