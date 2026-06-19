FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_MODE=demo \
    DEMO_ALLOW_EXTERNAL_API=0 \
    ENERGY_PULSE_TIMEZONE=Europe/Paris \
    ENERGY_PULSE_HISTORY_HOURS=72

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m scripts.bootstrap

EXPOSE 8501

CMD ["python", "run_app.py"]
