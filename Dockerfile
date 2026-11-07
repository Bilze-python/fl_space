FROM node:22-bookworm-slim AS frontend

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPACEFL_PUBLIC_DEMO=1

WORKDIR /app

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --from=frontend /app/node_modules ./node_modules
COPY fl_space ./fl_space
COPY web ./web
COPY deploy/demo_data ./deploy/demo_data
RUN useradd --create-home --uid 10001 spacefl \
    && chown -R spacefl:spacefl /app

USER spacefl
EXPOSE 8720

# Render/HF set PORT automatically; fall back to 8720 for the local Docker workflow.
# proxy-headers keeps request.url.scheme / client IP correct behind Render's TLS proxy.
CMD ["sh", "-c", "python -m uvicorn --factory web.server:create_app --host 0.0.0.0 --port ${PORT:-8720} --proxy-headers --forwarded-allow-ips='*' --log-level warning"]
