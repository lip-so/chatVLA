# Deployment Guide

This guide covers deployment options for Tune Robotics.

## Local Development

```bash
python start.py
# or
python backend/api/main.py
```

## Docker

Build and run with Docker:

```bash
docker build -t tune-robotics .
docker run -p 5000:5000 tune-robotics
```

## Heroku

1. Install Heroku CLI
2. Create app: `heroku create your-app-name`
3. Deploy: `git push heroku main`

The `deployment/Procfile` is configured for Heroku.

## Railway

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Deploy: `railway up`

The `deployment/railway.toml` is configured for Railway.

## Render

1. Connect your GitHub repository to Render
2. Use the `deployment/render.yaml` for automatic configuration
3. Deploy through Render dashboard

## Environment Variables

Set these environment variables in production:

- `FLASK_ENV=production`
- `PORT=5000` (or your preferred port)

## Static Files

For production, consider serving static files through a CDN:

1. Upload `frontend/` contents to CDN
2. Update API to only serve backend endpoints
3. Configure CORS appropriately
