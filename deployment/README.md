# Deployment Configuration

## Environment Variables

Set these environment variables in your deployment platform:

```bash
# Required
PORT=5000
SECRET_KEY=your-secret-key-here

# Optional - Firebase (if using authentication)
FIREBASE_CONFIG={"type": "service_account", ...}
# Or
FIREBASE_SERVICE_ACCOUNT_PATH=path/to/service-account.json

# Flask Configuration
FLASK_ENV=production
```

## Deployment Checklist

1. **Frontend API URLs**: All frontend files now use `window.location.origin` for API calls, which automatically uses the correct domain.

2. **CORS**: Backend is configured to accept requests from any origin with `CORS(app, origins=["*"])`.

3. **Static Files**: The Flask app serves static files from the frontend directory.

4. **API Endpoints**: 
   - Main app runs on the root domain
   - DataBench API: `/api/databench/*`
   - Plug & Play API: `/api/plugplay/*`
   - Static files: `/css/*`, `/js/*`, `/assets/*`, `/pages/*`

## Running Locally

```bash
# Development mode (with debug)
python backend/api/main.py

# Production mode
python deploy.py

# Custom port
PORT=5001 python deploy.py
```

## Common Issues

1. **"Failed to fetch" errors**: 
   - Check that all frontend files use relative URLs (`window.location.origin`)
   - Ensure CORS is properly configured
   - Verify the backend is running and accessible

2. **Port conflicts**:
   - Use the PORT environment variable to change the port
   - On macOS, disable AirPlay Receiver if port 5000 is in use

3. **Firebase errors**:
   - Authentication features require proper Firebase configuration
   - Set FIREBASE_CONFIG environment variable with your service account JSON