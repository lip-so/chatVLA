# Deployment Configuration

## FIXED: "databench_api.py" Error

The deployment error was caused by missing `databench_api.py` file references. This has been fixed by:

1. ✅ Created `simple_deploy.py` - lightweight deployment entry point
2. ✅ Created `requirements-deploy.txt` - minimal dependencies 
3. ✅ Updated all deployment configs to use the new entry point
4. ✅ Bypassed heavy databench dependencies that caused import errors

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
# Simple deployment (recommended for production)
python simple_deploy.py

# Full development mode (with debug)
python backend/api/main.py

# Full production mode (if all dependencies are available)
python deploy.py

# Custom port
PORT=5001 python simple_deploy.py
```

## Deployment Entry Points

### `simple_deploy.py` (Recommended for Production)
- ✅ Lightweight, minimal dependencies
- ✅ No databench import issues
- ✅ All core Plug & Play functionality
- ✅ Works in constrained deployment environments

### `deploy.py` (Full Features)
- ⚠️ Requires all heavy dependencies
- ⚠️ May have import issues in some environments
- ✅ Includes DataBench evaluation features

### `app.py` (Fallback)
- ✅ Automatic fallback if imports fail
- ✅ Graceful degradation

## Common Issues

1. **❌ "databench_api.py" not found**: 
   - ✅ **FIXED**: Use `simple_deploy.py` instead of `app.py` or `deploy.py`
   - Root cause: Heavy databench dependencies causing import failures
   - Solution: Lightweight deployment entry point

2. **"Failed to fetch" errors**: 
   - Check that all frontend files use relative URLs (`window.location.origin`)
   - Ensure CORS is properly configured
   - Verify the backend is running and accessible

3. **Port conflicts**:
   - Use the PORT environment variable to change the port
   - On macOS, disable AirPlay Receiver if port 5000 is in use

4. **Firebase errors**:
   - Authentication features require proper Firebase configuration
   - Set FIREBASE_CONFIG environment variable with your service account JSON

5. **Import errors during deployment**:
   - Use `requirements-deploy.txt` instead of `requirements.txt`
   - Use `simple_deploy.py` for constrained environments