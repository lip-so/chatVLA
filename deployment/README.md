# Deployment Configuration

## FIXED: Multiple Deployment Errors

### ✅ "databench_api.py" Error - RESOLVED
The deployment error was caused by missing `databench_api.py` file references. This has been fixed by:

1. ✅ Created `simple_deploy.py` - lightweight deployment entry point
2. ✅ Created `requirements-deploy.txt` - minimal dependencies 
3. ✅ Updated all deployment configs to use the new entry point
4. ✅ Bypassed heavy databench dependencies that caused import errors

### ✅ "NameError: name 'current_installation' is not defined" - RESOLVED
The threading error was caused by variable scope issues. Fixed by:

1. ✅ Moved `current_installation` to global scope
2. ✅ Added `global current_installation` declarations in all functions
3. ✅ Created global `socketio_instance` for thread access
4. ✅ Verified threading works correctly with installation simulation

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

## Production Deployment

The application now uses Gunicorn with eventlet worker for production-ready WebSocket support:

```bash
# Production command (used by Railway/Render/etc)
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT simple_deploy:app
```

## Running Locally

```bash
# Development mode (with auto-reload)
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
- ✅ Compatible with Gunicorn for production deployment
- ✅ WebSocket support via eventlet

### `deploy.py` (Full Features)
- ⚠️ Requires all heavy dependencies
- ⚠️ May have import issues in some environments
- ✅ Includes DataBench evaluation features

### `app.py` (Fallback)
- ✅ Automatic fallback if imports fail
- ✅ Graceful degradation

## Common Issues

1. **✅ "databench_api.py" not found**: 
   - ✅ **FIXED**: Use `simple_deploy.py` instead of `app.py` or `deploy.py`
   - Root cause: Heavy databench dependencies causing import failures
   - Solution: Lightweight deployment entry point

2. **✅ "NameError: name 'current_installation' is not defined"**: 
   - ✅ **FIXED**: Variable scope issue in threading resolved
   - Root cause: Local variable not accessible from background thread
   - Solution: Moved to global scope with proper declarations

3. **"Failed to fetch" errors**: 
   - Check that all frontend files use relative URLs (`window.location.origin`)
   - Ensure CORS is properly configured
   - Verify the backend is running and accessible

4. **Port conflicts**:
   - Use the PORT environment variable to change the port
   - On macOS, disable AirPlay Receiver if port 5000 is in use

5. **Firebase errors**:
   - Authentication features require proper Firebase configuration
   - Set FIREBASE_CONFIG environment variable with your service account JSON

6. **Import errors during deployment**:
   - Use `requirements-deploy.txt` instead of `requirements.txt`
   - Use `simple_deploy.py` for constrained environments

7. **✅ "RuntimeError: The Werkzeug web server is not designed to run in production"**:
   - ✅ **FIXED**: Now using Gunicorn with eventlet worker for production
   - Root cause: Flask-SocketIO's development server was being used in production
   - Solution: Configured Gunicorn with eventlet for WebSocket support

8. **✅ "Failed to start installation: Unexpected token '<', \"<html> <he\"... is not valid JSON"**:
   - ✅ **FIXED**: API endpoint now properly returns JSON instead of HTML error pages
   - Root cause: Parameter name mismatches between frontend files and blueprint routing issues  
   - Solution: Fixed parameter handling and created proper WSGI entry point (`wsgi.py`)