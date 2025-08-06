# Deployment Setup

## Current Configuration

### Main Files
- **Backend**: `simple_deploy.py` - Main Flask application with all routes
- **WSGI Entry**: `wsgi.py` - Imports app from simple_deploy for Gunicorn
- **Config**: `deployment/railway.toml` - Railway deployment configuration
- **Docker**: `Dockerfile` - Container setup
- **Process**: `Procfile` - Defines web process command

### Deployment Command
```bash
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT wsgi:app
```

### Key Features
- ✅ Proper JSON responses for all API endpoints
- ✅ WebSocket support via Flask-SocketIO with eventlet
- ✅ Error handlers for 404/500/405 that return JSON for API routes
- ✅ Serves pages from both `pages/` and `frontend/pages/` directories
- ✅ All plug & play functionality working

### API Endpoints
- `/api/plugplay/start-installation` - Start LeRobot installation
- `/api/plugplay/installation-status` - Check installation status
- `/api/plugplay/cancel-installation` - Cancel installation
- `/api/plugplay/system-info` - Get system information
- `/api/plugplay/list-ports` - List USB ports
- `/api/plugplay/save-port-config` - Save port configuration

### Static Routes
- `/` - Index page
- `/pages/<filename>` - Serve HTML pages
- `/css/<filename>` - Serve CSS files
- `/js/<filename>` - Serve JavaScript files
- `/assets/<filename>` - Serve asset files

### Requirements
All dependencies are in `requirements-deploy.txt`:
- Flask and extensions
- Gunicorn with eventlet for production
- PyYAML, pyserial for robot functionality
- Optional Firebase for authentication

## Deployment Steps
1. Commit all changes
2. Push to repository
3. Railway automatically deploys from main branch
4. Verify health check at `/health` endpoint

## Testing Locally
```bash
# Run development server
python simple_deploy.py

# Test with Gunicorn (needs pip install gunicorn eventlet)
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 wsgi:app
```