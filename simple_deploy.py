#!/usr/bin/env python3
"""
Production deployment configuration for ChatVLA
Handles graceful initialization and module loading for cloud deployment
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root and backend to Python path
PROJECT_ROOT = Path(__file__).parent
BACKEND_PATH = PROJECT_ROOT / 'backend'

# Ensure paths are added
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_PATH))

# Set production environment variables
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('PYTHONPATH', f"{PROJECT_ROOT}:{BACKEND_PATH}")

# Production configuration
PRODUCTION_CONFIG = {
    'DEBUG': False,
    'TESTING': False,
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'change-this-in-production-' + os.urandom(24).hex()),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'PERMANENT_SESSION_LIFETIME': 86400,  # 24 hours
}

def initialize_app():
    """Initialize the Flask application with proper error handling"""
    try:
        # Try to import the main app with all features
        from backend.api.main import app, socketio
        logger.info("✅ Successfully loaded main application with all features")
        
        # Apply production configuration
        app.config.update(PRODUCTION_CONFIG)
        
        # Log available services
        logger.info("Services available:")
        
        # Check DataBench availability
        try:
            import backend.databench.api
            logger.info("  ✅ DataBench API: /api/databench/*")
        except ImportError as e:
            logger.warning(f"  ⚠️  DataBench API not available: {e}")
        
        # Check Plug & Play availability
        try:
            import backend.plug_and_play.api
            logger.info("  ✅ Plug & Play API: /api/plugplay/*")
        except ImportError as e:
            logger.warning(f"  ⚠️  Plug & Play API not available: {e}")
        
        # Check Authentication availability
        try:
            import backend.auth.firebase_auth
            logger.info("  ✅ Firebase Authentication")
        except ImportError as e:
            logger.warning(f"  ⚠️  Firebase Authentication not available: {e}")
            logger.info("  ℹ️  Using fallback authentication")
        
        return app, socketio
        
    except ImportError as e:
        logger.error(f"Failed to import main application: {e}")
        logger.info("Attempting fallback initialization...")
        
        # Fallback: Create minimal Flask app
        try:
            from flask import Flask, send_from_directory
            from flask_cors import CORS
            from flask_socketio import SocketIO
            
            app = Flask(__name__, 
                       static_folder=str(PROJECT_ROOT),
                       static_url_path='')
            
            # Apply configuration
            app.config.update(PRODUCTION_CONFIG)
            
            # Enable CORS
            CORS(app, resources={
                r"/api/*": {
                    "origins": "*",
                    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"]
                }
            })
            
            # Initialize SocketIO with fallback
            socketio = SocketIO(
                app,
                cors_allowed_origins="*",
                async_mode='eventlet',
                ping_timeout=60,
                ping_interval=25,
                logger=False,
                engineio_logger=False
            )
            
            # Add health check endpoint
            @app.route('/health')
            def health_check():
                return {'status': 'healthy', 'mode': 'fallback'}, 200
            
            # Serve index.html for root
            @app.route('/')
            def index():
                return send_from_directory(str(PROJECT_ROOT), 'index.html')
            
            # Serve static files
            @app.route('/<path:path>')
            def serve_static(path):
                # Check common static directories
                static_dirs = ['frontend', 'css', 'js', 'assets', 'pages']
                for static_dir in static_dirs:
                    if path.startswith(static_dir):
                        file_path = PROJECT_ROOT / path
                        if file_path.exists():
                            return send_from_directory(str(file_path.parent), file_path.name)
                
                # Try serving from root
                file_path = PROJECT_ROOT / path
                if file_path.exists() and file_path.is_file():
                    return send_from_directory(str(file_path.parent), file_path.name)
                
                # 404 fallback
                return send_from_directory(str(PROJECT_ROOT), '404.html'), 404
            
            logger.warning("⚠️  Running in fallback mode with limited functionality")
            return app, socketio
            
        except Exception as fallback_error:
            logger.critical(f"Failed to create fallback application: {fallback_error}")
            raise RuntimeError("Unable to initialize application") from fallback_error

# Initialize the application
try:
    app, socketio = initialize_app()
    
    # Log deployment information
    logger.info(f"""
    ========================================
    ChatVLA Production Deployment
    ========================================
    Environment: {os.environ.get('FLASK_ENV', 'production')}
    Python Path: {sys.path[:2]}
    Port: {os.environ.get('PORT', 5000)}
    ========================================
    """)
    
except Exception as e:
    logger.critical(f"Application initialization failed: {e}")
    # Create minimal error app for debugging
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def error_page():
        return f"""
        <h1>Application Initialization Error</h1>
        <p>The application failed to start properly.</p>
        <pre>{str(e)}</pre>
        <p>Please check the logs for more information.</p>
        """, 500
    
    socketio = None

# Export for WSGI servers
if socketio:
    application = socketio
else:
    application = app

# Development server runner
if __name__ == "__main__":
    if socketio:
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting development server on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Cannot start development server: SocketIO not initialized")