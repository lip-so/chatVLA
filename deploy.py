#!/usr/bin/env python3
"""
Deployment entry point for production environments
"""

import os
import sys
from pathlib import Path

def main():
    """Main deployment entry point"""
    
    # Set environment variables for production
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('SECRET_KEY', os.environ.get('SECRET_KEY', 'change-this-in-production'))
    
    # Add backend to path
    backend_path = Path(__file__).parent / 'backend'
    sys.path.insert(0, str(backend_path))
    
    # Ensure all dependencies are available before importing
    try:
        # Check if databench dependencies are available
        import backend.databench.api
        print("✅ DataBench API available")
    except ImportError as e:
        print(f"⚠️  DataBench API not available: {e}")
    
    try:
        # Check if plug_and_play dependencies are available  
        import backend.plug_and_play.api
        print("✅ Plug & Play API available")
    except ImportError as e:
        print(f"⚠️  Plug & Play API not available: {e}")
    
    # Import and run the main app
    from api.main import app, socketio
    
    port = int(os.environ.get('PORT', 5000))
    
    print(f"""
    ========================================
    Tune Robotics Production Deployment
    ========================================
    Environment: {os.environ.get('FLASK_ENV', 'production')}
    Port: {port}
    
    Services Available:
    - DataBench API: /api/databench/*
    - Plug & Play API: /api/plugplay/*
    - Static Files: /css/*, /js/*, /assets/*, /pages/*
    - WebSocket: Enabled for real-time updates
    ========================================
    """)
    
    # Use SocketIO for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()