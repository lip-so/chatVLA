# Plug and Play module initialization
try:
    from .app import app, socketio
except ImportError as e:
    # Handle import errors gracefully for deployment
    import sys
    print(f"Import error in plug_and_play: {e}", file=sys.stderr)
    app = None
    socketio = None
