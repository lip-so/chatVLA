"""
Mock authentication module for testing and development
Use when Firebase is not available or configured
"""
import os
from functools import wraps
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

# Mock Firebase blueprint
mock_firebase_bp = Blueprint('mock_auth', __name__, url_prefix='/api/auth')

def requires_mock_auth(f):
    """Mock decorator that allows requests without authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if we should bypass auth completely
        if os.environ.get('DISABLE_AUTH', 'false').lower() == 'true':
            # Add a mock user to request context
            request.current_user = {
                'uid': 'mock-user-id',
                'email': 'test@example.com',
                'display_name': 'Test User',
                'email_verified': True,
                'created_at': None,
                'last_sign_in': None
            }
            return f(*args, **kwargs)
        
        # Otherwise, require some form of basic auth or API key
        auth_header = request.headers.get('Authorization')
        api_key = request.headers.get('X-API-Key')
        
        # Accept any non-empty auth header or API key for testing
        if auth_header or api_key:
            request.current_user = {
                'uid': 'mock-user-id',
                'email': 'test@example.com',
                'display_name': 'Test User',
                'email_verified': True,
                'created_at': None,
                'last_sign_in': None
            }
            return f(*args, **kwargs)
        
        return jsonify({'error': 'Authentication required (mock mode)'}), 401
    
    return decorated_function

@mock_firebase_bp.route('/status', methods=['GET'])
@cross_origin()
def mock_status():
    """Mock Firebase status endpoint"""
    return jsonify({
        'firebase_initialized': False,
        'mock_mode': True,
        'auth_disabled': os.environ.get('DISABLE_AUTH', 'false').lower() == 'true',
        'message': 'Using mock authentication - Firebase not configured'
    })

@mock_firebase_bp.route('/health', methods=['GET'])
@cross_origin()
def mock_health():
    """Mock auth health check"""
    return jsonify({
        'status': 'healthy',
        'auth_type': 'mock',
        'firebase_available': False
    })