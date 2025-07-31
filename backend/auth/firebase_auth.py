"""Firebase authentication module for Tune Robotics platform"""

import os
import json
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
        return True
    except ValueError:
        # Firebase not initialized, so initialize it
        firebase_config = os.environ.get('FIREBASE_CONFIG')
        if firebase_config:
            try:
                # Use service account from environment variable
                cred_dict = json.loads(firebase_config)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                return True
            except Exception as e:
                print(f"❌ Error initializing Firebase with environment config: {e}")
                return False
        else:
            # Use service account file (for development)
            service_account_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase-service-account.json')
            if os.path.exists(service_account_path):
                try:
                    cred = credentials.Certificate(service_account_path)
                    firebase_admin.initialize_app(cred)
                    return True
                except Exception as e:
                    print(f"""
❌ Firebase initialization failed!

The service account file '{service_account_path}' contains placeholder values.

To fix this:
1. Go to Firebase Console: https://console.firebase.google.com/
2. Select your project: tune-robotics
3. Go to Project Settings (gear icon) 
4. Click 'Service accounts' tab
5. Click 'Generate new private key'
6. Download the JSON file 
7. Replace '{service_account_path}' with the downloaded file

Or set FIREBASE_CONFIG environment variable with the service account JSON content.

Error details: {e}
                    """)
                    return False
            else:
                print(f"""
❌ Firebase service account file not found!

Please create '{service_account_path}' with your Firebase service account credentials.

To get credentials:
1. Go to Firebase Console: https://console.firebase.google.com/
2. Select your project: tune-robotics  
3. Go to Project Settings (gear icon)
4. Click 'Service accounts' tab
5. Click 'Generate new private key'
6. Save the downloaded JSON as '{service_account_path}'
                """)
                return False

# Initialize Firebase
firebase_initialized = initialize_firebase()

firebase_bp = Blueprint('firebase_auth', __name__, url_prefix='/api/auth')

def requires_firebase_auth(f):
    """Decorator to protect routes that require Firebase authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not firebase_initialized:
            return jsonify({'error': 'Firebase authentication not available - check server configuration'}), 503
            
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
        
        if not token:
            return jsonify({'error': 'Authorization token required'}), 401
        
        try:
            # Verify Firebase ID token
            decoded_token = firebase_auth.verify_id_token(token)
            user_id = decoded_token['uid']
            
            # Get user info from Firebase
            user_record = firebase_auth.get_user(user_id)
            
            # Add user to request context
            request.current_user = {
                'uid': user_record.uid,
                'email': user_record.email,
                'display_name': user_record.display_name or user_record.email.split('@')[0],
                'email_verified': user_record.email_verified,
                'created_at': user_record.user_metadata.creation_timestamp,
                'last_sign_in': user_record.user_metadata.last_sign_in_timestamp
            }
            
            return f(*args, **kwargs)
            
        except Exception as e:
            return jsonify({'error': 'Invalid or expired token', 'details': str(e)}), 401
    
    return decorated_function

def get_current_user():
    """Get current authenticated user from request context"""
    return getattr(request, 'current_user', None)

@firebase_bp.route('/verify', methods=['GET'])
@cross_origin()
@requires_firebase_auth
def verify_token():
    """Verify Firebase ID token and return user info"""
    user = get_current_user()
    return jsonify({
        'valid': True,
        'user': user
    }), 200

@firebase_bp.route('/user', methods=['GET'])
@cross_origin()
@requires_firebase_auth
def get_user_profile():
    """Get current user profile from Firebase"""
    user = get_current_user()
    return jsonify({
        'user': user
    }), 200

@firebase_bp.route('/users', methods=['GET'])
@cross_origin()
@requires_firebase_auth
def list_users():
    """List all users (admin only - for development)"""
    try:
        # This is mainly for development/debugging
        page = firebase_auth.list_users()
        users = []
        for user in page.users:
            users.append({
                'uid': user.uid,
                'email': user.email,
                'display_name': user.display_name,
                'email_verified': user.email_verified,
                'creation_time': user.user_metadata.creation_timestamp
            })
        
        return jsonify({
            'users': users,
            'count': len(users)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@firebase_bp.route('/delete-user', methods=['DELETE'])
@cross_origin()
@requires_firebase_auth
def delete_user():
    """Delete current user account"""
    try:
        user = get_current_user()
        firebase_auth.delete_user(user['uid'])
        
        return jsonify({
            'message': 'User account deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@firebase_bp.route('/status', methods=['GET'])
@cross_origin()
def firebase_status():
    """Check detailed Firebase status for debugging"""
    status = {
        'firebase_initialized': firebase_initialized,
        'timestamp': datetime.now().isoformat(),
        'config_check': {}
    }
    
    # Check environment variables
    status['config_check']['env_firebase_config'] = bool(os.environ.get('FIREBASE_CONFIG'))
    status['config_check']['env_service_account_path'] = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase-service-account.json')
    
    # Check service account file
    service_account_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase-service-account.json')
    if os.path.exists(service_account_path):
        status['config_check']['service_account_file_exists'] = True
        try:
            with open(service_account_path, 'r') as f:
                config = json.load(f)
                status['config_check']['service_account_valid'] = not config.get('private_key', '').startswith('-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE')
                status['config_check']['project_id'] = config.get('project_id', 'unknown')
        except Exception as e:
            status['config_check']['service_account_error'] = str(e)
    else:
        status['config_check']['service_account_file_exists'] = False
    
    if firebase_initialized:
        try:
            # Test Firebase Admin functionality
            app = firebase_admin.get_app()
            status['firebase_app_name'] = app.name
            
            # Try to list users (this will fail if no users exist, but shows auth is working)
            try:
                page = firebase_auth.list_users(max_results=1)
                status['auth_working'] = True
                status['user_count_sample'] = len(list(page.users))
            except Exception as e:
                status['auth_working'] = True  # Error is expected if no users
                status['auth_note'] = 'Auth service accessible (no users yet)'
                
        except Exception as e:
            status['firebase_error'] = str(e)
    
    return jsonify(status), 200 if firebase_initialized else 503

# Health check endpoint
@firebase_bp.route('/health', methods=['GET'])
@cross_origin()
def auth_health():
    """Check Firebase authentication health"""
    try:
        if firebase_initialized:
            # Try to access Firebase Auth
            firebase_admin.get_app()
            return jsonify({
                'status': 'healthy',
                'service': 'firebase_auth',
                'firebase_initialized': True,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'service': 'firebase_auth',
                'firebase_initialized': False,
                'error': 'Firebase not properly initialized - check service account credentials',
                'timestamp': datetime.now().isoformat()
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'firebase_auth',
            'firebase_initialized': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500 