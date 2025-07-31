"""Authentication routes and JWT handling"""

import os
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
from sqlalchemy.exc import IntegrityError

from .models import db, User

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# JWT configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token):
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def requires_auth(f):
    """Decorator to protect routes that require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
        
        if not token:
            return jsonify({'error': 'Authorization token required'}), 401
        
        payload = decode_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Get user from database
        user = User.query.get(payload['user_id'])
        if not user or not user.is_active:
            return jsonify({'error': 'User not found or inactive'}), 401
        
        # Add user to request context
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

def get_current_user():
    """Get current authenticated user from request context"""
    return getattr(request, 'current_user', None)

@auth_bp.route('/register', methods=['POST'])
@cross_origin()
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate input
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
        token = generate_token(user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'token': token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
@cross_origin()
def login():
    """Login user with username/email and password"""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['username', 'password']):
            return jsonify({'error': 'Missing username and password'}), 400
        
        username_or_email = data['username'].strip()
        password = data['password']
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | 
            (User.email == username_or_email.lower())
        ).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is inactive'}), 401
        
        # Generate token
        token = generate_token(user.id)
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/verify', methods=['GET'])
@cross_origin()
@requires_auth
def verify_token():
    """Verify JWT token and return user info"""
    user = get_current_user()
    return jsonify({
        'valid': True,
        'user': user.to_dict()
    }), 200

@auth_bp.route('/logout', methods=['POST'])
@cross_origin()
@requires_auth
def logout():
    """Logout user (client should remove token)"""
    # In a stateless JWT system, logout is handled client-side
    # This endpoint can be used for logging or token blacklisting if needed
    return jsonify({'message': 'Logout successful'}), 200

@auth_bp.route('/profile', methods=['GET'])
@cross_origin()
@requires_auth
def get_profile():
    """Get current user profile"""
    user = get_current_user()
    return jsonify({
        'user': user.to_dict()
    }), 200

@auth_bp.route('/profile', methods=['PUT'])
@cross_origin()
@requires_auth
def update_profile():
    """Update user profile"""
    try:
        user = get_current_user()
        data = request.get_json()
        
        # Update allowed fields
        if 'email' in data:
            new_email = data['email'].strip().lower()
            if new_email != user.email:
                # Check if email already exists
                if User.query.filter_by(email=new_email).first():
                    return jsonify({'error': 'Email already in use'}), 409
                user.email = new_email
        
        if 'password' in data:
            if len(data['password']) < 6:
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            user.set_password(data['password'])
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500 