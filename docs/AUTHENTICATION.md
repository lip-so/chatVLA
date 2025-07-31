# Tune Robotics Authentication System

## Overview

The Tune Robotics platform now includes a comprehensive authentication system that protects access to platform features like DataBench, Plug & Play, and Vision.

## Features

- **User Registration**: New users can create accounts with username, email, and password
- **User Login**: Existing users can log in with username/email and password
- **JWT Authentication**: Secure token-based authentication using JSON Web Tokens
- **Protected Routes**: Platform features require authentication to access
- **Session Management**: Automatic token verification and refresh
- **Responsive UI**: Beautiful login and registration pages with modern design

## Backend Setup

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

```bash
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-in-production

# JWT Configuration
JWT_SECRET=your-jwt-secret-key-change-in-production

# Database Configuration (defaults to SQLite)
DATABASE_URL=sqlite:///tune_robotics.db

# Server Configuration
PORT=5000
```

### Installation

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. The database will be created automatically when you first run the server.

## API Endpoints

### Authentication Endpoints

- `POST /api/auth/register` - Register a new user
  - Body: `{ "username": "string", "email": "string", "password": "string" }`
  - Returns: `{ "token": "jwt_token", "user": {...} }`

- `POST /api/auth/login` - Login user
  - Body: `{ "username": "string_or_email", "password": "string" }`
  - Returns: `{ "token": "jwt_token", "user": {...} }`

- `GET /api/auth/verify` - Verify token (requires auth)
  - Headers: `Authorization: Bearer <token>`
  - Returns: `{ "valid": true, "user": {...} }`

- `POST /api/auth/logout` - Logout user (requires auth)
  - Headers: `Authorization: Bearer <token>`
  - Returns: `{ "message": "Logout successful" }`

- `GET /api/auth/profile` - Get user profile (requires auth)
  - Headers: `Authorization: Bearer <token>`
  - Returns: `{ "user": {...} }`

- `PUT /api/auth/profile` - Update user profile (requires auth)
  - Headers: `Authorization: Bearer <token>`
  - Body: `{ "email": "string", "password": "string" }` (optional fields)
  - Returns: `{ "user": {...} }`

### Protected Endpoints

All DataBench and Plug & Play endpoints now require authentication:

- `/api/databench/*` - All DataBench endpoints
- `/api/plugplay/*` - All Plug & Play endpoints

Include the JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## Frontend Integration

### Authentication State

The authentication state is managed by the `authManager` object available globally:

```javascript
// Check if user is authenticated
if (authManager.isAuthenticated()) {
  // User is logged in
}

// Get current user
const user = authManager.user;

// Get auth headers for API calls
const headers = authManager.getAuthHeaders();

// Logout
authManager.clearAuthData();
```

### Protected Pages

The following pages are protected and require authentication:
- `/pages/databench.html`
- `/pages/plug-and-play.html`
- `/pages/port-detection.html`
- `/pages/vision.html`

Users will be automatically redirected to the login page if they try to access these pages without authentication.

### Navigation Bar

The navigation bar automatically updates based on authentication state:
- **Logged out**: Shows "Login" and "Sign Up" buttons
- **Logged in**: Shows "Hi, username" and "Logout" link

## Security Considerations

1. **Password Security**: Passwords are hashed using bcrypt before storage
2. **JWT Tokens**: Tokens expire after 24 hours
3. **HTTPS**: Always use HTTPS in production to protect token transmission
4. **Secret Keys**: Change the default secret keys in production
5. **CORS**: Configure CORS appropriately for your deployment

## Database Schema

### Users Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| username | String(80) | Unique username |
| email | String(120) | Unique email address |
| password_hash | String(255) | Bcrypt hashed password |
| created_at | DateTime | Account creation timestamp |
| updated_at | DateTime | Last update timestamp |
| is_active | Boolean | Account active status |

## Troubleshooting

### Common Issues

1. **"Authorization token required"**: Make sure you're including the JWT token in the Authorization header
2. **"Invalid or expired token"**: The token may have expired. Log in again to get a new token
3. **"Username already exists"**: Choose a different username during registration
4. **Database errors**: Make sure the database file has proper write permissions

### Development Tips

- The SQLite database file (`tune_robotics.db`) will be created in the backend directory
- You can use tools like DB Browser for SQLite to inspect the database
- JWT tokens can be decoded at jwt.io for debugging (never share your secret key)

## Future Enhancements

Potential improvements to the authentication system:
- Email verification for new accounts
- Password reset functionality
- Social authentication (Google, GitHub)
- Two-factor authentication
- Role-based access control
- API rate limiting 