// Authentication module for Tune Robotics
const AUTH_TOKEN_KEY = 'tune_auth_token';
const AUTH_USER_KEY = 'tune_auth_user';

// API base URL
const API_BASE_URL = window.location.origin + '/api';

// Authentication state manager
class AuthManager {
  constructor() {
    this.token = localStorage.getItem(AUTH_TOKEN_KEY);
    this.user = this.getStoredUser();
  }

  getStoredUser() {
    const userStr = localStorage.getItem(AUTH_USER_KEY);
    try {
      return userStr ? JSON.parse(userStr) : null;
    } catch {
      return null;
    }
  }

  isAuthenticated() {
    return !!this.token && !!this.user;
  }

  setAuthData(token, user) {
    this.token = token;
    this.user = user;
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
  }

  clearAuthData() {
    this.token = null;
    this.user = null;
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
  }

  getAuthHeaders() {
    return this.token ? { 'Authorization': `Bearer ${this.token}` } : {};
  }

  async verifyToken() {
    if (!this.token) return false;

    try {
      const response = await fetch(`${API_BASE_URL}/auth/verify`, {
        headers: {
          ...this.getAuthHeaders(),
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        this.user = data.user;
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
        return true;
      } else {
        this.clearAuthData();
        return false;
      }
    } catch (error) {
      console.error('Token verification failed:', error);
      return false;
    }
  }

  redirectToLogin() {
          window.location.href = '/pages/login.html?redirect=' + encodeURIComponent(window.location.pathname);
  }

  redirectAfterLogin() {
    const urlParams = new URLSearchParams(window.location.search);
    const redirect = urlParams.get('redirect');
    window.location.href = redirect || '/';
  }
}

// Create global auth manager instance
const authManager = new AuthManager();

// Helper functions for form handling
function showError(message) {
  const errorEl = document.getElementById('errorMessage');
  if (errorEl) {
    errorEl.textContent = message;
    errorEl.classList.add('show');
    setTimeout(() => errorEl.classList.remove('show'), 5000);
  }
}

function showSuccess(message) {
  const successEl = document.getElementById('successMessage');
  if (successEl) {
    successEl.textContent = message;
    successEl.classList.add('show');
    setTimeout(() => successEl.classList.remove('show'), 5000);
  }
}

function setLoading(button, loading) {
  if (loading) {
    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.textContent = 'Loading...';
  } else {
    button.disabled = false;
    button.textContent = button.dataset.originalText || button.textContent;
  }
}

// Handle login form
async function handleLogin(e) {
  e.preventDefault();
  
  const form = e.target;
  const button = form.querySelector('button[type="submit"]');
  const username = form.username.value.trim();
  const password = form.password.value;

  if (!username || !password) {
    showError('Please fill in all fields');
    return;
  }

  setLoading(button, true);

  try {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ username, password })
    });

    const data = await response.json();

    if (response.ok) {
      authManager.setAuthData(data.token, data.user);
      showSuccess('Login successful! Redirecting...');
      setTimeout(() => authManager.redirectAfterLogin(), 1000);
    } else {
      showError(data.error || 'Login failed');
    }
  } catch (error) {
    showError('Network error. Please try again.');
    console.error('Login error:', error);
  } finally {
    setLoading(button, false);
  }
}

// Handle registration form
async function handleRegister(e) {
  e.preventDefault();
  
  const form = e.target;
  const button = form.querySelector('button[type="submit"]');
  const username = form.username.value.trim();
  const email = form.email.value.trim();
  const password = form.password.value;
  const confirmPassword = form.confirmPassword.value;

  // Validate form
  if (!username || !email || !password || !confirmPassword) {
    showError('Please fill in all fields');
    return;
  }

  if (password !== confirmPassword) {
    showError('Passwords do not match');
    return;
  }

  if (password.length < 6) {
    showError('Password must be at least 6 characters');
    return;
  }

  setLoading(button, true);

  try {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ username, email, password })
    });

    const data = await response.json();

    if (response.ok) {
      authManager.setAuthData(data.token, data.user);
      showSuccess('Registration successful! Redirecting...');
      setTimeout(() => authManager.redirectAfterLogin(), 1500);
    } else {
      showError(data.error || 'Registration failed');
    }
  } catch (error) {
    showError('Network error. Please try again.');
    console.error('Registration error:', error);
  } finally {
    setLoading(button, false);
  }
}

// Initialize auth forms
document.addEventListener('DOMContentLoaded', () => {
  // Handle login form
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
  }

  // Handle registration form
  const registerForm = document.getElementById('registerForm');
  if (registerForm) {
    registerForm.addEventListener('submit', handleRegister);
  }

  // Check if user is already logged in on auth pages
  if ((loginForm || registerForm) && authManager.isAuthenticated()) {
    authManager.redirectAfterLogin();
  }
});

// Export for use in other scripts
window.authManager = authManager; 