// Firebase Authentication module for Tune Robotics
import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';

// Firebase configuration - Replace with your actual config
const firebaseConfig = {
  apiKey: "AIzaSyBWiDdaqFp8wiKV7yVbdhJdBdrQNn-V8O4",
  authDomain: "tune-robotics.firebaseapp.com",
  projectId: "tune-robotics",
  storageBucket: "tune-robotics.firebasestorage.app",
  messagingSenderId: "618052352489",
  appId: "1:618052352489:web:c64395ba84aa7e8bc13378",
  measurementId: "G-95T7DVMYXW"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Firebase Authentication Manager
class FirebaseAuthManager {
  constructor() {
    this.user = null;
    this.token = null;
    this.isReady = false;
    
    // Listen for auth state changes
    onAuthStateChanged(auth, async (user) => {
      if (user) {
        this.user = {
          uid: user.uid,
          email: user.email,
          displayName: user.displayName || user.email.split('@')[0],
          emailVerified: user.emailVerified,
          photoURL: user.photoURL
        };
        
        // Get fresh token
        try {
          this.token = await user.getIdToken();
        } catch (error) {
          console.error('Error getting token:', error);
          this.token = null;
        }
      } else {
        this.user = null;
        this.token = null;
      }
      
      this.isReady = true;
      
      // Trigger custom event for UI updates
      window.dispatchEvent(new CustomEvent('firebaseAuthStateChanged', {
        detail: { user: this.user, isAuthenticated: this.isAuthenticated() }
      }));
    });
  }

  isAuthenticated() {
    return !!this.user && !!this.token;
  }

  async getAuthHeaders() {
    if (!this.user) return {};
    
    try {
      // Always get fresh token
      const token = await this.user.getIdToken();
      this.token = token;
      return { 'Authorization': `Bearer ${token}` };
    } catch (error) {
      console.error('Error getting auth headers:', error);
      return {};
    }
  }

  async register(email, password, displayName) {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      
      // Update profile with display name
      if (displayName) {
        await updateProfile(userCredential.user, {
          displayName: displayName
        });
      }
      
      return {
        success: true,
        user: userCredential.user,
        message: 'Registration successful'
      };
    } catch (error) {
      return {
        success: false,
        error: this.getErrorMessage(error)
      };
    }
  }

  async login(email, password) {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      return {
        success: true,
        user: userCredential.user,
        message: 'Login successful'
      };
    } catch (error) {
      return {
        success: false,
        error: this.getErrorMessage(error)
      };
    }
  }

  async logout() {
    try {
      await signOut(auth);
      return {
        success: true,
        message: 'Logout successful'
      };
    } catch (error) {
      return {
        success: false,
        error: this.getErrorMessage(error)
      };
    }
  }

  getErrorMessage(error) {
    switch (error.code) {
      case 'auth/user-not-found':
        return 'No account found with this email address';
      case 'auth/wrong-password':
        return 'Incorrect password';
      case 'auth/email-already-in-use':
        return 'An account already exists with this email address';
      case 'auth/weak-password':
        return 'Password should be at least 6 characters';
      case 'auth/invalid-email':
        return 'Invalid email address';
      case 'auth/too-many-requests':
        return 'Too many failed attempts. Please try again later';
      default:
        return error.message || 'An error occurred. Please try again.';
    }
  }

  redirectToLogin() {
          window.location.href = window.location.origin + '/frontend/pages/login.html?redirect=' + encodeURIComponent(window.location.pathname);
  }

  redirectAfterLogin() {
    const urlParams = new URLSearchParams(window.location.search);
    const redirect = urlParams.get('redirect');
    window.location.href = redirect || '/';
  }

  async verifyToken() {
    if (!this.user) return false;
    
    try {
      // Get fresh token to verify it's still valid
      await this.user.getIdToken(true);
      return true;
    } catch (error) {
      console.error('Token verification failed:', error);
      return false;
    }
  }
}

// Create global Firebase auth manager instance
const firebaseAuthManager = new FirebaseAuthManager();

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
  const email = form.username.value.trim(); // Using username field for email
  const password = form.password.value;

  if (!email || !password) {
    showError('Please fill in all fields');
    return;
  }

  setLoading(button, true);

  try {
    const result = await firebaseAuthManager.login(email, password);
    
    if (result.success) {
      showSuccess('Login successful! Redirecting...');
      setTimeout(() => firebaseAuthManager.redirectAfterLogin(), 1000);
    } else {
      showError(result.error);
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
    const result = await firebaseAuthManager.register(email, password, username);
    
    if (result.success) {
      showSuccess('Registration successful! Redirecting...');
      setTimeout(() => firebaseAuthManager.redirectAfterLogin(), 1500);
    } else {
      showError(result.error);
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
  window.addEventListener('firebaseAuthStateChanged', (e) => {
    const { isAuthenticated } = e.detail;
    if ((loginForm || registerForm) && isAuthenticated) {
      firebaseAuthManager.redirectAfterLogin();
    }
  });
});

// Export for use in other scripts
window.firebaseAuthManager = firebaseAuthManager;
window.authManager = firebaseAuthManager; // Keep compatibility with existing code 