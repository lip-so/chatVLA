// Firebase Authentication for browser (no modules)
// Load Firebase scripts first, then this file

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBWiDdaqFp8wiKV7yVbdhJdBdrQNn-V8O4",
  authDomain: "tune-robotics.firebaseapp.com",
  projectId: "tune-robotics",
  storageBucket: "tune-robotics.firebasestorage.app",
  messagingSenderId: "618052352489",
  appId: "1:618052352489:web:c64395ba84aa7e8bc13378",
  measurementId: "G-95T7DVMYXW"
};

// Initialize Firebase when the script loads
let app, auth;

// Wait for Firebase to be available
function initializeFirebase() {
  if (typeof firebase !== 'undefined') {
    // Firebase v9 compat mode
    app = firebase.initializeApp(firebaseConfig);
    auth = firebase.auth();
    
    // Make auth available globally
    window.firebaseAuth = auth;
    window.firebaseApp = app;
    
    console.log('Firebase initialized successfully');
    
    // Set up auth state listener
    auth.onAuthStateChanged((user) => {
      console.log('Auth state changed:', user ? user.email : 'no user');
      
      if (window.onAuthStateChanged) {
        window.onAuthStateChanged(auth, user);
      }
      
      // Trigger custom auth event
      window.dispatchEvent(new CustomEvent('authStateChanged', { detail: user }));
    });
    
    return true;
  }
  return false;
}

// Firebase Authentication Manager
class FirebaseAuthManager {
  constructor() {
    this.user = null;
    this.token = null;
    this.isReady = false;
    
    // Initialize when Firebase is ready
    const checkFirebase = setInterval(() => {
      if (initializeFirebase()) {
        clearInterval(checkFirebase);
        this.isReady = true;
        
        // Listen for auth state changes
        auth.onAuthStateChanged(async (user) => {
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
          
          console.log('FirebaseAuthManager: User state updated', this.user);
        });
      }
    }, 100);
  }

  async signUp(email, password, displayName = null) {
    try {
      const userCredential = await auth.createUserWithEmailAndPassword(email, password);
      
      if (displayName) {
        await userCredential.user.updateProfile({ displayName });
      }
      
      return { success: true, user: userCredential.user };
    } catch (error) {
      console.error('Sign up error:', error);
      return { success: false, error: error.message };
    }
  }

  async signIn(email, password) {
    try {
      const userCredential = await auth.signInWithEmailAndPassword(email, password);
      return { success: true, user: userCredential.user };
    } catch (error) {
      console.error('Sign in error:', error);
      return { success: false, error: error.message };
    }
  }

  async signOut() {
    try {
      await auth.signOut();
      return { success: true };
    } catch (error) {
      console.error('Sign out error:', error);
      return { success: false, error: error.message };
    }
  }

  getCurrentUser() {
    return auth.currentUser;
  }

  getAuthHeaders() {
    if (this.token) {
      return {
        'Authorization': `Bearer ${this.token}`,
        'X-User-ID': this.user?.uid || ''
      };
    }
    return {};
  }

  redirectAfterLogin() {
    const urlParams = new URLSearchParams(window.location.search);
    const redirect = urlParams.get('redirect');
    window.location.href = redirect || '/';
  }
}

// Global auth state change handler
window.onAuthStateChanged = (auth, user) => {
  console.log('Global auth state changed:', user ? user.email : 'no user');
};

// Initialize auth manager when script loads
window.authManager = new FirebaseAuthManager();

console.log('Firebase auth browser script loaded'); 