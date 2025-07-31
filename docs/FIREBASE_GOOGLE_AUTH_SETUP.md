# Firebase Google Authentication Setup Guide

## 🔥 Complete Firebase Setup with Google Auth

### **Step 1: Create Firebase Project**

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click **"Create a project"**
3. Enter project name: `tune-robotics`
4. Enable Google Analytics (optional)
5. Click **"Create project"**

### **Step 2: Enable Authentication**

1. In your Firebase project, go to **"Authentication"**
2. Click **"Get started"**
3. Go to **"Sign-in method"** tab
4. Enable **"Email/Password"** provider ✅
5. Enable **"Google"** provider ✅
   - Click on Google
   - Toggle **"Enable"**
   - Enter **Project support email** (your email)
   - Click **"Save"**

### **Step 3: Add Web App**

1. Go to **Project Settings** (gear icon)
2. Scroll down to **"Your apps"**
3. Click **"Add app"** and select **Web (</>) **
4. Enter app nickname: `tune-robotics-web`
5. **DO NOT** check "Also set up Firebase Hosting"
6. Click **"Register app"**
7. **Copy the Firebase config** (you'll need this!)

### **Step 4: Generate Service Account (Backend)**

1. Go to **Project Settings** > **Service accounts**
2. Click **"Generate new private key"**
3. Download the JSON file
4. Replace `backend/firebase-service-account.json` with the downloaded file

### **Step 5: Test Firebase Status**

Visit these URLs to check if Firebase is working:

```bash
# Basic health check
curl http://localhost:5000/api/auth/health

# Detailed status
curl http://localhost:5000/api/auth/status
```

### **Step 6: Test Authentication**

1. **Visit your site:** http://localhost:5000
2. **Click "Login"** or **"Sign Up"**
3. **Try Google sign-in** - should open Google popup
4. **Try email registration** - should work with any email

## 🧪 **How to Test if Firebase is Working**

### **Frontend Test (Browser Console):**
```javascript
// Open browser console on login/register page
console.log('Firebase Auth:', window.firebaseAuth);
console.log('Is authenticated:', window.authManager.isAuthenticated());

// Test Google sign-in manually
window.signInWithPopup(window.firebaseAuth, window.googleProvider)
  .then(result => console.log('Google sign-in success:', result))
  .catch(error => console.error('Google sign-in error:', error));
```

### **Backend Test (Terminal):**
```bash
# Test Firebase status
curl -s http://localhost:5000/api/auth/status | python -m json.tool

# Expected response when working:
# {
#   "firebase_initialized": true,
#   "auth_working": true,
#   "config_check": {
#     "service_account_valid": true,
#     "project_id": "tune-robotics"
#   }
# }
```

### **Authentication Flow Test:**
1. **Register with Google** → Should create user in Firebase Console
2. **Login with Google** → Should work seamlessly  
3. **Access protected pages** → Should work after authentication
4. **Logout** → Should redirect to homepage

## 🔧 **Troubleshooting**

### **Google Sign-in Not Working:**
- **Check Firebase Console** → Authentication → Sign-in method → Google is enabled
- **Check domain authorization** → Settings → Authorized domains includes `localhost`
- **Browser console errors** → Look for popup blocked or network errors

### **Service Account Issues:**
```bash
# Check if service account is valid
curl http://localhost:5000/api/auth/status
# Look for "service_account_valid": true
```

### **Common Errors:**
- **"Popup blocked"** → Allow popups in browser
- **"Domain not authorized"** → Add domain in Firebase Console
- **"Service account invalid"** → Re-download service account JSON
- **"Network error"** → Check internet connection

## 📱 **Production Setup**

### **Add Production Domain:**
1. Firebase Console → Authentication → Settings → Authorized domains
2. Add your production domain (e.g., `tunerobotics.com`)

### **Environment Variables:**
```bash
export FIREBASE_CONFIG='{"type":"service_account",...}'  # Full service account JSON
export SECRET_KEY='your-secure-secret-key'
```

## ✅ **Success Indicators**

**✅ Firebase Working When:**
- `curl http://localhost:5000/api/auth/status` shows `firebase_initialized: true`
- Google sign-in button opens Google popup
- Users appear in Firebase Console → Authentication → Users
- Protected pages redirect unauthenticated users to login

**✅ Google Auth Working When:**
- Clicking "Continue with Google" opens Google account selector
- Successful sign-in redirects to homepage with user greeting
- Firebase Console shows new users with Google provider

## 🎯 **Quick Test Checklist**

- [ ] Firebase project created
- [ ] Email/Password auth enabled  
- [ ] Google auth enabled
- [ ] Web app registered
- [ ] Service account downloaded and replaced
- [ ] Server running without Firebase errors
- [ ] Google sign-in popup opens
- [ ] Can register/login with Google
- [ ] Users appear in Firebase Console
- [ ] Protected pages work after auth

Your Firebase Google Authentication is ready! 🚀 