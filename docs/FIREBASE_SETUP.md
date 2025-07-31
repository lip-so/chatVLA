# Firebase Authentication Setup Guide

## 1. Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project"
3. Enter project name (e.g., "tune-robotics")
4. Enable Google Analytics (optional)
5. Click "Create project"

## 2. Enable Authentication

1. In your Firebase project, go to "Authentication"
2. Click "Get started"
3. Go to "Sign-in method" tab
4. Enable "Email/Password" provider
5. Click "Save"

## 3. Get Firebase Configuration

1. Go to Project Settings (gear icon)
2. Scroll down to "Your apps"
3. Click "Add app" and select Web (</>) 
4. Register your app with a nickname
5. Copy the Firebase config object

## 4. Update Frontend Configuration

Replace the config in `frontend/js/firebase-auth.js`:

```javascript
const firebaseConfig = {
  apiKey: "your-actual-api-key",
  authDomain: "your-project-id.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project-id.appspot.com",
  messagingSenderId: "your-sender-id",
  appId: "your-app-id"
};
```

## 5. Generate Service Account (Backend)

1. Go to Project Settings > Service accounts
2. Click "Generate new private key"
3. Download the JSON file
4. Save as `backend/firebase-service-account.json`

**Or use environment variables:**

Set `FIREBASE_CONFIG` environment variable with the service account JSON:

```bash
export FIREBASE_CONFIG='{"type": "service_account", "project_id": "your-project", ...}'
```

## 6. Update HTML Files

Update your login and register pages to use Firebase SDK:

### Login Page (`frontend/pages/login.html`)
Add before closing `</body>` tag:
```html
<script type="module">
  import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
  import { getAuth } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js';
  
  // Your config here
  const firebaseConfig = { /* your config */ };
  
  const app = initializeApp(firebaseConfig);
  window.firebaseApp = app;
  window.firebaseAuth = getAuth(app);
</script>
<script src="/js/firebase-auth.js" type="module"></script>
```

## 7. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 8. Run with Firebase

```bash
cd backend
python api/main.py
```

## Environment Variables

For production, set these environment variables:

- `FIREBASE_CONFIG`: Service account JSON string
- `FIREBASE_SERVICE_ACCOUNT_PATH`: Path to service account file
- `SECRET_KEY`: Flask secret key

## Testing

1. Visit your site
2. Try registering a new user
3. Check Firebase Console > Authentication > Users
4. Test login/logout functionality

## Security Rules (Optional)

In Firebase Console > Firestore Database > Rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

Your authentication system is now powered by Firebase! 🔥 