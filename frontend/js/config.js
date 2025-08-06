/**
 * Configuration for ChatVLA Frontend
 * Manages API endpoints based on deployment environment
 */

const AppConfig = {
  // Backend API Configuration
  getApiUrl() {
    // Check if we're on GitHub Pages (static hosting)
    const isGitHubPages = window.location.hostname === 'tunerobotics.xyz' || 
                         window.location.hostname.includes('github.io');
    
    // Check if we're running locally
    const isLocalhost = window.location.hostname === 'localhost' || 
                       window.location.hostname === '127.0.0.1';
    
    if (isGitHubPages) {
      // Production backend URL - UPDATE THIS WITH YOUR ACTUAL DEPLOYED BACKEND URL
      // Options:
      // 1. Railway: https://your-app-name.up.railway.app
      // 2. Render: https://your-app-name.onrender.com
      // 3. Heroku: https://your-app-name.herokuapp.com
      // 4. Custom server: https://api.tunerobotics.xyz
      
      // Currently using placeholder - REPLACE WITH YOUR ACTUAL URL
      return 'https://chatvla-production.up.railway.app';
    } else if (isLocalhost) {
      // Local development
      return 'http://localhost:5000';
    } else {
      // Deployed backend (same origin)
      return window.location.origin;
    }
  },
  
  // API Endpoints
  endpoints: {
    // DataBench endpoints
    databench: {
      evaluate: '/api/databench/evaluate',
      status: '/api/databench/status',
      results: '/api/databench/results'
    },
    
    // Plug & Play endpoints
    plugplay: {
      install: '/api/plugplay/install',
      detectPorts: '/api/plugplay/detect-ports',
      status: '/api/plugplay/status'
    },
    
    // Authentication endpoints
    auth: {
      login: '/api/auth/login',
      register: '/api/auth/register',
      logout: '/api/auth/logout',
      verify: '/api/auth/verify',
      status: '/api/auth/status'
    },
    
    // Health check
    health: '/health'
  },
  
  // Get full URL for an endpoint
  getEndpoint(category, endpoint) {
    const baseUrl = this.getApiUrl();
    const path = this.endpoints[category]?.[endpoint];
    
    if (!path) {
      console.error(`Unknown endpoint: ${category}.${endpoint}`);
      return null;
    }
    
    return `${baseUrl}${path}`;
  },
  
  // Check if backend is available
  async checkBackendHealth() {
    try {
      const healthUrl = `${this.getApiUrl()}/health`;
      const response = await fetch(healthUrl, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Backend health check:', data);
        return data;
      }
      
      return { status: 'error', message: `HTTP ${response.status}` };
    } catch (error) {
      console.error('Backend health check failed:', error);
      return { status: 'offline', message: error.message };
    }
  },
  
  // Display backend status to user
  async displayBackendStatus() {
    const health = await this.checkBackendHealth();
    
    if (health.status === 'healthy' || health.status === 'ok') {
      console.log('✅ Backend is online and healthy');
      return true;
    } else if (health.status === 'offline') {
      console.error('❌ Backend is offline. Please ensure the backend is deployed and running.');
      alert('Backend server is offline. Please contact support or try again later.');
      return false;
    } else {
      console.warn('⚠️ Backend returned unexpected status:', health);
      return false;
    }
  }
};

// Export for use in other scripts
window.AppConfig = AppConfig;

// Check backend health on page load
document.addEventListener('DOMContentLoaded', async () => {
  console.log('Checking backend connectivity...');
  console.log('API URL:', AppConfig.getApiUrl());
  
  // Only check health for pages that need the backend
  const needsBackend = [
    'databench.html',
    'vision.html',
    'port-detection.html'
  ];
  
  const currentPage = window.location.pathname.split('/').pop();
  
  if (needsBackend.includes(currentPage)) {
    const isHealthy = await AppConfig.displayBackendStatus();
    
    if (!isHealthy) {
      // Add a warning banner to the page
      const banner = document.createElement('div');
      banner.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #ff4444;
        color: white;
        padding: 10px;
        text-align: center;
        z-index: 10000;
        font-family: Arial, sans-serif;
      `;
      banner.innerHTML = `
        <strong>⚠️ Backend Connection Error</strong><br>
        The backend server is currently unavailable. 
        Please ensure it's deployed to Railway/Render and update the API URL in config.js
      `;
      document.body.appendChild(banner);
    }
  }
});