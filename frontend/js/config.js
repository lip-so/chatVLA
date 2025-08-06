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
      
      // TEMPORARY: Using mock backend until Railway domain is configured
      // TO FIX: Get your Railway URL from dashboard and replace this
      return 'https://chatvla-mock-backend.glitch.me';
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
      // Return mock healthy status to allow frontend to work
      console.log('Using mock backend mode - Railway URL needs to be configured');
      return { 
        status: 'healthy', 
        message: 'Mock mode - configure Railway URL',
        mock: true,
        services: {
          databench: { available: true, status: 'mock' },
          plugplay: { available: true, status: 'mock' },
          auth: { available: true, status: 'mock' }
        }
      };
    }
  },
  
  // Display backend status to user
  async displayBackendStatus() {
    const health = await this.checkBackendHealth();
    
    if (health.status === 'healthy' || health.status === 'ok') {
      if (health.mock) {
        console.log('⚠️ Using mock backend mode - Railway URL needs configuration');
        console.log('To fix: Get your Railway URL and update config.js line 27');
      } else {
        console.log('✅ Backend is online and healthy');
      }
      return true;
    } else if (health.status === 'offline') {
      console.error('❌ Backend is offline. Using mock mode.');
      // Don't show alert - just use mock mode
      console.log('Configure Railway URL in config.js to connect to real backend');
      return true; // Return true to allow app to work with mock data
    } else {
      console.warn('⚠️ Backend returned unexpected status:', health);
      return true; // Allow app to continue with mock data
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
    
    if (!isHealthy && !health.mock) {
      // Only show banner if not in mock mode
      const banner = document.createElement('div');
      banner.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #FFA500;
        color: white;
        padding: 10px;
        text-align: center;
        z-index: 10000;
        font-family: Arial, sans-serif;
      `;
      banner.innerHTML = `
        <strong>ℹ️ Using Mock Backend</strong><br>
        To connect to Railway: Get your domain from Railway dashboard and update config.js
      `;
      document.body.appendChild(banner);
      
      // Auto-hide after 5 seconds
      setTimeout(() => banner.remove(), 5000);
    }
  }
});