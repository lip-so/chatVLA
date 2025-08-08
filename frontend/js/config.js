// API Configuration for ChatVLA
const AppConfig = {
  // Get the appropriate API URL based on environment
  getApiUrl() {
    // Prefer same-origin in production to avoid CORS and mixed-content issues
    try {
      const host = window.location.hostname;
      if (host === 'localhost' || host === '127.0.0.1') {
        return 'http://localhost:5000';
      }
      // Allow manual override via global variable if defined
      if (window.APP_API_URL && typeof window.APP_API_URL === 'string') {
        return window.APP_API_URL;
      }
      return window.location.origin;
    } catch (e) {
      // Fallback to relative root
      return '';
    }
  },
  
  // API endpoints mapping
  endpoints: {
    databench: {
      evaluate: '/api/databench/evaluate',
      status: '/api/databench/status',
      results: '/api/databench/results'
    },
    plugplay: {
      detect: '/api/plugplay/detect',
      install: '/api/plugplay/install',
      configure: '/api/plugplay/configure'
    },
    auth: {
      login: '/api/auth/login',
      register: '/api/auth/register',
      logout: '/api/auth/logout',
      verify: '/api/auth/verify'
    }
  },
  
  // Get full endpoint URL
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
      return { 
        status: 'error', 
        message: 'Cannot connect to backend',
        error: error.message 
      };
    }
  },
  
  // Show connection status banner
  async showConnectionStatus() {
    const status = await this.checkBackendHealth();
    const isHealthy = status.status === 'healthy' || status.status === 'ok';
    
    if (!isHealthy) {
      const banner = document.createElement('div');
      banner.className = 'alert alert-warning';
      banner.style.cssText = 'position: fixed; top: 60px; left: 50%; transform: translateX(-50%); z-index: 9999; padding: 10px 20px; border-radius: 5px;';
      banner.innerHTML = `
        <strong>⚠️ Backend Connection Issue</strong><br>
        Railway backend may be starting up. Please wait a moment and refresh.
      `;
      document.body.appendChild(banner);
      
      setTimeout(() => banner.remove(), 10000);
    }
  }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  console.log('API URL:', AppConfig.getApiUrl());
  
  // Only check health for pages that need the backend
  const needsBackend = [
    'databench.html',
    'vision.html',
    'plug-and-play-databench-style.html'
  ].some(page => window.location.pathname.includes(page));
  
  if (needsBackend) {
    AppConfig.showConnectionStatus();
  }
});

// Export for use in other scripts
window.AppConfig = AppConfig;
