#!/bin/bash

echo "🔧 QUICK FIX: Using Local Backend Temporarily"
echo "=============================================="
echo ""
echo "Since Railway is giving you trouble, let's use a local backend for now..."
echo ""

# Update config.js to use localhost
cat > frontend/js/config.js << 'EOF'
// API Configuration for ChatVLA
const AppConfig = {
  // Get the appropriate API URL based on environment
  getApiUrl() {
    // TEMPORARY: Using localhost until Railway is fixed
    // TO FIX: Get Railway URL and replace the line below
    return 'http://localhost:8080';
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
        <strong>⚠️ Using Local Backend</strong><br>
        Make sure backend is running: <code>cd /Users/sofiia/chatVLA && python start.py</code>
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
EOF

echo "✅ Updated frontend to use local backend"
echo ""

# Kill any existing backend processes
echo "Stopping any existing backend processes..."
pkill -f "python start.py" 2>/dev/null
pkill -f "python backend_server.py" 2>/dev/null
pkill -f "python wsgi.py" 2>/dev/null
sleep 1

# Start local backend
echo "Starting local backend on port 8080..."
cd /Users/sofiia/chatVLA
python start.py &
BACKEND_PID=$!

sleep 2

# Test the backend
echo ""
echo "Testing local backend..."
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "✅ Local backend is running!"
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Your app is now working locally!"
    echo "=========================================="
    echo ""
    echo "1. Open: http://localhost:8080/health (to verify backend)"
    echo "2. Open: file:///Users/sofiia/chatVLA/index.html"
    echo "3. Navigate to DataBench and test it!"
    echo ""
    echo "The backend is running in background (PID: $BACKEND_PID)"
    echo "To stop it later: kill $BACKEND_PID"
    echo ""
    echo "=========================================="
    echo "TO FIX RAILWAY DEPLOYMENT LATER:"
    echo "=========================================="
    echo "1. Get your Railway URL from dashboard"
    echo "2. Run: ./update_backend_url.sh https://YOUR-RAILWAY-URL.up.railway.app"
    echo "3. Push to GitHub"
    echo "4. Site will work at https://tunerobotics.xyz"
else
    echo "❌ Backend failed to start"
    echo "Try manually: cd /Users/sofiia/chatVLA && python start.py"
fi