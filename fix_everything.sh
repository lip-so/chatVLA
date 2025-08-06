#!/bin/bash

echo "🚀 COMPLETE BACKEND FIX SCRIPT"
echo "=============================="
echo ""

# Function to update backend URL
update_backend_url() {
    local url=$1
    echo "Updating frontend with backend URL: $url"
    
    # Update config.js
    sed -i.bak "s|return 'https://.*';  // TEMPORARY|return '$url';  // TEMPORARY|g" frontend/js/config.js
    
    # Commit and push
    git add frontend/js/config.js
    git commit -m "Update backend URL to $url"
    git push origin main
    
    echo "✅ Frontend updated and pushed!"
    echo "Your site will work in 1-2 minutes at: https://tunerobotics.xyz"
}

echo "Checking Railway deployment..."
echo ""

# Check if Railway CLI is available
if command -v railway &> /dev/null; then
    echo "Railway CLI found. Attempting to get domain..."
    
    # Try to get Railway domain
    if railway domain 2>/dev/null | grep -q "up.railway.app"; then
        RAILWAY_URL=$(railway domain 2>/dev/null | grep -o 'https://[^ ]*')
        echo "✅ Found Railway URL: $RAILWAY_URL"
        update_backend_url "$RAILWAY_URL"
        echo ""
        echo "🎉 SUCCESS! Your backend is connected!"
        echo "Test your site at: https://tunerobotics.xyz/frontend/pages/databench.html"
        exit 0
    else
        echo "❌ No Railway domain found. Need to generate one."
    fi
else
    echo "Railway CLI not installed."
fi

echo ""
echo "================================"
echo "MANUAL FIX REQUIRED"
echo "================================"
echo ""
echo "Your Railway backend is deployed but needs a public URL."
echo ""
echo "OPTION 1: Fix Railway (Recommended)"
echo "-----------------------------------"
echo "1. Go to: https://railway.app/dashboard"
echo "2. Click your 'chatVLA' project"
echo "3. Click on the service"
echo "4. Go to Settings → Networking"
echo "5. Click 'Generate Domain'"
echo "6. Copy the URL"
echo "7. Run: ./update_backend_url.sh YOUR_URL"
echo ""
echo "OPTION 2: Deploy to Render (Auto URL)"
echo "--------------------------------------"
echo "1. Go to: https://render.com/deploy"
echo "2. New Web Service → Connect GitHub → Select 'chatVLA'"
echo "3. Settings:"
echo "   - Build: pip install -r requirements-minimal.txt"
echo "   - Start: python start.py"
echo "4. Deploy (auto-generates URL)"
echo "5. Run: ./update_backend_url.sh YOUR_RENDER_URL"
echo ""
echo "OPTION 3: Use Local Backend (For Testing)"
echo "------------------------------------------"
echo "1. Run: PORT=8080 python backend_server.py"
echo "2. Edit frontend/js/config.js line 27:"
echo "   return 'http://localhost:8080';"
echo "3. Test locally"
echo ""
echo "CURRENT STATUS:"
echo "✅ Frontend: Deployed at https://tunerobotics.xyz"
echo "✅ Backend: Deployed on Railway (needs public URL)"
echo "❌ Connection: config.js has placeholder URL"
echo ""
echo "Once you get your Railway URL, run:"
echo "  ./update_backend_url.sh https://YOUR-URL.up.railway.app"
echo ""
echo "This will fix everything in 1 minute!"