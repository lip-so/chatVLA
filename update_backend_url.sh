#!/bin/bash

# Check if URL was provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Railway URL"
    echo ""
    echo "Usage: ./update_backend_url.sh YOUR_RAILWAY_URL"
    echo "Example: ./update_backend_url.sh https://chatvla-production-abc123.up.railway.app"
    exit 1
fi

RAILWAY_URL=$1

echo "🚀 Updating Backend URL"
echo "======================="
echo ""
echo "New backend URL: $RAILWAY_URL"
echo ""

# Update the config.js file
sed -i.bak "s|return 'https://chatvla-production.up.railway.app';|return '$RAILWAY_URL';|g" frontend/js/config.js

if [ $? -eq 0 ]; then
    echo "✅ Updated frontend/js/config.js"
    echo ""
    
    # Test the backend
    echo "📡 Testing backend connection..."
    HEALTH_CHECK=$(curl -s "$RAILWAY_URL/health" 2>/dev/null)
    
    if [[ $HEALTH_CHECK == *"healthy"* ]]; then
        echo "✅ Backend is responding correctly!"
        echo ""
        echo "Response: $HEALTH_CHECK"
    else
        echo "⚠️  Backend might not be responding correctly"
        echo "Response: $HEALTH_CHECK"
    fi
    
    echo ""
    echo "📦 Committing and pushing changes..."
    git add frontend/js/config.js
    git commit -m "Update backend URL to $RAILWAY_URL"
    git push origin main
    
    echo ""
    echo "✅ Done! Your changes are pushed to GitHub"
    echo ""
    echo "⏳ GitHub Pages will update in 1-2 minutes"
    echo ""
    echo "🎯 Test your site at:"
    echo "   https://tunerobotics.xyz/frontend/pages/databench.html"
    echo ""
    echo "The 'Backend offline' error should be gone!"
else
    echo "❌ Failed to update config.js"
    echo "Please update it manually:"
    echo "  Edit: frontend/js/config.js"
    echo "  Line 26: return '$RAILWAY_URL';"
fi