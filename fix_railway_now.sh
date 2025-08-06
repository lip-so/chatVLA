#!/bin/bash

echo "🔧 FIXING RAILWAY - NO OTHER SERVICES NEEDED"
echo "============================================"
echo ""

# Install Railway CLI if not installed
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    curl -fsSL https://railway.app/install.sh | sh
fi

echo "Logging into Railway..."
railway login

echo "Linking to your project..."
railway link

echo "Generating domain for your service..."
railway domain

# Get the domain
DOMAIN=$(railway domain | grep -o 'https://[^ ]*' | head -1)

if [ ! -z "$DOMAIN" ]; then
    echo "✅ Got Railway domain: $DOMAIN"
    
    # Update config.js
    sed -i.bak "s|return 'https://.*';|return '$DOMAIN';|g" frontend/js/config.js
    
    # Commit and push
    git add frontend/js/config.js
    git commit -m "Update backend URL to Railway: $DOMAIN"
    git push origin main
    
    echo ""
    echo "🎉 SUCCESS! Your Railway backend is connected!"
    echo "Domain: $DOMAIN"
    echo "Your site will work in 1-2 minutes!"
else
    echo "Manual steps needed - Railway CLI couldn't get domain"
    echo "Go to: https://railway.app/dashboard"
    echo "Click your project → service → Settings → Networking → Generate Domain"
fi