#!/bin/bash

echo "🚀 INSTALLING RAILWAY CLI & GENERATING DOMAIN"
echo "=============================================="
echo ""

# Install Railway CLI
echo "Installing Railway CLI..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Using Homebrew..."
        brew install railway
    else
        echo "Using install script..."
        curl -fsSL https://railway.app/install.sh | sh
    fi
else
    # Linux/Other
    curl -fsSL https://railway.app/install.sh | sh
fi

# Verify installation
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI installation failed"
    echo ""
    echo "Try manual install:"
    echo "  brew install railway"
    echo "Or:"
    echo "  npm install -g @railway/cli"
    exit 1
fi

echo "✅ Railway CLI installed successfully"
echo ""

# Login to Railway
echo "Logging into Railway..."
echo "This will open your browser for authentication"
railway login

echo ""
echo "Linking to your project..."
echo "Select 'chatVLA' when prompted"
railway link

echo ""
echo "Generating public domain..."
railway domain

# Try to get the domain
echo ""
echo "Checking for domain..."
DOMAIN=$(railway domain 2>/dev/null | grep -o 'https://[^ ]*' | head -1)

if [ ! -z "$DOMAIN" ]; then
    echo ""
    echo "✅ SUCCESS! Your Railway domain is:"
    echo "$DOMAIN"
    echo ""
    echo "Updating your frontend configuration..."
    
    # Update config.js
    sed -i.bak "s|return 'https://.*';|return '$DOMAIN';|g" frontend/js/config.js
    
    # Commit and push
    git add frontend/js/config.js
    git commit -m "Update backend URL to Railway: $DOMAIN"
    git push origin main
    
    echo ""
    echo "🎉 DONE! Your site is now connected to Railway!"
    echo "It will be live at https://tunerobotics.xyz in 1-2 minutes"
    echo ""
    echo "Test your backend: curl $DOMAIN/health"
else
    echo ""
    echo "⚠️  Couldn't get domain automatically"
    echo ""
    echo "Run this command to see your domain:"
    echo "  railway domain"
    echo ""
    echo "Then update your frontend:"
    echo "  ./update_backend_url.sh YOUR-DOMAIN"
fi