#!/bin/bash

echo "🚀 RAILWAY DOMAIN GENERATOR"
echo "==========================="
echo ""
echo "Your Railway backend is DEPLOYED and RUNNING."
echo "We just need to expose it with a public domain."
echo ""
echo "📍 AUTOMATIC METHOD:"
echo "==================="
echo ""

# Check if railway CLI exists
if command -v railway &> /dev/null; then
    echo "Railway CLI found. Attempting automatic domain generation..."
    
    # Try to generate domain
    echo "Running: railway domain"
    railway domain
    
    echo ""
    echo "If you see a URL above, copy it and run:"
    echo "  ./update_backend_url.sh YOUR-URL"
else
    echo "Railway CLI not found."
    echo ""
    echo "To install Railway CLI:"
    echo "  brew install railway"
    echo ""
    echo "Or use npm:"
    echo "  npm install -g @railway/cli"
fi

echo ""
echo "📍 MANUAL METHOD (DO THIS NOW):"
echo "==============================="
echo ""
echo "1. I'm opening Railway dashboard for you..."
open "https://railway.app/dashboard"
sleep 2

echo ""
echo "2. Follow these EXACT steps:"
echo "   a) Click on your 'chatVLA' project"
echo "   b) Click on the service (green box)"
echo "   c) Click 'Settings' tab"
echo "   d) Scroll down to 'Networking' section"
echo "   e) Look for these options:"
echo ""
echo "   IF YOU SEE 'TCP Proxy':"
echo "   - Click the trash icon to DELETE it"
echo "   - Then 'Generate Domain' button will appear"
echo ""
echo "   IF YOU SEE 'Generate Domain':"
echo "   - Click it!"
echo ""
echo "   IF YOU SEE A URL ALREADY:"
echo "   - Copy it! That's your domain"
echo ""
echo "3. Once you have the URL (like chatvla-production-xxx.up.railway.app):"
echo "   Run: ./update_backend_url.sh https://YOUR-URL.up.railway.app"
echo ""
echo "================================"
echo "WHY THIS IS HAPPENING:"
echo "================================"
echo "Railway deployed your backend successfully ✅"
echo "But Railway doesn't auto-generate public URLs ❌"
echo "You must manually click 'Generate Domain' ⚠️"
echo ""
echo "Once you do this, your site will work perfectly!"
echo ""
echo "Need help? The Railway dashboard is now open."
echo "Look for 'Settings' → 'Networking' → 'Generate Domain'"