#!/bin/bash

echo "🚀 ChatVLA Backend Deployment Script"
echo "====================================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    echo ""
    echo "To install Railway CLI, run:"
    echo "  brew install railway (macOS)"
    echo "  or"
    echo "  npm install -g @railway/cli"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Check if logged in to Railway
echo "Checking Railway login status..."
if ! railway whoami &> /dev/null; then
    echo "📝 Please log in to Railway:"
    railway login
fi

echo ""
echo "Starting deployment process..."
echo ""

# Option 1: Deploy with Railway CLI
echo "Option 1: Deploy with Railway CLI (Recommended)"
echo "================================================"
echo ""
echo "Run these commands:"
echo ""
echo "  # Create new project and link it"
echo "  railway link"
echo ""
echo "  # Deploy the project"
echo "  railway up"
echo ""
echo "  # Get your deployment URL"
echo "  railway domain"
echo ""

# Option 2: Deploy via GitHub
echo "Option 2: Deploy via GitHub"
echo "==========================="
echo ""
echo "1. Push your code to GitHub:"
echo "   git add ."
echo "   git commit -m 'Deploy backend to Railway'"
echo "   git push origin main"
echo ""
echo "2. Go to https://railway.app/new"
echo "3. Click 'Deploy from GitHub repo'"
echo "4. Select your 'chatVLA' repository"
echo "5. Railway will auto-detect the Dockerfile"
echo "6. Wait for deployment (3-5 minutes)"
echo "7. Get your URL from the deployment page"
echo ""

# Option 3: Quick deploy with nixpacks
echo "Option 3: Quick Deploy (No Docker)"
echo "==================================="
echo ""
echo "If Docker build fails, Railway can auto-build with nixpacks:"
echo ""
echo "1. Delete or rename Dockerfile temporarily:"
echo "   mv Dockerfile Dockerfile.backup"
echo ""
echo "2. Create nixpacks.toml:"
cat > nixpacks.toml.example << 'EOF'
[phases.setup]
nixPkgs = ["python311", "gcc"]

[phases.install]
cmds = ["pip install -r requirements-minimal.txt"]

[start]
cmd = "gunicorn --bind 0.0.0.0:$PORT wsgi:app"
EOF
echo ""
echo "3. Deploy with:"
echo "   railway up"
echo ""

echo "📝 IMPORTANT: After deployment, update your frontend!"
echo "======================================================"
echo ""
echo "1. Get your Railway URL (e.g., https://chatvla-production.up.railway.app)"
echo ""
echo "2. Update frontend/js/config.js line 26:"
echo "   return 'YOUR_RAILWAY_URL_HERE';"
echo ""
echo "3. Commit and push to GitHub:"
echo "   git add frontend/js/config.js"
echo "   git commit -m 'Update backend URL'"
echo "   git push origin main"
echo ""
echo "4. Test at https://tunerobotics.xyz/frontend/pages/databench.html"
echo ""
echo "Need help? Check deployment logs with: railway logs"