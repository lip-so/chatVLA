#!/bin/bash

echo "🔧 FORCING RAILWAY TO SHOW DOMAIN OPTION"
echo "========================================"
echo ""
echo "Railway isn't showing the domain option because it doesn't detect your web server."
echo "Let's fix that..."
echo ""

# Add PORT to railway.json
echo "Adding PORT configuration..."
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "numReplicas": 1,
    "startCommand": "python start.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  },
  "services": [
    {
      "name": "web",
      "port": 8080
    }
  ]
}
EOF

echo "✅ Updated railway.json with port configuration"
echo ""

# Update Dockerfile to ensure PORT is exposed
echo "Updating Dockerfile..."
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy app files
COPY . .

# Expose the port Railway expects
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Start the app
CMD ["python", "start.py"]
EOF

echo "✅ Updated Dockerfile"
echo ""

# Commit and push
echo "Pushing changes to trigger Railway rebuild..."
git add railway.json Dockerfile
git commit -m "Force Railway to detect web service with PORT configuration"
git push origin main

echo ""
echo "✅ Changes pushed! Railway is rebuilding now..."
echo ""
echo "WAIT 2 MINUTES for the rebuild to complete, then:"
echo ""
echo "1. Go to: https://railway.app/dashboard"
echo "2. Click your 'chatVLA' project"
echo "3. Click on your service"
echo "4. You should now see one of these:"
echo "   - A URL at the top of the service panel"
echo "   - 'Settings' tab → 'Domains' or 'Public Networking'"
echo "   - A globe icon 🌐 to expose the service"
echo ""
echo "If you still don't see it, try:"
echo "- Go to Variables tab"
echo "- Add: PORT = 8080"
echo "- Save (Railway will redeploy)"
echo ""
echo "Once you get your domain, run:"
echo "./update_backend_url.sh https://YOUR-DOMAIN.up.railway.app"