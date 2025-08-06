# RAILWAY - HOW TO GENERATE DOMAIN (Updated 2024)

## THE NEW RAILWAY UI:

### Step 1: Go to Your Service
1. Open: https://railway.app/dashboard
2. Click on your **chatVLA** project
3. Click on your **service** (the box that says "chatVLA" or shows your deployments)

### Step 2: Find the Domain Setting
Railway changed their UI. The domain generation is now in one of these places:

#### Option A: In the Service View
1. When you click on your service, look at the **top of the panel**
2. You should see tabs like: **Deployments | Variables | Settings | Logs**
3. Click **Settings**
4. Look for **"Public Networking"** or **"Domains"**
5. Click **"Generate Domain"** or **"Add Domain"**

#### Option B: Quick Actions
1. On your service card, look for a **globe icon** 🌐
2. Or look for **"Expose"** button
3. Click it to generate a public URL

#### Option C: Service Settings Direct Link
Try this direct link to your service settings:
https://railway.app/project/*/service/*/settings#public-networking

### Step 3: If You See NOTHING About Domains

This means your service might not be detecting the web server. Fix it:

1. **Check if deployment is Active**
   - Look at your latest deployment
   - Should show green/active, not red/failed

2. **Set PORT Variable**
   - Go to **Variables** tab
   - Add: `PORT = 8080`
   - Railway will redeploy automatically

3. **Wait 2 minutes, then refresh**
   - Go back to Settings
   - "Generate Domain" should appear

## ALTERNATIVE: Use Railway CLI

Install and generate domain via command line:
```bash
# Install Railway CLI
curl -fsSL https://railway.app/install.sh | sh

# Or with Homebrew
brew install railway

# Login
railway login

# Link to your project
railway link

# Generate domain
railway domain
```

## IF STILL NO DOMAIN OPTION:

Your service needs to be listening on a PORT. Let's verify:

1. Go to **Logs** tab in Railway
2. Look for: "Starting Flask app on port" or similar
3. If you see errors, the deployment failed

To fix:
```bash
# Push this simple fix
echo "PORT=8080" >> .env
git add .env
git commit -m "Add PORT for Railway"
git push origin main
```

## WHAT YOUR RAILWAY DASHBOARD SHOULD SHOW:

```
chatVLA (project)
  └── chatVLA (service) ← Click here
       ├── Deployments (tab)
       ├── Variables (tab)
       ├── Settings (tab) ← Then click here
       │    └── Public Networking
       │         └── Generate Domain ← Click this
       └── Logs (tab)
```

Once you find and click "Generate Domain", you'll get:
`https://chatvla-production-xxxx.up.railway.app`

Then run:
```bash
./update_backend_url.sh https://YOUR-URL.up.railway.app
```