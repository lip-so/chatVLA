# 🚀 INSTANT FIX - Deploy to Glitch (2 minutes)

Since Railway is giving you trouble, let's use Glitch for an instant fix:

## Step 1: Deploy to Glitch (30 seconds)
1. **Click this link**: https://glitch.com/edit/#!/remix/glitch-hello-node
2. Once it opens, click the project name at top-left
3. Click "Import from GitHub"
4. Enter: `lip-so/chatVLA`
5. Click "OK"

## Step 2: Configure Glitch (30 seconds)
1. In Glitch editor, open `package.json`
2. Change the "start" script to:
   ```json
   "scripts": {
     "start": "python backend_server.py"
   }
   ```

3. Open `.env` file and add:
   ```
   PORT=3000
   ```

## Step 3: Get Your URL (30 seconds)
1. Click "Show" button → "In a New Window"
2. Copy the URL (e.g., `https://silent-amazing-vulture.glitch.me`)
3. Test it: `https://YOUR-PROJECT.glitch.me/health`

## Step 4: Update Your Site (30 seconds)
```bash
# Run this with YOUR Glitch URL:
./update_backend_url.sh https://YOUR-PROJECT.glitch.me

# Or manually edit frontend/js/config.js line 27:
# return 'https://YOUR-PROJECT.glitch.me';
```

## That's it! Your site will work in 1-2 minutes.

---

## Alternative: Use This Test Backend
While you set up your own, you can test with this mock backend:

```bash
# Update config.js line 27 to:
return 'https://jsonplaceholder.typicode.com';

# Then push:
git add frontend/js/config.js
git commit -m "Use test backend"
git push origin main
```

This will at least stop the "Backend offline" error while you fix Railway.