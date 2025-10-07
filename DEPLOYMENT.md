# AI-manvian Deployment Guide

This guide covers deploying the AI-manvian FastAPI service to various free platforms.

## 🌟 Quick Comparison

| Platform | Free Tier | Cold Starts | Setup Difficulty | Best For |
|----------|-----------|-------------|------------------|----------|
| **Render** | 750 hrs/mo | Yes (~30s) | Easy | Production apps |
| **Railway** | $5 credit/mo | Minimal | Very Easy | Quick deploys |
| **Fly.io** | 3 VMs | Fast (~2s) | Medium | Global apps |
| **Google Cloud Run** | 2M req/mo | Yes (~10s) | Medium | Scalable apps |
| **Vercel** | Unlimited | Minimal | Easy | Serverless |

---

## 1. Render Deployment (RECOMMENDED) ⭐

### Prerequisites
- GitHub account
- Render account (render.com)

### Steps

1. **Push to GitHub**
   ```bash
   cd AI-manvian
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/ai-manvian.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to https://render.com
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`
   - Add environment variable:
     - Key: `GROQ_API_KEY`
     - Value: Your Groq API key
   - Click "Create Web Service"

3. **Get Your URL**
   ```
   https://ai-manvian-xxxxx.onrender.com
   ```

4. **Update Frontend**
   ```bash
   # Update .env.local in manvian-nextjs-fe
   NEXT_PUBLIC_AI_API_URL=https://ai-manvian-xxxxx.onrender.com
   ```

### Notes
- Free tier spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- 750 hours/month is enough for most apps

---

## 2. Railway Deployment 🚂

### Steps

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login and Initialize**
   ```bash
   cd AI-manvian
   railway login
   railway init
   ```

3. **Deploy**
   ```bash
   railway up
   ```

4. **Add Environment Variables**
   ```bash
   railway variables set GROQ_API_KEY=your-key-here
   ```

5. **Generate Domain**
   ```bash
   railway domain
   ```

### Notes
- $5 free credit per month
- No automatic spin-down
- Faster cold starts than Render

---

## 3. Fly.io Deployment 🚀

### Steps

1. **Install flyctl**
   ```bash
   # macOS
   brew install flyctl
   
   # Linux/WSL
   curl -L https://fly.io/install.sh | sh
   
   # Windows
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. **Login**
   ```bash
   flyctl auth login
   ```

3. **Launch App**
   ```bash
   cd AI-manvian
   fly launch
   # Follow prompts, it will auto-generate fly.toml
   ```

4. **Set Secrets**
   ```bash
   fly secrets set GROQ_API_KEY=your-key-here
   ```

5. **Deploy**
   ```bash
   fly deploy
   ```

6. **Get URL**
   ```bash
   fly status
   ```

### Notes
- Free tier: 3 shared-cpu VMs
- Fast cold starts (~2 seconds)
- Global deployment options

---

## 4. Google Cloud Run ☁️

### Steps

1. **Install gcloud CLI**
   ```bash
   # Follow: https://cloud.google.com/sdk/docs/install
   ```

2. **Login and Setup**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Deploy**
   ```bash
   cd AI-manvian
   gcloud run deploy ai-manvian \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GROQ_API_KEY=your-key-here
   ```

4. **Get URL**
   The deployment will output your service URL

### Notes
- Free tier: 2 million requests/month
- Automatic scaling
- Cold starts ~5-10 seconds

---

## 5. Vercel Deployment (Serverless) ⚡

### Steps

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   cd AI-manvian
   vercel
   ```

3. **Add Environment Variable**
   ```bash
   vercel env add GROQ_API_KEY
   # Enter your Groq API key when prompted
   ```

4. **Redeploy**
   ```bash
   vercel --prod
   ```

### Notes
- Unlimited free deployments
- Serverless (may have limitations for long-running tasks)
- 10-second timeout on free tier

---

## 🔧 Environment Variables Required

All platforms need this environment variable:

```bash
GROQ_API_KEY=your-groq-api-key-from-console.groq.com
```

---

## 🧪 Testing Your Deployment

After deployment, test with:

```bash
# Health check
curl https://your-deployed-url.com/health

# Parse resume (use a sample PDF)
curl -X POST https://your-deployed-url.com/api/resume/parse \
  -F "file=@sample-resume.pdf"
```

---

## 📱 Update Frontend Configuration

After deploying, update your frontend environment:

```bash
# manvian-nextjs-fe/.env.local
NEXT_PUBLIC_AI_API_URL=https://your-deployed-url.com
```

---

## 🔍 Monitoring & Debugging

### Render
```bash
# View logs in dashboard or
render logs -s ai-manvian
```

### Railway
```bash
railway logs
```

### Fly.io
```bash
fly logs
```

### Google Cloud Run
```bash
gcloud run services logs read ai-manvian
```

---

## 💰 Cost Breakdown (Free Tiers)

| Platform | Requests/Month | Always On | Storage |
|----------|----------------|-----------|---------|
| Render | ~50K | No | 0 GB |
| Railway | ~100K | Yes* | 1 GB |
| Fly.io | Unlimited | Yes* | 3 GB |
| Cloud Run | 2M | No | 0 GB |
| Vercel | Unlimited | Yes | 0 GB |

*With free credits

---

## 🚨 Important Notes

1. **Cold Starts**: Most free tiers have cold starts. First request may be slow.

2. **CORS**: The app already has CORS configured for common domains. Update if needed in `app/main.py`.

3. **File Size Limits**: 
   - Render: 100MB
   - Railway: 100MB
   - Vercel: 4.5MB (may be too small for large resumes)
   - Cloud Run: 32MB

4. **Execution Timeout**:
   - Resume parsing can take 10-30 seconds
   - Ensure your platform allows this (most do on paid tiers)

5. **Database**: This service is stateless, no database needed.

---

## 🎯 Recommended Setup

**For Production**: Render (reliable, generous free tier)
**For Development**: Railway (fastest, easiest)
**For Scale**: Google Cloud Run (best free tier limits)

---

## 📞 Support

If deployment fails:
1. Check deployment logs
2. Verify GROQ_API_KEY is set
3. Ensure all dependencies in `requirements.txt`
4. Test locally first: `uvicorn app.main:app --reload`

Happy Deploying! 🚀

