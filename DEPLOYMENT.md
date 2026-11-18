# GradeForge Deployment Guide

## Quick Start: Deploy to Render (Free)

### Step 1: Create a Render Account
1. Go to https://render.com
2. Sign up with GitHub (easiest)
3. Connect your GitHub account

### Step 2: Create Web Service
1. Dashboard → "New +" → "Web Service"
2. Connect repository: select `gradeforgePSA`
3. Configure:
   - **Name:** gradeforge
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python -m uvicorn gradeforge.app:app --host 0.0.0.0 --port 8000`
   - **Instance type:** Free tier (0.5 CPU, 512 MB RAM)

### Step 3: Set Environment Variables
In Render dashboard, add environment variables:

```
GRADEFORGE_PAYWALL=1
GRADEFORGE_BASE_URL=https://your-app.render.com  (replace with actual URL)
STRIPE_SECRET_KEY=sk_test_...  (get from Stripe dashboard)
STRIPE_WEBHOOK_SECRET=whsec_...  (get from Stripe dashboard)
GRADEFORGE_ADMIN_EMAILS=admin@gradeforge.local
GRADEFORGE_TRUST_PROXY=1
```

### Step 4: Configure Stripe Webhook (Optional for local testing)
1. Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://your-app.render.com/api/billing/webhook`
3. Events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy signing secret to `STRIPE_WEBHOOK_SECRET`

### Step 5: Deploy
1. Click "Deploy" button
2. Watch build logs
3. App will be live at `https://your-app.render.com`

---

## Auto-Deploy on Push
Render automatically redeploys when you push to `main` branch on GitHub.

```bash
# Just push and Render deploys automatically
git push origin main
```

---

## Testing Locally Before Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment (optional for local testing)
export GRADEFORGE_PAYWALL=1

# Run server
python -m uvicorn gradeforge.app:app --port 8000

# Visit http://localhost:8000
```

---

## Production Checklist

- [ ] Render account created
- [ ] Web service deployed
- [ ] Environment variables set
- [ ] Stripe test keys added
- [ ] Admin account registered (use setup code from logs)
- [ ] Webhook configured in Stripe
- [ ] Test login/signup flow
- [ ] Test payment with card: 4242 4242 4242 4242
- [ ] Upgrade to Stripe live keys when ready

---

## Troubleshooting

**App crashes on deploy?**
- Check build logs in Render dashboard
- Ensure `requirements.txt` is up to date
- Verify Python 3.13 is available

**Database not persisting?**
- Render's free tier uses ephemeral storage
- For production, add a Render PostgreSQL database
- Or use a persistent storage solution

**Webhook failing?**
- Verify URL is correct in Stripe dashboard
- Check webhook signing secret matches
- Monitor webhook deliveries in Stripe dashboard

---

## Upgrade to PostgreSQL (Recommended for Production)

1. Render Dashboard → "New +" → "PostgreSQL"
2. Create database
3. Copy connection string
4. Update code to use PostgreSQL instead of SQLite
5. Redeploy

---

## Support

- Render docs: https://render.com/docs
- GradeForge README: See README.md
- Stripe docs: https://stripe.com/docs
