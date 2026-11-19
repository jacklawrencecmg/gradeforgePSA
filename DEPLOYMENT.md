# GradeForge Deployment Guide

## Quick Start: Deploy to Fly.io (Free Tier) ⭐

### Step 1: Create Fly.io Account
1. Go to https://fly.io
2. Sign up with GitHub (easiest)
3. Verify email

### Step 2: Install Fly CLI
**On Mac/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

**On Windows (PowerShell):**
```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

Or download: https://github.com/superfly/flyctl/releases

### Step 3: Login to Fly
```bash
flyctl auth login
```
(Opens browser to authenticate with GitHub)

### Step 4: Clone Repository
```bash
git clone https://github.com/jacklawrencecmg/gradeforgePSA.git
cd gradeforgePSA
```

### Step 5: Deploy to Fly.io
```bash
flyctl deploy
```

That's it! 🚀 Fly.io will:
- Build Docker image
- Deploy to their servers
- Provide live URL (e.g., `https://gradeforgepsa.fly.dev`)

### Step 6: Set Environment Variables
```bash
flyctl secrets set GRADEFORGE_BASE_URL=https://gradeforgepsa.fly.dev
flyctl secrets set STRIPE_SECRET_KEY=sk_test_...
flyctl secrets set STRIPE_WEBHOOK_SECRET=whsec_...
flyctl secrets set GRADEFORGE_ADMIN_EMAILS=admin@gradeforge.local
```

Then redeploy:
```bash
flyctl deploy
```

### Step 7: Configure Stripe Webhook
1. Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://gradeforgepsa.fly.dev/api/billing/webhook`
3. Events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy signing secret to `STRIPE_WEBHOOK_SECRET`

### Step 8: Register Admin Account
1. Visit your app: `https://gradeforgepsa.fly.dev`
2. Check logs for admin setup code:
   ```bash
   flyctl logs
   ```
3. Register with the code

---

## Auto-Deploy on Git Push

Fly.io automatically redeploys when you push to `main` branch:

```bash
git push origin main
```

---

## Testing Locally Before Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn gradeforge.app:app --port 8000

# Visit http://localhost:8000
```

---

## Production Checklist

- [ ] Fly.io account created
- [ ] CLI installed (`flyctl` command available)
- [ ] Repository cloned
- [ ] Deployed with `flyctl deploy`
- [ ] Environment variables set with `flyctl secrets set`
- [ ] Live URL working
- [ ] Admin account registered
- [ ] Stripe test keys added
- [ ] Webhook configured in Stripe
- [ ] Test login/signup flow
- [ ] Test payment with card: 4242 4242 4242 4242
- [ ] Upgrade to Stripe live keys when ready

---

## Troubleshooting

**Deployment fails?**
```bash
# Check logs
flyctl logs

# Restart app
flyctl restart
```

**Can't connect after deploy?**
- Wait 30-60 seconds for DNS to propagate
- Check live URL: `flyctl info`
- Review logs: `flyctl logs`

**Database issues?**
- Fly.io uses ephemeral storage (data lost on restart)
- For production, attach PostgreSQL: `flyctl postgres create`
- Or use Fly.io Volumes for persistent storage

**Need more resources?**
```bash
# Scale up
flyctl scale count 2

# More RAM
flyctl scale memory 1024
```

---

## Useful Commands

```bash
# View app status
flyctl status

# View logs
flyctl logs

# SSH into app
flyctl ssh console

# View environment variables
flyctl secrets list

# Restart app
flyctl restart

# Deploy latest code
flyctl deploy

# View app info
flyctl info
```

---

## Upgrade from Free to Paid

Fly.io free tier includes:
- 3 shared CPU instances
- 512MB RAM per instance
- 160GB bandwidth/month

If you outgrow free tier, add credit card and scale up:
```bash
flyctl scale count 3
flyctl scale memory 1024
```

---

## Support

- Fly.io docs: https://fly.io/docs/
- GradeForge README: See README.md
- Stripe docs: https://stripe.com/docs
- GitHub repo: https://github.com/jacklawrencecmg/gradeforgePSA
