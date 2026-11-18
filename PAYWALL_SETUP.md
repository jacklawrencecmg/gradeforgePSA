# GradeForge as a paid site: setup

What it does (when `GRADEFORGE_PAYWALL=1`):
- A visitor gets **1 free grade** with no account. After that, the same IP address is blocked until someone creates an account and subscribes.
- Account = email + password. Subscribe through **Stripe Checkout** at **$6.99/month**; cancel or update the card in Stripe's billing portal.
- **Admin page** (the Admin tab, or go to /admin): daily unique visitors, cards graded per day, every card graded with who/when/source/country/verdict/best-possible grade/floor/real PSA result (if recorded), where grades came from, plus a CSV download. `jacklawrence713@gmail.com` is the default admin; change it with `GRADEFORGE_ADMIN_EMAILS`.
- Each person only sees their own analyses. The Dataset & Accuracy/training tabs and endpoints are admin-only (`GRADEFORGE_ADMIN_EMAILS`).
- With `GRADEFORGE_PAYWALL` unset (the default) the app behaves as the single-user local app it always was.

## Steps
1. **Stripe account**: create one, then copy the *test-mode* secret key (`sk_test_...`).
2. **Host it publicly over HTTPS.** The Windows double-click package is for your own computer; a public site needs a host (any service that runs a Python web app, with a persistent disk and a domain + HTTPS). Start command: `python -m uvicorn gradeforge.app:app --host 0.0.0.0 --port 8000`. Behind the host's proxy set `GRADEFORGE_TRUST_PROXY=1`.
3. Copy `gradeforge.env.example` to `gradeforge.env` (or set the same names as environment variables on the host) and fill in the keys and `GRADEFORGE_BASE_URL`.
4. **Webhook**: Stripe Dashboard > Developers > Webhooks > add endpoint `https://YOUR-SITE/api/billing/webhook`, events: `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`. Paste its signing secret into `STRIPE_WEBHOOK_SECRET`.
5. **Customer portal**: Stripe Dashboard > Settings > Billing > Customer portal > turn it on (the Manage billing button needs it).
6. Test with Stripe's test card `4242 4242 4242 4242` (any future date, any CVC). Create an account, subscribe, confirm grading works, cancel in the portal, confirm access ends at the period end. Then swap in live keys.

## Becoming the admin (one time)
Accounts are not email-verified, so nobody can register an admin email without a **setup code**. When the server starts it prints `Admin setup code: ...` in its log and saves it to `admin_setup_code.txt` in the data folder. On the site choose Create account with your admin email; a code box appears; enter the code. Do this before announcing the site.

## What the admin page measures
- Unique visitors: distinct (hashed) IP addresses per day on your browser's clock; repeat page loads within 30 minutes count once; bots are skipped. Many people behind one IP count as one, and one person on two networks counts as two.
- "Grade" is GradeForge's photo-based estimate (best possible grade, floor, verdict). It is not a PSA grade. The real PSA result shows only if the user recorded it.
- Source: free grade, subscriber, batch, or admin. Country appears only if your host adds a country header (Cloudflare, Vercel, CloudFront, App Engine) with `GRADEFORGE_TRUST_PROXY=1`; otherwise it reads "unknown". No GeoIP database is bundled, and raw IP addresses are not stored. Referrer is the site that sent the visitor.
- Tracking starts when the paywall mode is on; nothing is back-filled.
- Tell your visitors about this tracking in your privacy policy.

## How access is decided
- Subscribed = Stripe says `active` or `trialing` and the paid period (plus a 3-day safety margin) has not ended.
- When the customer returns from checkout the app asks Stripe directly, so access starts immediately; the webhook keeps renewals, cancellations and failed payments in sync. **Without the webhook, access would lapse after about a month.**

## Limits you should know about
- **Per-IP blocking is imperfect.** Anyone can get another free grade with a VPN or a different network. People sharing one IP (office, school, mobile carrier) share one free grade. A cookie adds a little friction but is easy to clear. Treat the free grade as a sample, not a security boundary. IPv6 addresses are counted per /64.
- A photo set where no card can be found does not use up the free grade (so people can retake), which also means failed attempts are not limited.
- Only hashes of IP addresses are stored. Passwords are scrypt-hashed. Session cookies are HttpOnly and SameSite=Lax; logins are rate-limited.
- **Not built:** password reset and email verification (they need an email-sending service), a terms of service / privacy policy / refund policy, and sales-tax handling (Stripe Tax can do this). You need these before taking real payments; get them reviewed by someone qualified.
- **Training data:** users can record "actual PSA results" on their own cards. These are unverified, so do not feed them into model training without checking them. The verified-dataset workflow (admin only) is the place for real outcomes.
- The Stripe integration was tested only against a simulated Stripe and locally signed webhooks, not against live or test-mode Stripe. Run step 6 before you launch.
