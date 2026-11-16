"""Stage 4: push the daily Practice Leader digest to where people read it.

  python src/notify.py            # sends to every channel that is configured, prints the digest otherwise

Channels (all optional, set as env vars / GitHub secrets):
  SLACK_WEBHOOK_URL   Slack incoming webhook
  TEAMS_WEBHOOK_URL   Microsoft Teams incoming webhook
  SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASS, MAIL_TO (comma separated), MAIL_FROM
  DASHBOARD_URL       link to the published dashboard (GitHub Pages)
"""
import os, smtplib, sys
from email.message import EmailMessage
import requests
from dotenv import load_dotenv
load_dotenv()

DIGEST = os.getenv("DIGEST_FILE", "reports/latest_digest.md")
LINK = os.getenv("DASHBOARD_URL", "")

def load():
    try:
        text = open(DIGEST, encoding="utf-8").read()
    except OSError:
        sys.exit(f"{DIGEST} not found: run src/report.py first.")
    return text + (f"\n\nFull dashboard: {LINK}" if LINK else "")

def slack(text):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return False
    body = text.replace("**", "*")[:2900]
    requests.post(url, json={"text": body}, timeout=20).raise_for_status()
    return True

def teams(text):
    url = os.getenv("TEAMS_WEBHOOK_URL")
    if not url:
        return False
    requests.post(url, json={"text": text[:6000]}, timeout=20).raise_for_status()
    return True

def email(text):
    host, to = os.getenv("SMTP_HOST"), os.getenv("MAIL_TO")
    if not (host and to):
        return False
    m = EmailMessage()
    m["Subject"] = text.splitlines()[0].lstrip("# ")
    m["From"] = os.getenv("MAIL_FROM", os.getenv("SMTP_USER", "ci-agent@localhost"))
    m["To"] = to
    m.set_content(text)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=30) as s:
        s.starttls()
        if os.getenv("SMTP_USER"):
            s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS", ""))
        s.send_message(m)
    return True

if __name__ == "__main__":
    text = load()
    sent = []
    for name, fn in (("slack", slack), ("teams", teams), ("email", email)):
        try:
            if fn(text):
                sent.append(name)
        except Exception as e:                      # one broken channel must not fail the pipeline
            print(f"[notify] {name} failed: {str(e)[:120]}")
    print("[notify] sent to:", ", ".join(sent) if sent else "no channel configured (digest printed below)\n\n" + text[:1500])
