"""marlin login — Google sign-in via Supabase, for access + product updates.

Browser PKCE flow (the gh/stripe pattern): open Supabase's Google authorize URL
with a localhost redirect, catch the one-time code on a throwaway local server,
exchange it for a session, and stash {email, refresh_token} in
~/.marlin/auth.json. The email lands in Supabase `auth.users` automatically.
One-time; later runs reuse the stored session.

The Supabase *publishable* key below is public by design (client-side identity,
guarded by RLS) — safe to ship. Override with MARLIN_SUPABASE_URL / _KEY.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import CONFIG_DIR

SUPABASE_URL = os.environ.get("MARLIN_SUPABASE_URL", "https://iqjjodhgoohixngrlqaf.supabase.co").rstrip("/")
SUPABASE_KEY = os.environ.get("MARLIN_SUPABASE_KEY", "sb_publishable_BheDPS6J2QRE7bnyqzCyfA_KErwKoGp")
AUTH_FILE = CONFIG_DIR / "auth.json"


def current() -> dict | None:
    try:
        return json.loads(AUTH_FILE.read_text())
    except Exception:
        return None


def email() -> str | None:
    return (current() or {}).get("email")


def logout() -> bool:
    existed = AUTH_FILE.exists()
    AUTH_FILE.unlink(missing_ok=True)
    return existed


def google_enabled() -> bool | None:
    """True/False if the Supabase Google provider is on; None if unreachable.
    Lets callers skip a doomed browser open before the provider is configured."""
    try:
        req = urllib.request.Request(f"{SUPABASE_URL}/auth/v1/settings", headers={"apikey": SUPABASE_KEY})
        with urllib.request.urlopen(req, timeout=8) as r:
            return bool((json.loads(r.read()).get("external") or {}).get("google"))
    except Exception:
        return None


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


# Fixed localhost redirect port → one URL to allowlist in Supabase
# (Auth → URL Configuration → Redirect URLs: http://localhost:54123).
REDIRECT_PORT = int(os.environ.get("MARLIN_AUTH_PORT", "54123"))


_DONE_PAGE = (
    b"<!doctype html><meta charset=utf-8><title>marlin</title>"
    b"<body style=\"font-family:system-ui;text-align:center;padding-top:90px;"
    b"color:#2B2220;background:#F4ECD8\">"
    b"<div style=\"font-size:24px;color:#E76F57;font-weight:600\">marlin</div>"
    b"<p>Signed in. Close this tab and return to your terminal.</p></body>"
)


def login(*, log=lambda m: None, timeout: float = 300.0) -> dict:
    """Run the browser sign-in; return {email, user_id, refresh_token}. Raises
    RuntimeError with a clear message on any failure."""
    if google_enabled() is False:
        raise RuntimeError("Google sign-in isn't enabled on the server yet")

    verifier, challenge = _pkce()
    port = REDIRECT_PORT
    redirect = f"http://localhost:{port}"
    authorize = (
        f"{SUPABASE_URL}/auth/v1/authorize?provider=google"
        f"&redirect_to={urllib.parse.quote(redirect, safe='')}"
        f"&code_challenge={challenge}&code_challenge_method=s256"
    )

    caught: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            caught["code"] = (q.get("code") or [None])[0]
            caught["error"] = (q.get("error_description") or q.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_DONE_PAGE)

        def log_message(self, *a):
            pass

    try:
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        raise RuntimeError(f"local port {port} is busy — free it (or set MARLIN_AUTH_PORT) and retry")
    server.timeout = 1
    import webbrowser

    log("opening your browser to sign in…")
    if not webbrowser.open(authorize):
        log(f"open this URL to sign in:\n{authorize}")

    deadline = time.time() + timeout
    while "code" not in caught and time.time() < deadline:
        server.handle_request()
    server.server_close()

    if caught.get("error"):
        raise RuntimeError(f"sign-in failed: {caught['error']}")
    if not caught.get("code"):
        raise RuntimeError("sign-in timed out")

    body = json.dumps({"auth_code": caught["code"], "code_verifier": verifier}).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=pkce",
        data=body, method="POST",
        headers={"Content-Type": "application/json", "apikey": SUPABASE_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            session = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"token exchange failed ({e.code}): {e.read().decode(errors='ignore')[:200]}")

    user = session.get("user") or {}
    out = {"email": user.get("email"), "user_id": user.get("id"),
           "refresh_token": session.get("refresh_token")}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(out, indent=2) + "\n")
    AUTH_FILE.chmod(0o600)
    return out
