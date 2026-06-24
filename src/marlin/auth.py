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
# SAFETY: this MUST stay the `sb_publishable_` (anon) key — public by design,
# guarded by RLS, safe to ship. NEVER put a `sb_secret_`/service_role key here.
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

# Branded onboarding form, served by the CLI's own localhost server (no hosting).
# Two required fields → POST /profile → redirect to Google. __AUTHORIZE__ is the
# Supabase Google authorize URL, injected per-run.
_FORM_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>marlin — sign in</title><style>
:root{--coral:#E76F57;--ink:#2B2220;--cream:#F4ECD8}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--cream);color:var(--ink);
 margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.card{width:min(420px,90vw);padding:40px 36px}
.mark{font-size:28px;font-weight:700;color:var(--coral);text-align:center;margin-bottom:6px}
.sub{text-align:center;color:#6b5d56;font-size:14px;margin-bottom:24px}
label{display:block;font-size:13px;font-weight:600;margin:18px 0 6px}
input{width:100%;padding:11px 13px;border:1px solid #d8ccb4;border-radius:8px;
 background:#fff;font-size:15px;color:var(--ink)}
input:focus{outline:none;border-color:var(--coral)}
button{width:100%;margin-top:26px;padding:12px;border:0;border-radius:8px;background:var(--coral);
 color:#fff;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{background:#e2d6c2;color:#ab9d92;cursor:not-allowed}
.foot{text-align:center;color:#9a8c84;font-size:12px;margin-top:18px}
</style></head><body><div class="card">
<div class="mark">marlin</div>
<div class="sub">Two quick questions, then sign in with Google.</div>
<form id="f">
<label for="aff">Affiliation or company</label>
<input id="aff" autocomplete="organization" placeholder="e.g. Stanford, Acme Inc, independent" required>
<label for="use">What do you want to use Marlin for?</label>
<input id="use" placeholder="e.g. labelling sports clips, video search" required>
<button id="go" type="submit" disabled>Continue with Google</button>
</form>
<div class="foot">We only use this to understand who's building with Marlin.</div>
</div><script>
var aff=document.getElementById('aff'),use=document.getElementById('use'),
go=document.getElementById('go'),f=document.getElementById('f');
function chk(){go.disabled=!(aff.value.trim()&&use.value.trim());}
aff.oninput=chk;use.oninput=chk;
f.onsubmit=function(e){e.preventDefault();
if(!(aff.value.trim()&&use.value.trim()))return;
go.disabled=true;go.textContent='Redirecting to Google\\u2026';
fetch('/profile',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({affiliation:aff.value.trim(),use_case:use.value.trim()})})
.then(function(){window.location='__AUTHORIZE__';})
.catch(function(){window.location='__AUTHORIZE__';});};
</script></body></html>"""


def _form_html(authorize: str) -> bytes:
    return _FORM_TEMPLATE.replace("__AUTHORIZE__", authorize).encode("utf-8")


def _update_profile(access_token: str, data: dict) -> None:
    """PUT the two onboarding answers into Supabase user_metadata (auth.users).
    Access token is used in-memory only — never written to disk."""
    body = json.dumps({"data": data}).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/user",
        data=body, method="PUT",
        headers={"Content-Type": "application/json", "apikey": SUPABASE_KEY,
                 "Authorization": f"Bearer {access_token}"},
    )
    urllib.request.urlopen(req, timeout=15).close()


_MAX_PROFILE_BYTES = 8192


def _clean_profile(raw: bytes) -> dict:
    """Whitelist the onboarding answers before they're written to user_metadata:
    only affiliation + use_case, strings, length-capped. Anything else is dropped,
    so a stray cross-origin POST can't stuff arbitrary keys into the user's record."""
    try:
        data = json.loads(raw or b"{}")
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k in ("affiliation", "use_case"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()[:500]
    return out


def login(*, log=lambda m: None, timeout: float = 300.0) -> dict:
    """Run the browser sign-in (2-question form → Google); return
    {email, user_id, refresh_token}. The answers land in Supabase user_metadata.
    Raises RuntimeError with a clear message on any failure."""
    if google_enabled() is False:
        raise RuntimeError("Google sign-in isn't enabled on the server yet")

    verifier, challenge = _pkce()
    port = REDIRECT_PORT
    redirect = f"http://localhost:{port}"
    authorize = (
        f"{SUPABASE_URL}/auth/v1/authorize?provider=google"
        f"&redirect_to={urllib.parse.quote(redirect, safe='')}"
        f"&code_challenge={challenge}&code_challenge_method=S256"
    )
    form_page = _form_html(authorize)
    # Only our own loopback form may POST answers; reject cross-origin writes.
    allowed_origins = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}

    caught: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _reply(self, body: bytes, code: int = 200, ctype: str = "text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(parsed.query)
            # Google redirect back → callback (carries code/error)
            if "code" in q or "error" in q or "error_description" in q:
                caught["code"] = (q.get("code") or [None])[0]
                caught["error"] = (q.get("error_description") or q.get("error") or [None])[0]
                self._reply(_DONE_PAGE)
                return
            # First load → serve the onboarding form
            if parsed.path in ("", "/"):
                self._reply(form_page)
                return
            self._reply(b"", 204)  # favicon, etc.

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/profile":
                self._reply(b"", 404)
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                self._reply(b"", 403)  # cross-origin write attempt
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > _MAX_PROFILE_BYTES:
                self._reply(b"", 413)
                return
            raw = self.rfile.read(n) if n else b"{}"
            caught["profile"] = _clean_profile(raw)
            self._reply(b"", 204)

        def log_message(self, *a):
            pass

    try:
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        raise RuntimeError(f"local port {port} is busy — free it (or set MARLIN_AUTH_PORT) and retry")
    server.timeout = 1
    import webbrowser

    log("opening your browser — 2 quick questions, then Google sign-in…")
    if os.environ.get("MARLIN_NO_BROWSER") or not webbrowser.open(redirect):
        log(f"open this URL to continue:\n{redirect}")

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

    # Best-effort: write the two answers to user_metadata. Never fail sign-in over it.
    profile = caught.get("profile") or {}
    access_token = session.get("access_token")
    if profile and access_token:
        try:
            _update_profile(access_token, profile)
        except Exception:
            pass

    user = session.get("user") or {}
    out = {"email": user.get("email"), "user_id": user.get("id"),
           "refresh_token": session.get("refresh_token")}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic 0600 create — auth.json holds the refresh_token, so never leave a
    # world/group-readable window between write and chmod. fchmod covers the
    # case where the file already existed with looser perms.
    fd = os.open(AUTH_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(out, indent=2) + "\n")
    return out
