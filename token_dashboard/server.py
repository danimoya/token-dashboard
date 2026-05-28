"""HTTP server: static frontend + JSON endpoints + SSE diff stream."""
from __future__ import annotations

import http.server
import json
import mimetypes
import queue
import threading
import time
import urllib.parse
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .db import (
    overview_totals, expensive_prompts, project_summary,
    tool_token_breakdown, recent_sessions, session_turns,
    daily_token_breakdown, model_breakdown, skill_breakdown,
    mcp_summary, mcp_per_tool, mcp_recent_calls, mcp_top_callers,
    project_mcp_share, session_mcp_share, prompt_mcp_share,
)

# TD_PRIMARY_STORE=helios routes the analytic read functions through the
# HeliosDB-backed implementation (db_helios.py). Default 'sqlite' keeps the
# existing behaviour (and is still the source of truth for ingestion until
# the dual-write cutover is complete).
import os as _os
if _os.environ.get("TD_PRIMARY_STORE", "sqlite").lower() == "helios":
    from .db_helios import (
        overview_totals, expensive_prompts, project_summary,
        tool_token_breakdown, recent_sessions, session_turns,
        daily_token_breakdown, model_breakdown, skill_breakdown,
        mcp_summary, mcp_per_tool, mcp_recent_calls, mcp_top_callers,
        project_mcp_share, session_mcp_share, prompt_mcp_share,
    )
from .pricing import load_pricing, cost_for, get_plan, set_plan
from .tips import all_tips, dismiss_tip
from .scanner import scan_dir
from .skills import cached_catalog
from . import mcp as mcp_client
from . import pulse as pulse_mod
from . import baseline as baseline_mod
from . import helios_writer as helios
from . import embedder
from . import clusters as clusters_mod
from . import alerts as alerts_mod
from . import audit as audit_mod
from . import branches as branches_mod
from . import docling_ingest
from . import code_resolution
from . import bulk_migrate
from . import auth as auth_mod


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
PRICING_JSON = Path(__file__).resolve().parent.parent / "pricing.json"

EVENTS: "queue.Queue[dict]" = queue.Queue()

MAX_POST_BYTES = 1_000_000  # 1 MB — we only accept tiny JSON bodies (plan, tip key)
MAX_LIMIT = 1000


def _send_json(handler, obj, status: int = 200) -> None:
    body = json.dumps(obj, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler, status: int, msg: str) -> None:
    _send_json(handler, {"error": msg}, status=status)


def _clamp_limit(raw, default: int) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(v, MAX_LIMIT))


def _savings_usd(tokens: int, pricing: dict) -> float:
    """Convert saved tool-output tokens to USD. Tool outputs flow back into
    the assistant's context as input tokens, so we price them at the input
    rate. We use the Sonnet 4.6 input rate as a defensible single number —
    Sonnet is the active default, and using Opus pricing would inflate the
    headline; using Haiku would understate it."""
    rate = (pricing.get("tier_fallback") or {}).get("sonnet", {}).get("input", 3.0)
    return round((tokens or 0) * rate / 1_000_000, 4)


def _serve_static(handler, rel: str) -> None:
    rel = rel.lstrip("/")
    p = (WEB_ROOT / rel).resolve()
    if not str(p).startswith(str(WEB_ROOT.resolve())) or not p.is_file():
        handler.send_response(404)
        handler.end_headers()
        return
    body = p.read_bytes()
    ctype, _ = mimetypes.guess_type(str(p))
    handler.send_response(200)
    handler.send_header("Content-Type", ctype or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_handler(db_path: str, projects_dir: str):
    pricing = load_pricing(PRICING_JSON)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_HEAD(self):
            return self.do_GET()

        # ---------------- auth helpers ----------------
        def _is_authenticated(self) -> bool:
            if not auth_mod.is_enabled():
                return True  # Open mode (no creds configured)
            cookies = auth_mod.parse_cookie_header(self.headers.get("Cookie"))
            return auth_mod.verify_cookie(cookies.get(auth_mod.COOKIE_NAME)) is not None

        def _redirect_to_login(self, next_path: str) -> None:
            self.send_response(302)
            self.send_header("Location", "/login?next=" + urllib.parse.quote(next_path))
            self.end_headers()

        def _serve_login(self, next_path: str = "/", error: str = "") -> None:
            body = auth_mod.render_login(error=error, next_path=next_path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            qs = parse_qs(url.query or "")
            path = url.path
            since = qs.get("since", [None])[0]
            until = qs.get("until", [None])[0]
            # ---- public routes (no auth required) ----
            if path == "/login":
                # Already logged in? Bounce to next.
                if self._is_authenticated():
                    self.send_response(302)
                    self.send_header("Location", qs.get("next", ["/"])[0] or "/")
                    self.end_headers()
                    return
                return self._serve_login(next_path=qs.get("next", ["/"])[0] or "/")
            if path == "/logout":
                self.send_response(302)
                self.send_header("Set-Cookie", auth_mod.cookie_clear_header())
                self.send_header("Location", "/login")
                self.end_headers()
                return
            # ---- gated routes ----
            if not self._is_authenticated():
                # API requests get 401 JSON; HTML requests get a redirect.
                if path.startswith("/api/"):
                    return _send_error(self, 401, "not authenticated")
                return self._redirect_to_login(self.path)
            if path in ("/", "/index.html"):
                return _serve_static(self, "index.html")
            if path.startswith("/web/"):
                return _serve_static(self, path[5:])
            if path == "/api/overview":
                totals = overview_totals(db_path, since, until)
                cost_usd = 0.0
                for m in model_breakdown(db_path, since, until):
                    c = cost_for(m["model"], m, pricing)
                    if c["usd"] is not None:
                        cost_usd += c["usd"]
                totals["cost_usd"] = round(cost_usd, 4)
                return _send_json(self, totals)
            if path == "/api/prompts":
                limit = _clamp_limit(qs.get("limit", ["50"])[0], 50)
                sort = qs.get("sort", ["tokens"])[0]
                rows = expensive_prompts(db_path, limit=limit, sort=sort)
                for r in rows:
                    c = cost_for(r["model"], {
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_read_tokens": r["cache_read_tokens"],
                        "cache_create_5m_tokens": 0, "cache_create_1h_tokens": 0,
                    }, pricing)
                    r["estimated_cost_usd"] = c["usd"]
                # Layer D — attach MCP stats per prompt
                share = prompt_mcp_share(db_path, [r["user_uuid"] for r in rows])
                for r in rows:
                    s = share.get(r["user_uuid"], {})
                    r["mcp_calls"]       = s.get("mcp_calls", 0)
                    r["mcp_tokens"]      = s.get("mcp_tokens", 0)
                    r["baseline_tokens"] = s.get("baseline_tokens", 0)
                    r["savings_tokens"]  = max(0, (r["baseline_tokens"] or 0) - (r["mcp_tokens"] or 0))
                return _send_json(self, rows)
            if path == "/api/projects":
                rows = project_summary(db_path, since, until)
                share = project_mcp_share(db_path, since, until)
                for r in rows:
                    s = share.get(r["project_slug"], {})
                    r["mcp_calls"]       = s.get("mcp_calls", 0)
                    r["mcp_tokens"]      = s.get("mcp_tokens", 0)
                    r["baseline_tokens"] = s.get("baseline_tokens", 0)
                    r["savings_tokens"]  = max(0, r["baseline_tokens"] - r["mcp_tokens"])
                return _send_json(self, rows)
            if path == "/api/tools":
                return _send_json(self, tool_token_breakdown(db_path, since, until))
            if path == "/api/sessions":
                rows = recent_sessions(
                    db_path, limit=_clamp_limit(qs.get("limit", ["20"])[0], 20),
                    since=since, until=until,
                )
                share = session_mcp_share(db_path, since, until)
                for r in rows:
                    s = share.get(r["session_id"], {})
                    r["mcp_calls"]       = s.get("mcp_calls", 0)
                    r["mcp_tokens"]      = s.get("mcp_tokens", 0)
                    r["baseline_tokens"] = s.get("baseline_tokens", 0)
                    r["savings_tokens"]  = max(0, r["baseline_tokens"] - r["mcp_tokens"])
                return _send_json(self, rows)
            if path == "/api/daily":
                return _send_json(self, daily_token_breakdown(db_path, since, until))
            if path == "/api/skills":
                rows = skill_breakdown(db_path, since, until)
                catalog = cached_catalog()
                for r in rows:
                    info = catalog.get(r["skill"])
                    r["tokens_per_call"] = info["tokens"] if info else None
                return _send_json(self, rows)
            if path == "/api/by-model":
                rows = model_breakdown(db_path, since, until)
                for r in rows:
                    c = cost_for(r["model"], r, pricing)
                    r["cost_usd"] = c["usd"]
                    r["cost_estimated"] = c["estimated"]
                return _send_json(self, rows)
            if path.startswith("/api/sessions/"):
                sid = path.rsplit("/", 1)[1]
                return _send_json(self, session_turns(db_path, sid))
            if path == "/api/tips":
                return _send_json(self, all_tips(db_path))
            if path == "/api/plan":
                return _send_json(self, {"plan": get_plan(db_path), "pricing": pricing})
            if path == "/api/scan":
                n = scan_dir(projects_dir, db_path)
                return _send_json(self, n)
            if path == "/api/scan/force":
                # Clear file-watermarks so every JSONL is re-read end-to-end.
                # Used after the tool_use_id schema migration to attribute
                # historical rows. The actual scan kicks off in the background
                # and respects WAL mode so reads stay fast during the rescan.
                from .db import connect as _conn
                with _conn(db_path) as c:
                    c.execute("DELETE FROM files")
                    c.commit()
                threading.Thread(target=scan_dir, args=(projects_dir, db_path), daemon=True).start()
                return _send_json(self, {"ok": True, "started": True,
                                         "note": "Full rescan running in background. Watch /api/overview for updates."})
            # ---------------- MCP / HeliosDB ----------------
            if path == "/api/mcp/summary":
                summary = mcp_summary(db_path, since, until)
                summary["per_tool"] = mcp_per_tool(db_path, since, until)
                summary["top_callers"] = mcp_top_callers(db_path, since, until,
                    limit=_clamp_limit(qs.get("top", ["10"])[0], 10))
                # Pricing context: convert savings tokens to USD using the
                # configured plan and a reasonable input-token price proxy.
                # We use the unweighted average of all model input prices
                # so the "savings" headline is a defensible single number.
                summary["estimated_savings_usd"] = _savings_usd(
                    summary.get("savings_tokens", 0), pricing)
                return _send_json(self, summary)
            if path == "/api/mcp/recent":
                lim = _clamp_limit(qs.get("limit", ["50"])[0], 50)
                return _send_json(self, mcp_recent_calls(db_path, limit=lim))
            if path == "/api/mcp/catalog":
                force = qs.get("force", ["0"])[0] == "1"
                return _send_json(self, mcp_client.get_catalog(force=force))
            if path == "/api/mcp/verify":
                return _send_json(self, mcp_client.conformance_check())
            if path == "/api/mcp/state":
                # Combined "is everything wired up?" check used by the UI to
                # decide what to render or grey out.
                catalog = mcp_client.get_catalog()
                pulse_configured = pulse_mod.is_configured()
                return _send_json(self, {
                    "mcp_configured":   mcp_client.is_configured(),
                    "mcp_endpoint":     mcp_client.http_url(),
                    "pulse_configured": pulse_configured,
                    "catalog_ok":       bool(catalog.get("tools")) and not catalog.get("error"),
                    "catalog_error":    catalog.get("error"),
                    "min_helios_version": mcp_client.MIN_HELIOS_VERSION,
                })
            if path == "/api/pulse":
                return _send_json(self, pulse_mod.pulse())
            if path == "/api/savings/backfill":
                return _send_json(self, baseline_mod.backfill(db_path))
            # ---------------- Path B — HeliosDB-backed features ----------------
            if path == "/api/helios/state":
                # Pulse already exposes server_version via SELECT version().
                # Reuse that here so the topbar attribution doesn't need a
                # second round-trip.
                pulse_data = pulse_mod.pulse() if pulse_mod.is_configured() else {}
                return _send_json(self, {
                    "configured": helios.is_configured(),
                    "stats": helios.stats() if helios.is_configured() else None,
                    "real_embeddings": embedder.is_real_embedding_available(),
                    "server_version": pulse_data.get("server_version"),
                })
            if path == "/api/search":
                # Vector search over prompts (#1)
                q = qs.get("q", [""])[0]
                lim = _clamp_limit(qs.get("limit", ["20"])[0], 20)
                proj = qs.get("project", [None])[0]
                if not q:
                    return _send_json(self, {"error": "missing q"}, 400)
                vec = embedder.embed(q)
                rows = helios.search_prompts(vec, limit=lim, project_slug=proj, since=since)
                return _send_json(self, {"query": q, "results": rows,
                                         "real_embeddings": embedder.is_real_embedding_available()})
            if path == "/api/explain":
                # RAG-style explain panel (#2)
                uuid = qs.get("uuid", [""])[0]
                if not uuid:
                    return _send_json(self, {"error": "missing uuid"}, 400)
                return _send_json(self, helios.explain_prompt(uuid,
                    neighbours=_clamp_limit(qs.get("neighbours",["5"])[0], 5)))
            if path == "/api/duplicate-retrievals":
                # Item #9
                threshold = float(qs.get("threshold", ["0.05"])[0])
                lim = _clamp_limit(qs.get("limit", ["50"])[0], 50)
                return _send_json(self, helios.find_duplicate_retrievals(threshold=threshold, limit=lim))
            if path == "/api/clusters":
                return _send_json(self, {
                    "clusters": clusters_mod.list_clusters(),
                    "configured": helios.is_configured(),
                })
            if path.startswith("/api/clusters/"):
                cid = int(path.rsplit("/", 1)[1])
                return _send_json(self, clusters_mod.cluster_members(cid,
                    limit=_clamp_limit(qs.get("limit",["50"])[0], 50)))
            if path == "/api/alerts":
                return _send_json(self, alerts_mod.list_alerts(
                    only_open=(qs.get("open",["1"])[0]=="1"),
                    limit=_clamp_limit(qs.get("limit",["50"])[0], 50)))
            if path == "/api/branches":
                return _send_json(self, {"snapshots": branches_mod.list_snapshots()})
            if path == "/api/branches/overview":
                br = qs.get("branch", [""])[0]
                if not br:
                    return _send_json(self, {"error": "missing branch"}, 400)
                return _send_json(self, branches_mod.overview_as_of(br))
            if path == "/api/audit":
                return _send_json(self, audit_mod.list_receipts(
                    limit=_clamp_limit(qs.get("limit",["100"])[0], 100)))
            if path == "/api/audit/verify":
                cid = int(qs.get("call_id",["0"])[0])
                return _send_json(self, audit_mod.verify(cid))
            if path == "/api/docs":
                return _send_json(self, docling_ingest.list_docs())
            if path == "/api/code-graph/status":
                return _send_json(self, code_resolution.code_graph_status())
            if path == "/api/helios/refresh-mvs":
                from . import helios_mv
                helios_mv.ensure_mvs()
                return _send_json(self, helios_mv.refresh_all())
            if path == "/api/helios/mv-status":
                from . import helios_mv
                return _send_json(self, {"staleness": helios_mv.staleness()})
            if path == "/api/helios/bulk-migrate":
                # Migration accepts higher caps than the analytic routes.
                try:
                    lim = max(1, int(qs.get("limit",["5000"])[0]))
                except ValueError:
                    lim = 5000
                return _send_json(self, bulk_migrate.migrate(db_path,
                    limit_messages=lim))
            if path == "/api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                while True:
                    try:
                        evt = EVENTS.get(timeout=15)
                        chunk = f"data: {json.dumps(evt, default=str)}\n\n".encode()
                    except queue.Empty:
                        chunk = b": ping\n\n"
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            url = urlparse(self.path)
            # Login uses form-encoded POSTs — handle before the JSON parser.
            if url.path == "/login":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return _send_error(self, 400, "invalid Content-Length")
                if length < 0 or length > 10000:
                    return _send_error(self, 413, "form too large")
                raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
                fields = urllib.parse.parse_qs(raw, keep_blank_values=True)
                username = (fields.get("username", [""])[0] or "").strip()
                password = fields.get("password", [""])[0] or ""
                next_path = fields.get("next", ["/"])[0] or "/"
                # Lock next_path to a same-origin path to block open-redirects.
                if not next_path.startswith("/") or next_path.startswith("//"):
                    next_path = "/"
                if auth_mod.check_credentials(username, password):
                    cookie = auth_mod.make_cookie(username)
                    self.send_response(302)
                    self.send_header("Set-Cookie", auth_mod.cookie_set_header(cookie))
                    self.send_header("Location", next_path)
                    self.end_headers()
                    return
                # Bad creds — re-render the form with an error.
                return self._serve_login(next_path=next_path,
                                         error="Wrong username or password.")
            # All other POSTs require auth.
            if not self._is_authenticated():
                return _send_error(self, 401, "not authenticated")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return _send_error(self, 400, "invalid Content-Length")
            if length < 0 or length > MAX_POST_BYTES:
                return _send_error(self, 413, f"body too large (max {MAX_POST_BYTES} bytes)")
            try:
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            except json.JSONDecodeError:
                return _send_error(self, 400, "invalid JSON")
            if not isinstance(body, dict):
                return _send_error(self, 400, "body must be a JSON object")
            if url.path == "/api/plan":
                set_plan(db_path, body.get("plan", "api"))
                return _send_json(self, {"ok": True})
            if url.path == "/api/tips/dismiss":
                dismiss_tip(db_path, body.get("key", ""))
                return _send_json(self, {"ok": True})
            if url.path == "/api/mcp/replay":
                # Body: {call_id?: int, recent?: int}
                replays = []
                if "call_id" in body:
                    r = mcp_client.replay_call(db_path, int(body["call_id"]),
                                               timeout=float(body.get("timeout", 30)))
                    replays = [r]
                else:
                    n = int(body.get("recent", 5))
                    replays = mcp_client.replay_recent(db_path, limit=max(1, min(n, 50)),
                                                       timeout=float(body.get("timeout", 30)))
                # Mirror replay receipts into the audit log (#10)
                if helios.is_configured():
                    for r in replays:
                        if not isinstance(r, dict) or "call_id" not in r:
                            continue
                        audit_mod.record(int(r["call_id"]), r.get("tool_name") or "?",
                                         r.get("response_tokens"),
                                         int(r.get("took_ms") or 0),
                                         bool(r.get("ok")), r.get("detail") or "")
                return _send_json(self, {"replays": replays})
            if url.path == "/api/clusters/build":
                k = int(body.get("k", 8))
                proj = body.get("project_slug")
                return _send_json(self, clusters_mod.cluster_prompts(k=k, project_slug=proj))
            if url.path == "/api/branches/snapshot":
                return _send_json(self, branches_mod.create_snapshot(body.get("name")))
            if url.path == "/api/alerts/ack":
                return _send_json(self, {"ok": alerts_mod.acknowledge(int(body.get("id", 0)))})
            if url.path == "/api/docs/ingest":
                slug = body.get("project_slug") or ""
                if not slug:
                    return _send_error(self, 400, "missing project_slug")
                return _send_json(self, docling_ingest.ingest_project(slug, body.get("cwd")))
            if url.path == "/api/code-graph/index":
                cwd = body.get("cwd") or ""
                if not cwd:
                    return _send_error(self, 400, "missing cwd")
                return _send_json(self, code_resolution.index_project(cwd))
            self.send_response(404)
            self.end_headers()

    return H


def _scan_loop(db_path: str, projects_dir: str, interval: float = 3600.0):
    while True:
        try:
            n = scan_dir(projects_dir, db_path)
            if n["messages"] > 0:
                EVENTS.put({"type": "scan", "n": n, "ts": time.time()})
        except Exception as e:
            EVENTS.put({"type": "error", "message": str(e)})
        time.sleep(interval)


def run(host: str, port: int, db_path: str, projects_dir: str):
    threading.Thread(target=_scan_loop, args=(db_path, projects_dir), daemon=True).start()
    H = build_handler(db_path, projects_dir)
    httpd = http.server.ThreadingHTTPServer((host, port), H)
    httpd.serve_forever()
