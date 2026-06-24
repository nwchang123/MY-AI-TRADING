"""Local read-only HTTP server for the trading dashboard.

Binds to 127.0.0.1 only.  Serves static files and JSON APIs from the
``web_dashboard.state`` module.  Phase 5 adds controlled write endpoints
with confirmation.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from trading_agent.web_dashboard import state

_STATIC_DIR = Path(__file__).parent / "static"


class DashboardHandler(BaseHTTPRequestHandler):
    """Handle GET/POST for dashboard routes."""

    root_dir: Path  # set by the factory
    mode: str = "paper"  # set by the factory
    web_token: str = ""  # set by the factory; blank keeps local legacy behavior

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress default access logging to keep stdout clean.
        pass

    def _authorized(self, qs: dict[str, list[str]] | None = None) -> bool:
        if not self.web_token:
            return True
        supplied = self.headers.get("X-Trading-Agent-Web-Token", "")
        if not supplied and qs is not None:
            supplied = (qs.get("token") or [""])[0]
        return hmac.compare_digest(supplied, self.web_token)

    def _send_unauthorized(self) -> None:
        self._send_json({"ok": False, "error": "unauthorized"}, 401)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0 or length > 65536:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            return data
        except (json.JSONDecodeError, ValueError):
            return {}

    def _serve_static(self, rel: str) -> None:
        if not rel:
            rel = "index.html"
        # Prevent path traversal: resolve and verify the file is under _STATIC_DIR
        file_path = (_STATIC_DIR / rel).resolve()
        if not str(file_path).startswith(str(_STATIC_DIR.resolve())):
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_health())
        elif path == "/api/status":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_status(self.root_dir))
        elif path == "/api/positions":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_positions(self.root_dir, self.mode))
        elif path == "/api/portfolio-risk":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_portfolio_risk(self.root_dir, self.mode))
        elif path == "/api/report/today":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_report_today(self.root_dir))
        elif path == "/api/funnel/today":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_funnel_today(self.root_dir))
        elif path == "/api/calibration/today":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_calibration_today(self.root_dir))
        elif path == "/api/audit/recent":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            limit = int(qs.get("limit", ["80"])[0])
            limit = max(1, min(limit, 500))
            self._send_json(state.get_audit_recent(self.root_dir, limit))
        elif path == "/api/audit/24h":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            limit = int(qs.get("limit", ["200"])[0])
            limit = max(1, min(limit, 500))
            self._send_json(state.get_audit_24h(self.root_dir, limit))
        elif path == "/api/iv":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_iv(self.root_dir))
        elif path == "/api/config/mandate":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._send_json(state.get_config_mandate(self.root_dir, self.mode))
        elif path == "/":
            if not self._authorized(qs):
                self._send_unauthorized()
                return
            self._serve_static("index.html")
        elif path.startswith("/"):
            self._serve_static(path[1:])
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_body()
        if not self._authorized():
            self._send_unauthorized()
            return

        if path == "/api/halt":
            confirm = body.get("confirm", False)
            if not confirm:
                self._send_json({"ok": False, "error": "confirmation required"}, 400)
                return
            reason = body.get("reason", "web dashboard halt")
            self._send_json(state.execute_halt(self.root_dir, reason))
        elif path == "/api/resume":
            confirm = body.get("confirm", False)
            if not confirm:
                self._send_json({"ok": False, "error": "confirmation required"}, 400)
                return
            self._send_json(state.execute_resume(self.root_dir))
        elif path == "/api/run-cycle":
            confirm = body.get("confirm", False)
            if not confirm:
                self._send_json({"ok": False, "error": "confirmation required"}, 400)
                return
            # Run-cycle is a blocking operation — return immediately with
            # a pending status.  The cycle runs in a background thread.
            import threading

            def _run() -> None:
                try:
                    from trading_agent.cli import Settings, _build_cycle, _cycle_lock_path
                    from trading_agent.execution.lock import single_instance_lock
                    from trading_agent.brokers.moomoo import assert_opend_reachable

                    settings = Settings.from_env(self.root_dir)
                    tickers = body.get("tickers", [])
                    if not tickers:
                        return

                    assert_opend_reachable(settings.moomoo_host, settings.moomoo_port)
                    with single_instance_lock(_cycle_lock_path(settings)):
                        result = _build_cycle(settings, trd_env="SIMULATE").run_once(tickers)
                    from trading_agent.storage.audit import AuditWriter
                    AuditWriter(self.root_dir / "runtime" / "audit.jsonl").append(
                        "web_cycle_completed",
                        {"entries": len(result.entries), "exits": len(result.exits)},
                    )
                except Exception as exc:
                    from trading_agent.storage.audit import AuditWriter
                    AuditWriter(self.root_dir / "runtime" / "audit.jsonl").append(
                        "web_cycle_failed",
                        {"error": str(exc)},
                    )

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._send_json({"ok": True, "data": {"status": "cycle_started"}})
        else:
            self.send_error(404)


def make_handler(root_dir: Path, mode: str = "paper", web_token: str = ""):
    """Create a handler class bound to the given root_dir and mode."""

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.root_dir = root_dir
    BoundHandler.mode = mode
    BoundHandler.web_token = web_token
    return BoundHandler


def serve(
    root_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: str = "paper",
    web_token: str = "",
) -> None:
    """Start the dashboard server (blocking)."""
    handler_cls = make_handler(root_dir, mode, web_token)
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"Dashboard serving on http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()
