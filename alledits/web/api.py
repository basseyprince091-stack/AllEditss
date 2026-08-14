"""Web API (Spec §26).

Deliberately built on the standard library. Every operation this exposes is
long-running and already has a job queue behind it, so a framework would buy
routing sugar and cost a dependency the deployment has to carry. Zero
dependencies means the product runs wherever Python does.

Two rules the spec sets for this layer, both load-bearing:

**"Do not spend excessive time on generic SaaS decoration before the engine
works."** So this is an API over the real engine and a single page that drives
it — not a dashboard shell with placeholder panels.

**No mock buttons.** Every endpoint performs the real operation. Where a
capability is genuinely absent (semantic search, subject tracking), the UI reads
that from /api/capabilities and says so, rather than showing a control that
quietly does nothing.

Video is served with HTTP range support, because a browser will not scrub — and
in Safari will not play at all — without it.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from ..core.jobs import BackgroundJobQueue, JobState
from ..pipeline.tasks import TASKS


STATIC = Path(__file__).parent / "static"


class AlleditsAPI:
    """Application state shared across requests."""

    def __init__(self, workdir, jobs_dir=None, styles_dir=None,
                 profile_path=None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.queue = BackgroundJobQueue(root=jobs_dir
                                        or (self.workdir / "jobs"), workers=2)
        self.styles_dir = Path(styles_dir) if styles_dir else \
            (Path.home() / ".alledits" / "styles")
        self.profile_path = Path(profile_path) if profile_path else \
            (Path.home() / ".alledits" / "profile.json")

    # ------------------------------------------------------------- endpoints
    def health(self):
        return {"ok": True, "workdir": str(self.workdir),
                "jobs": len(self.queue.list())}

    def capabilities(self):
        from ..intelligence.capabilities import default_registry
        reg = default_registry()
        return {"summary": reg.summary(),
                "capabilities": [s.to_dict() for s in reg.status()]}

    def delivery_profiles(self):
        from ..master import PROFILES
        return {"profiles": [{"name": n, "width": p.width, "height": p.height,
                              "fps": p.fps, "notes": p.notes}
                             for n, p in sorted(PROFILES.items())]}

    def styles(self):
        from ..reference.style import StyleLibrary
        lib = StyleLibrary(self.styles_dir)
        out = []
        for name in lib.list_names():
            try:
                g = lib.load(name)
                out.append({"name": name,
                            "cuts_per_second": round(g.pacing.cuts_per_second, 2),
                            "mean_shot": round(g.pacing.mean_shot, 2),
                            "rhythm": g.pacing.rhythm})
            except Exception as e:
                out.append({"name": name, "error": str(e)})
        return {"styles": out}

    def sequences(self):
        from ..shoot import SEQUENCES
        return {"sequences": [{"name": n, "description": d}
                              for n, (d, _) in sorted(SEQUENCES.items())]}

    def list_jobs(self, project_id=None, state=None):
        return {"jobs": [j.to_dict()
                         for j in self.queue.list(project_id=project_id,
                                                  state=state)]}

    def get_job(self, job_id):
        j = self.queue.get(job_id)
        return j.to_dict() if j else None

    def submit_job(self, body: dict):
        kind = body.get("kind")
        if kind not in TASKS:
            raise ValueError(f"kind must be one of {sorted(TASKS)}")
        params = dict(body.get("params") or {})
        params.setdefault("workdir", str(self.workdir))
        # Fail here, not inside the worker: a bad path should be a 400 the user
        # sees immediately, not a job that starts and dies.
        for key in ("clips_dir", "reference", "music", "src"):
            v = params.get(key)
            if v and not Path(v).exists():
                raise ValueError(f"{key} does not exist: {v}")
        if kind in ("edit", "autopilot") and not params.get("clips_dir"):
            raise ValueError("clips_dir is required")
        job = self.queue.submit(kind, TASKS[kind],
                                project_id=body.get("project_id"), **params)
        return job.to_dict()

    def cancel_job(self, job_id):
        ok = self.queue.cancel(job_id)
        return {"cancelled": ok,
                "detail": ("cancellation requested; it takes effect at the next "
                           "progress step" if ok else
                           "job already finished — nothing to cancel")}

    def project(self):
        p = self.workdir / "project.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def note(self, text: str, dry_run: bool = True):
        from ..core.project import Project, DirectiveKind
        from ..intelligence.director import parse_note, timeline_from_project
        path = self.workdir / "project.json"
        if not path.exists():
            raise FileNotFoundError("no project in this workdir yet — run an "
                                    "edit first")
        proj = Project.load(path)
        plan = parse_note(text, timeline_from_project(proj))
        out = plan.to_dict()
        out["applied"] = False
        if plan.understood and not dry_run:
            clips = timeline_from_project(proj).clips
            for ch in plan.changes:
                kw = {"note": text}
                if ch.slot_index is not None:
                    kw["slot_index"] = ch.slot_index
                if ch.shot_id:
                    kw["shot_id"] = ch.shot_id
                if ch.value is not None:
                    kw["value"] = ch.value
                d = proj.overrides.add(DirectiveKind(ch.kind), **kw)
                proj.overrides.anchor_to(d, clips)
            if plan.brief_delta:
                proj.brief = (proj.brief + ", " + plan.brief_delta).strip(", ")
            proj.record("note", text)
            proj.save(path)
            out["applied"] = True
        return out


# ------------------------------------------------------------------- routing
class Handler(BaseHTTPRequestHandler):
    api: AlleditsAPI = None
    server_version = "ALLEDITS"

    def log_message(self, fmt, *args):
        pass                       # quiet by default; the CLI prints what matters

    # -------------------------------------------------------------- helpers
    def _json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if getattr(self, "_head_only", False):
            return
        self.wfile.write(body)

    def _error(self, msg, status=400):
        self._json({"error": msg}, status=status)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_HEAD(self):
        """Needed for real reasons, not completeness: the page asks HEAD whether
        a render exists before showing a player, and video elements probe with
        HEAD before requesting ranges. Without it the player never appears."""
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    def _serve_file(self, path: Path, download=False):
        """Static and media, with range support so video can be scrubbed."""
        if not path.exists() or not path.is_file():
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{path.name}"')
        self.end_headers()
        if getattr(self, "_head_only", False):
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return          # the browser seeked away; not an error
                remaining -= len(chunk)

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        api = self.api
        try:
            if p in ("/", "/index.html"):
                return self._serve_file(STATIC / "index.html")
            if p == "/api/health":
                return self._json(api.health())
            if p == "/api/capabilities":
                return self._json(api.capabilities())
            if p == "/api/delivery-profiles":
                return self._json(api.delivery_profiles())
            if p == "/api/styles":
                return self._json(api.styles())
            if p == "/api/sequences":
                return self._json(api.sequences())
            if p == "/api/jobs":
                return self._json(api.list_jobs(
                    project_id=(q.get("project_id") or [None])[0],
                    state=(q.get("state") or [None])[0]))
            m = re.match(r"^/api/jobs/([^/]+)$", p)
            if m:
                j = api.get_job(m.group(1))
                return self._json(j) if j else self._error("no such job", 404)
            if p == "/api/project":
                pr = api.project()
                return self._json(pr) if pr else self._error("no project yet", 404)
            if p.startswith("/media/"):
                rel = unquote(p[len("/media/"):])
                target = (api.workdir / rel).resolve()
                # Path traversal guard: a media URL must not escape the workdir.
                if not str(target).startswith(str(api.workdir.resolve())):
                    return self._error("forbidden", 403)
                return self._serve_file(target,
                                        download=bool(q.get("download")))
            if p.startswith("/static/"):
                target = (STATIC / p[len("/static/"):]).resolve()
                if not str(target).startswith(str(STATIC.resolve())):
                    return self._error("forbidden", 403)
                return self._serve_file(target)
            return self._error("not found", 404)
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}", 500)

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        p = u.path
        api = self.api
        try:
            if p == "/api/jobs":
                return self._json(api.submit_job(self._body()), status=201)
            m = re.match(r"^/api/jobs/([^/]+)/cancel$", p)
            if m:
                return self._json(api.cancel_job(m.group(1)))
            if p == "/api/note":
                b = self._body()
                if not b.get("text"):
                    return self._error("text is required")
                return self._json(api.note(b["text"],
                                           dry_run=bool(b.get("dry_run", True))))
            return self._error("not found", 404)
        except (ValueError, FileNotFoundError) as e:
            return self._error(str(e), 400)
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}", 500)


def make_server(workdir, port: int = 8080, host: str = "127.0.0.1", **kw):
    handler = type("BoundHandler", (Handler,),
                   {"api": AlleditsAPI(workdir, **kw)})
    return ThreadingHTTPServer((host, port), handler)


def serve(workdir, port: int = 8080, host: str = "127.0.0.1", log=print, **kw):
    httpd = make_server(workdir, port=port, host=host, **kw)
    log(f"ALLEDITS on http://{host}:{port}  (workdir {workdir})")
    log("press ctrl-c to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("\nstopping")
    finally:
        httpd.server_close()
        httpd.RequestHandlerClass.api.queue.shutdown(wait=False)
