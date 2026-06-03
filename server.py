"""
PixelForge AI — Local REST API Server
======================================
SETUP:
    pip install flask flask-cors pillow

RUN:
    python server.py

NOTES:
    - Downloads Real-ESRGAN engine on first run (~50 MB)
    - Works on Windows, macOS, and Linux
    - Serves the frontend at http://localhost:5000/ui  (no CORS issues)
"""

import os
import sys
import platform
import subprocess
import zipfile
import tarfile
import urllib.request
import threading
import uuid
import time
from pathlib import Path
from functools import wraps

from flask import (
    Flask, request, jsonify, send_file, make_response,
    send_from_directory, after_this_request,
)
from flask_cors import CORS
from PIL import Image

# ═══════════════════════════════════════════════════════════════
#  USER SETTINGS
# ═══════════════════════════════════════════════════════════════
API_KEY           = "pixelforge-secret-2024"
HOST              = "0.0.0.0"
PORT              = 5000
MAX_KB_HARD_LIMIT = 10000
PROCESS_TIMEOUT   = 3600   # 60 minutes
# ═══════════════════════════════════════════════════════════════

ALLOWED_MODELS = {
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "realesr-animevideov3",
}

app = Flask(__name__)

# ── CORS — allow all origins (frontend served separately) ────────
CORS(app,
     origins="*",
     allow_headers=["Content-Type", "X-API-Key"],
     methods=["GET", "POST", "OPTIONS"],
     expose_headers=["X-Original-Size", "X-Output-Size", "X-Output-Dims",
                     "Content-Disposition"])

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Original-Size, X-Output-Size, X-Output-Dims, Content-Disposition"
    )
    return response

BASE_DIR    = Path(__file__).parent
ENGINE_DIR  = BASE_DIR / "AI_engine"
TEMP_DIR    = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VERSION   = "1.1.0"

# ── Engine state ─────────────────────────────────────────────────
ENGINE_PATH  = None
ENGINE_READY = threading.Event()   # set when engine is confirmed available


# ─── PLATFORM DETECTION ──────────────────────────────────────────
def _detect_platform():
    """Return (os_name, arch) tuple for engine selection."""
    os_name = platform.system().lower()   # 'windows', 'darwin', 'linux'
    machine = platform.machine().lower()  # 'amd64', 'x86_64', 'arm64', etc.
    arch = "arm" if "arm" in machine or "aarch" in machine else "x86"
    return os_name, arch


# ─── ENGINE DOWNLOAD TABLE ───────────────────────────────────────
_ENGINE_RELEASES = {
    # (os, arch): (exe_name_in_zip, zip_url)
    ("windows", "x86"): (
        "realesrgan-ncnn-vulkan.exe",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesrgan-ncnn-vulkan-20220424-windows.zip",
    ),
    ("linux", "x86"): (
        "realesrgan-ncnn-vulkan",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
    ),
    ("darwin", "x86"): (
        "realesrgan-ncnn-vulkan",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesrgan-ncnn-vulkan-20220424-macos.zip",
    ),
    ("darwin", "arm"): (
        "realesrgan-ncnn-vulkan",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesrgan-ncnn-vulkan-20220424-macos.zip",
    ),
}


def get_engine_path():
    """Download (if needed) and return the engine executable path, or None on failure."""
    os_name, arch = _detect_platform()
    key = (os_name, arch)

    if key not in _ENGINE_RELEASES:
        print(f"[PixelForge] ⚠️  Unsupported platform: {os_name}/{arch}")
        return None

    exe_name, url = _ENGINE_RELEASES[key]
    exe_path = ENGINE_DIR / exe_name

    if exe_path.exists():
        # Ensure it's executable on Unix
        if os_name != "windows":
            exe_path.chmod(exe_path.stat().st_mode | 0o111)
        print(f"[PixelForge] ✅ AI Engine found: {exe_path}")
        return str(exe_path)

    print(f"\n[PixelForge] 📥 Downloading AI Engine for {os_name}/{arch} (~50 MB)…")
    ENGINE_DIR.mkdir(exist_ok=True)
    zip_path = ENGINE_DIR / "realesrgan_download.zip"

    cleanup_scheduled = False
    try:
        start = time.time()

        def progress(count, block_size, total):
            pct = min(100, count * block_size * 100 // max(total, 1))
            if count % 50 == 0:
                print(f"\r[PixelForge]   {pct}% …", end="", flush=True)

        urllib.request.urlretrieve(url, str(zip_path), reporthook=progress)
        print(f"\n[PixelForge] Downloaded in {time.time()-start:.1f}s")

        print("[PixelForge] 📦 Extracting…")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(str(ENGINE_DIR))
        zip_path.unlink(missing_ok=True)

        if not exe_path.exists():
            # Some archives nest inside a subfolder — search recursively
            found = list(ENGINE_DIR.rglob(exe_name))
            if found:
                found[0].rename(exe_path)
            else:
                print(f"[PixelForge] ❌ Executable '{exe_name}' not found after extraction")
                return None

        if os_name != "windows":
            exe_path.chmod(exe_path.stat().st_mode | 0o111)

        print("[PixelForge] ✅ AI Engine ready!")
        return str(exe_path)

    except Exception as e:
        print(f"\n[PixelForge] ❌ Download failed: {e}")
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


# ─── AUTH ────────────────────────────────────────────────────────
def require_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return make_response("", 200)
        if request.headers.get("X-API-Key", "") != API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return wrapper


# ─── CLEANUP HELPER ──────────────────────────────────────────────
def _cleanup(*paths):
    for p in paths:
        try:
            if p and Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass


# ─── ROUTES ──────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":    "PixelForge AI API",
        "version": VERSION,
        "status":  "running",
        "engine":  ENGINE_PATH is not None,
        "ui":      f"http://localhost:{PORT}/ui",
    })


@app.route("/ui", methods=["GET"])
def serve_ui():
    """Serve the frontend HTML from the same directory — no CORS issues."""
    # Look for the HTML file next to server.py
    candidates = [
        BASE_DIR / "pixelforge_frontend.html",
        BASE_DIR / "frountend_fixed.html",
        BASE_DIR / "frontend.html",
        BASE_DIR / "new dfile.html",
    ]
    for p in candidates:
        if p.exists():
            return send_file(str(p))
    return (
        "<h2>Frontend file not found.</h2>"
        "<p>Place <code>pixelforge_frontend.html</code> next to <code>server.py</code>.</p>",
        404,
    )


@app.route("/ping", methods=["GET", "OPTIONS"])
@require_key
def ping():
    """Health check endpoint."""
    return jsonify({
        "status":  "ok",
        "version": f"PixelForge API v{VERSION}",
        "engine":  ENGINE_PATH is not None,
        "message": "Ready to enhance!" if ENGINE_PATH else "Engine loading…",
        "timeout": f"{PROCESS_TIMEOUT}s",
    })


@app.route("/enhance", methods=["POST", "OPTIONS"])
@require_key
def enhance():
    """Main image enhancement endpoint."""
    global ENGINE_PATH

    # ── Input validation ──────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f   = request.files["image"]
    ext = Path(f.filename).suffix.lower()
    if ext not in SUPPORTED:
        return jsonify({
            "error": f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED))}"
        }), 400

    try:
        scale = int(request.form.get("scale", 4))
        model = request.form.get("model", "realesrgan-x4plus")
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    max_kb_raw = request.form.get("max_kb")
    max_kb = None
    if max_kb_raw not in (None, ""):
        try:
            max_kb = min(int(max_kb_raw), MAX_KB_HARD_LIMIT)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid max_kb: {e}"}), 400

    if scale not in (2, 4):
        return jsonify({"error": "Scale must be 2 or 4"}), 400

    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"Unknown model '{model}'"}), 400

    # ── Engine check ─────────────────────────────────────────────
    if ENGINE_PATH is None:
        ENGINE_PATH = get_engine_path()
    if not ENGINE_PATH:
        return jsonify({
            "error": "AI engine not available. Check server logs."
        }), 503

    # ── Job setup ────────────────────────────────────────────────
    job_id  = str(uuid.uuid4())[:8]
    in_path  = TEMP_DIR / f"{job_id}_in{ext}"
    out_png  = TEMP_DIR / f"{job_id}_out.png"
    out_jpg  = TEMP_DIR / f"{job_id}_out.jpg"
    cleanup_scheduled = False

    try:
        f.save(str(in_path))
        orig_size_kb = in_path.stat().st_size // 1024

        print(f"\n[{job_id}] 🧠 Job started: {f.filename}")
        print(f"[{job_id}]   Input  : {orig_size_kb} KB")
        print(f"[{job_id}]   Scale  : {scale}×  Model: {model}")

        # Verify it's a valid image
        try:
            with Image.open(str(in_path)) as img:
                orig_w, orig_h = img.size
                img.load()   # force decode
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {e}"}), 400

        print(f"[{job_id}]   Dims   : {orig_w}×{orig_h}")

        # ── Run engine ───────────────────────────────────────────
        models_dir = ENGINE_DIR / "models"
        cmd = [ENGINE_PATH, "-i", str(in_path), "-o", str(out_png),
               "-s", str(scale), "-n", model]
        if models_dir.is_dir():
            cmd += ["-m", str(models_dir)]

        print(f"[{job_id}]   CMD    : {' '.join(cmd)}", flush=True)
        print(f"[{job_id}] ⏳ Running AI (timeout {PROCESS_TIMEOUT}s)…", flush=True)
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=PROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"[{job_id}] ❌ Timeout after {elapsed:.0f}s")
            return jsonify({
                "error": f"Processing timed out after {PROCESS_TIMEOUT}s. Try a smaller image."
            }), 504

        elapsed = time.time() - start_time

        if result.stderr:
            print(f"[{job_id}] stderr: {result.stderr[:800]}", flush=True)
        if result.stdout:
            print(f"[{job_id}] stdout: {result.stdout[:400]}", flush=True)

        if not out_png.exists():
            err = (result.stderr or "No output produced")[:600]
            print(f"[{job_id}] ❌ Engine failed: {err}")
            return jsonify({"error": f"AI enhancement failed: {err}"}), 500

        # ── Post-process & compress ──────────────────────────────
        try:
            ai_img = Image.open(str(out_png))
            out_w, out_h = ai_img.size
            # JPEG cannot store alpha — flatten transparent PNGs (e.g. logos)
            if ai_img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", ai_img.size, (255, 255, 255))
                bg.paste(ai_img, mask=ai_img.split()[-1])
                ai_img = bg
            elif ai_img.mode != "RGB":
                ai_img = ai_img.convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Cannot read enhanced image: {e}"}), 500

        def save_jpg(quality):
            ai_img.save(str(out_jpg), "JPEG", quality=quality, optimize=True)
            return out_jpg.stat().st_size

        quality = 95
        size = save_jpg(quality)

        if max_kb is not None:
            max_bytes = max_kb * 1024
            if size > max_bytes:
                print(f"[{job_id}] ⚙️  Optimising size ({size//1024}KB → target {max_kb}KB)…")
                lo, hi, best = 50, 94, 50
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if save_jpg(mid) <= max_bytes:
                        best, lo = mid, mid + 1
                    else:
                        hi = mid - 1
                save_jpg(best)
                quality = best

        ai_img.close()
        out_size_kb = out_jpg.stat().st_size // 1024
        out_name    = "AI_" + Path(f.filename).stem + ".jpg"

        print(f"[{job_id}] ✅ Done  : {out_w}×{out_h}  {out_size_kb}KB  q={quality}%  t={elapsed:.1f}s")

        resp = make_response(send_file(
            str(out_jpg),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=out_name,
        ))
        resp.headers["X-Original-Size"]    = str(orig_size_kb)
        resp.headers["X-Output-Size"]      = str(out_size_kb)
        resp.headers["X-Output-Dims"]      = f"{out_w}×{out_h}"
        resp.headers["Content-Disposition"] = f'attachment; filename="{out_name}"'

        # Schedule cleanup after response is fully sent to avoid deleting files while Flask streams them
        @after_this_request
        def _cleanup_after(resp2):
            try:
                _cleanup(in_path, out_png, out_jpg)
            except Exception:
                pass
            return resp2

        cleanup_scheduled = True
        return resp

    except Exception as e:
        import traceback
        print(f"[{job_id}] ❌ Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Only remove leftover temp files if we didn't schedule after-request cleanup
        if not cleanup_scheduled:
            try:
                for p in (in_path, out_png, out_jpg):
                    if p and p.exists():
                        p.unlink()
            except Exception:
                pass


# ─── ERROR HANDLERS ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    print(f"[Server] ❌ Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# ─── STARTUP ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    os_name, arch = _detect_platform()

    print("=" * 70)
    print("  ⚡ PixelForge AI — Local REST API Server")
    print(f"  Version      : {VERSION}")
    print(f"  Platform     : {os_name} / {arch}")
    print(f"  Port         : {PORT}")
    print(f"  API Key      : {API_KEY}")
    print(f"  Timeout      : {PROCESS_TIMEOUT}s ({PROCESS_TIMEOUT//60}m)")
    print("=" * 70)

    def preload():
        global ENGINE_PATH
        ENGINE_PATH = get_engine_path()
        if ENGINE_PATH:
            print("\n[PixelForge] ✅ Engine ready — accepting requests!")
            ENGINE_READY.set()
        else:
            print("\n[PixelForge] ⚠️  Engine unavailable — fix errors above and restart.")

    threading.Thread(target=preload, daemon=True).start()

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print(f"\n  📍 API      → http://localhost:{PORT}")
    print(f"  🖼️  Frontend → http://localhost:{PORT}/ui   ← open this in browser")
    print(f"  📍 Network  → http://{local_ip}:{PORT}")
    print(f"  🔑 API Key  → {API_KEY}")
    print("=" * 70 + "\n")

    app.run(host=HOST, port=PORT, debug=False, threaded=True)