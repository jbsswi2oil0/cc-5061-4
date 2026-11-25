"""Video stabilization motion-analysis player - web server.

Start the interface:

    cd /workspace
    python3 app.py
    # open http://127.0.0.1:8000

The server exposes the stabilization kernel over JSON, runs analysis/render
jobs on background threads, and serves the player UI from web/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from stabilizer.pipeline import (  # noqa: E402
    DEFAULT_PARAMS,
    compute_plan,
    plan_frames_json,
    plan_metrics,
    render_plan,
)
from stabilizer.acceptance import run_acceptance  # noqa: E402
from stabilizer.synth import SAMPLE_INFO, SAMPLE_DIR, render_sample  # noqa: E402
from stabilizer.testfield import TESTFIELD_INFO, render_testfield  # noqa: E402
from stabilizer.tracking import analyze_video  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "runs", "jobs")
os.makedirs(RUNS_DIR, exist_ok=True)

ACCEPT_DIR = os.path.join(ROOT, "reports", "testfield")
os.makedirs(ACCEPT_DIR, exist_ok=True)

ALL_SAMPLE_INFO = {**SAMPLE_INFO, **TESTFIELD_INFO}

app = Flask(
    __name__,
    static_folder=os.path.join(ROOT, "web", "static"),
    template_folder=os.path.join(ROOT, "web", "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 900 * 1024 * 1024

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
BENCH_STATE = {"running": False, "proc": None, "log": ""}
ACCEPT_STATE = {"running": False, "error": None, "message": ""}

ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def job_dir(jid: str) -> str:
    return os.path.join(RUNS_DIR, jid)


def public_job(j: dict) -> dict:
    return {
        "id": j["id"],
        "stage": j["stage"],
        "progress": j.get("progress", 0.0),
        "message": j.get("message", ""),
        "error": j.get("error"),
        "source_url": f"/api/jobs/{j['id']}/source?t={j.get('src_mtime', 0)}",
        "video_url": (
            f"/api/jobs/{j['id']}/stabilized.mp4?v={j.get('video_version', 0)}"
            if j.get("has_video")
            else None
        ),
        "params": j.get("params", DEFAULT_PARAMS),
        "video_version": j.get("video_version", 0),
        "summary": j.get("summary"),
        "n_frames": j.get("n_frames"),
        "fps": j.get("fps"),
        "width": j.get("width"),
        "height": j.get("height"),
        "has_result": j.get("result") is not None
        or os.path.exists(os.path.join(job_dir(j["id"]), "result.json")),
    }


def _set_progress(j: dict, frac: float, msg: str) -> None:
    j["progress"] = float(max(0.0, min(1.0, frac)))
    j["message"] = msg


def _analysis_worker(jid: str, source_path: str, model: str,
                     render_params: dict | None):
    j = JOBS[jid]
    try:
        j["stage"] = "analyzing"
        j["error"] = None
        _set_progress(j, 0.02, "读取视频并检测特征")

        def cb(frac, msg):
            _set_progress(j, frac * 0.55, msg)

        an = analyze_video(source_path, model=model, progress=cb)
        j["analysis"] = an
        j["summary"] = an.summary()
        j["n_frames"] = an.n_frames
        j["fps"] = an.fps
        j["width"] = an.width
        j["height"] = an.height
        j["stage"] = "ready"
        j["progress"] = 0.6
        params = render_params or j.get("params") or dict(DEFAULT_PARAMS)
        _render_worker(jid, params)
    except Exception as e:  # surfaced to the UI as failure feedback
        traceback.print_exc()
        j["stage"] = "error_analysis"
        j["error"] = f"{type(e).__name__}: {e}"
        j["message"] = "分析失败"


def _render_worker(jid: str, params: dict):
    j = JOBS[jid]
    an = j.get("analysis")
    if an is None:
        j["stage"] = "error_render"
        j["error"] = "运动分析已失效，请重新分析"
        return
    try:
        j["stage"] = "rendering"
        j["error"] = None
        j["params"] = {
            "smoothness": int(params.get("smoothness", DEFAULT_PARAMS["smoothness"])),
            "min_fov": float(params.get("min_fov", DEFAULT_PARAMS["min_fov"])),
            "model": an.model,
        }

        def cb(frac, msg):
            _set_progress(j, 0.6 + 0.4 * frac, msg)

        result = render_plan(
            an,
            job_dir(jid),
            j["params"]["smoothness"],
            j["params"]["min_fov"],
            progress=cb,
            verify=True,
        )
        j["result"] = result
        j["has_video"] = True
        j["video_version"] = int(time.time())
        j["stage"] = "done"
        j["progress"] = 1.0
        j["message"] = "完成"
    except Exception as e:
        traceback.print_exc()
        j["stage"] = "error_render"
        j["error"] = f"{type(e).__name__}: {e}"
        j["message"] = "渲染失败，可调整参数后重试"


def _create_job(source_path: str, source_filename: str, model: str,
                start: bool = True) -> str:
    jid = uuid.uuid4().hex[:12]
    os.makedirs(job_dir(jid), exist_ok=True)
    ext = os.path.splitext(source_filename)[1].lower() or ".mp4"
    stored_name = "source" + ext
    dest = os.path.join(job_dir(jid), stored_name)
    if os.path.abspath(source_path) != os.path.abspath(dest):
        shutil.copy(source_path, dest)
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid,
            "stage": "queued",
            "progress": 0.0,
            "message": "排队中",
            "error": None,
            "source_file": stored_name,
            "src_mtime": int(os.path.getmtime(dest)),
            "has_video": False,
            "video_version": 0,
            "params": dict(DEFAULT_PARAMS, model=model),
            "analysis": None,
            "result": None,
        }
    if start:
        t = threading.Thread(
            target=_analysis_worker,
            args=(jid, dest, model, dict(DEFAULT_PARAMS)),
            daemon=True,
        )
        t.start()
    return jid


# ---------------------------------------------------------------- pages

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------- samples

@app.get("/api/samples")
def list_samples():
    out = []
    for key, info in ALL_SAMPLE_INFO.items():
        path = os.path.join(SAMPLE_DIR, info["file"])
        out.append(
            {
                "key": key,
                "title": info["title"],
                "desc": info["desc"],
                "testfield": key in TESTFIELD_INFO,
                "exists": os.path.exists(path),
                "size_mb": round(os.path.getsize(path) / 1e6, 2)
                if os.path.exists(path)
                else 0,
            }
        )
    return jsonify(out)


@app.post("/api/samples/<key>/create")
def create_sample(key: str):
    if key not in ALL_SAMPLE_INFO:
        return jsonify({"error": "未知样片"}), 404
    force = bool((request.json or {}).get("force"))
    path = os.path.join(SAMPLE_DIR, ALL_SAMPLE_INFO[key]["file"])
    if force or not os.path.exists(path):
        if key in TESTFIELD_INFO:
            render_testfield(key)
        else:
            render_sample(key)
    model = (request.json or {}).get("model", "similarity")
    jid = _create_job(path, ALL_SAMPLE_INFO[key]["file"], model)
    return jsonify({"job_id": jid})


# ---------------------------------------------------------------- upload

@app.post("/api/upload")
def upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify({"error": "没有收到视频文件（字段名 video）"}), 400
    ext = os.path.splitext(secure_filename(f.filename))[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(
            {"error": f"不支持的扩展名 {ext}，支持 {sorted(ALLOWED_EXT)}"}
        ), 400
    jid = uuid.uuid4().hex[:12]
    os.makedirs(job_dir(jid), exist_ok=True)
    dest = os.path.join(job_dir(jid), "source" + ext)
    f.save(dest)
    model = request.form.get("model", "similarity")
    new_jid = _create_job(dest, f"source{ext}", model)
    if new_jid != jid:
        # _create_job makes its own id; keep the simpler path by removing jid dir
        shutil.rmtree(job_dir(jid), ignore_errors=True)
    return jsonify({"job_id": new_jid})


# ---------------------------------------------------------------- jobs

@app.get("/api/jobs/<jid>")
def job_status(jid: str):
    j = JOBS.get(jid)
    if j is None:
        # server restarted: job dirs still exist but memory analysis is gone
        if os.path.isdir(job_dir(jid)):
            return jsonify(
                {
                    "id": jid,
                    "stage": "error_analysis",
                    "progress": 0,
                    "message": "服务已重启，内存中的运动分析丢失",
                    "error": "请重新载入或重新分析该任务",
                    "source_url": None,
                    "video_url": f"/api/jobs/{jid}/stabilized.mp4"
                    if os.path.exists(os.path.join(job_dir(jid), "stabilized.mp4"))
                    else None,
                    "has_result": os.path.exists(
                        os.path.join(job_dir(jid), "result.json")
                    ),
                    "params": DEFAULT_PARAMS,
                }
            )
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(public_job(j))


@app.post("/api/jobs/<jid>/analyze")
def reanalyze(jid: str):
    j = JOBS.get(jid)
    if j is None:
        return jsonify({"error": "任务不存在或服务已重启，请重新载入视频"}), 404
    if j["stage"] in ("analyzing", "rendering"):
        return jsonify({"error": "任务正在处理中"}), 409
    model = (request.json or {}).get("model", "similarity")
    if model not in ("similarity", "affine"):
        return jsonify({"error": "不支持的模型"}), 400
    source_path = os.path.join(job_dir(jid), j["source_file"])
    params = j.get("params") or dict(DEFAULT_PARAMS)
    t = threading.Thread(
        target=_analysis_worker, args=(jid, source_path, model, params), daemon=True
    )
    t.start()
    return jsonify({"ok": True})


@app.post("/api/jobs/<jid>/plan")
def plan_preview(jid: str):
    j = JOBS.get(jid)
    if j is None or j.get("analysis") is None:
        return jsonify({"error": "运动分析尚未完成"}), 409
    if j["stage"] == "analyzing":
        return jsonify({"error": "仍在分析中"}), 409
    data = request.json or {}
    smoothness = float(data.get("smoothness", DEFAULT_PARAMS["smoothness"]))
    min_fov = float(data.get("min_fov", DEFAULT_PARAMS["min_fov"]))
    try:
        plan = compute_plan(j["analysis"], smoothness, min_fov)
        return jsonify(
            {
                "params": {"smoothness": int(smoothness), "min_fov": min_fov},
                "frames": plan_frames_json(j["analysis"], plan),
                "metrics": plan_metrics(j["analysis"], plan),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"规划失败: {e}"}), 500


@app.post("/api/jobs/<jid>/render")
def render(jid: str):
    j = JOBS.get(jid)
    if j is None or j.get("analysis") is None:
        return jsonify({"error": "运动分析尚未完成或已失效"}), 409
    if j["stage"] in ("analyzing", "rendering"):
        return jsonify({"error": "已有渲染任务在运行"}), 409
    data = request.json or {}
    params = {
        "smoothness": int(data.get("smoothness", j["params"]["smoothness"])),
        "min_fov": float(data.get("min_fov", j["params"]["min_fov"])),
    }
    t = threading.Thread(target=_render_worker, args=(jid, params), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.get("/api/jobs/<jid>/result")
def result(jid: str):
    j = JOBS.get(jid)
    if j is not None and j.get("result") is not None:
        return jsonify(j["result"])
    p = os.path.join(job_dir(jid), "result.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "结果尚未生成"}), 404


@app.get("/api/jobs/<jid>/source")
def source_media(jid: str):
    j = JOBS.get(jid)
    if j is not None:
        return send_from_directory(job_dir(jid), j["source_file"])
    # fallback: find source.* on disk
    d = job_dir(jid)
    for name in os.listdir(d) if os.path.isdir(d) else []:
        if name.startswith("source."):
            return send_from_directory(d, name)
    return jsonify({"error": "源视频丢失"}), 404


@app.get("/api/jobs/<jid>/stabilized.mp4")
def stabilized_media(jid: str):
    return send_from_directory(job_dir(jid), "stabilized.mp4")


# ---------------------------------------------------------------- benchmark

# ---------------------------------------------------------------- acceptance

@app.get("/api/acceptance")
def get_acceptance():
    p = os.path.join(ACCEPT_DIR, "acceptance.json")
    if not os.path.exists(p):
        return jsonify({"ready": False, "running": ACCEPT_STATE["running"]})
    with open(p, encoding="utf-8") as f:
        rep = json.load(f)
    return jsonify(
        {
            "ready": True,
            "running": ACCEPT_STATE["running"],
            "message": ACCEPT_STATE.get("message", ""),
            "report": rep,
        }
    )


@app.get("/api/acceptance/artifacts/<path:name>")
def acceptance_artifact(name: str):
    return send_from_directory(ACCEPT_DIR, name)


@app.post("/api/acceptance/run")
def run_acceptance_api():
    if ACCEPT_STATE["running"]:
        return jsonify({"ok": True, "already_running": True})
    data = request.json or {}
    smoothness = int(data.get("smoothness", DEFAULT_PARAMS["smoothness"]))
    min_fov = float(data.get("min_fov", DEFAULT_PARAMS["min_fov"]))
    ACCEPT_STATE.update(running=True, error=None, message="启动")

    def worker():
        try:
            def prog(frac, msg):
                ACCEPT_STATE["message"] = f"{int(frac * 100)}% · {msg}"

            run_acceptance(ACCEPT_DIR, smoothness=smoothness, min_fov=min_fov,
                           progress=prog)
            ACCEPT_STATE["message"] = "完成"
        except Exception as e:
            traceback.print_exc()
            ACCEPT_STATE["error"] = f"{type(e).__name__}: {e}"
            ACCEPT_STATE["message"] = "失败"
        finally:
            ACCEPT_STATE["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


# ---------------------------------------------------------------- benchmark

@app.get("/api/benchmark")
def get_benchmark():
    p = os.path.join(ROOT, "reports", "benchmark.json")
    if not os.path.exists(p):
        return jsonify({"ready": False})
    with open(p, encoding="utf-8") as f:
        return jsonify({"ready": True, "report": json.load(f)})


@app.get("/api/benchmark/status")
def benchmark_status():
    proc = BENCH_STATE.get("proc")
    running = bool(BENCH_STATE["running"] and proc and proc.poll() is None)
    tail = ""
    log_path = os.path.join(ROOT, "reports", "benchmark_server.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, "rb") as f:
                f.seek(max(0, os.path.getsize(log_path) - 1200))
                tail = f.read().decode("utf-8", "ignore")
        except OSError:
            tail = ""
    return jsonify({"running": running, "tail": tail[-800:]})


@app.post("/api/benchmark/run")
def run_benchmark():
    if BENCH_STATE["running"]:
        return jsonify({"ok": True, "already_running": True})
    log = open(os.path.join(ROOT, "reports", "benchmark_server.log"), "w")
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "scripts", "run_benchmark.py")],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    BENCH_STATE.update(running=True, proc=proc)

    def waiter():
        proc.wait()
        BENCH_STATE["running"] = False

    threading.Thread(target=waiter, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"视频稳定运动分析播放器已启动: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
