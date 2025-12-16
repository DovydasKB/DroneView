from __future__ import annotations

import os
import sys
import shutil
import signal
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

HERE = Path(__file__).resolve().parent


MEDIAMTX_BIN = HERE / ("mediamtx.exe" if os.name == "nt" else "mediamtx")
CFG_PATH = HERE / "mediamtx.generated.yml"

RTMP_PORT = 1935
HLS_PORT = 8888
WEB_PORT = 5000

MEDIAMTX_CONFIG = f"""
logLevel: info

rtmp: yes
rtmpAddress: :{RTMP_PORT}

hls: yes
hlsAddress: :{HLS_PORT}
hlsAllowOrigins: ['*']
# Generate HLS immediately when a publisher connects (reduces initial wait)
hlsAlwaysRemux: yes

# Accept any incoming publish path (e.g. /mystream or /live/stream)
paths:
  "~^.*$":
    source: publisher
""".lstrip()

app = FastAPI()
mediamtx_proc: subprocess.Popen | None = None


@app.get("/", response_class=HTMLResponse)
def index():

    return HTMLResponse(f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RTMP → Web (HLS)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    input {{ padding:10px; min-width: 320px; }}
    button {{ padding:10px 14px; cursor:pointer; }}
    video {{ width: 100%; max-width: 980px; background:#000; margin-top:12px; }}
    .hint {{ color:#444; margin-top:10px; line-height:1.4; }}
    code {{ background:#f3f3f3; padding:2px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <h2>Live viewer (DJI Fly RTMP → MediaMTX → HLS)</h2>

  <div class="row">
    <label>Stream path:</label>
    <input id="path" value="live/stream" />
    <button onclick="play()">Play</button>
    <button onclick="stop()">Stop</button>
  </div>

  <div class="hint">
    DJI Fly usually publishes to something like:<br/>
    <code>rtmp://YOUR_PC_IP:{RTMP_PORT}/live/stream</code><br/>
    This page plays HLS from:<br/>
    <code>http://YOUR_PC_IP:{HLS_PORT}/live/stream/index.m3u8</code>
  </div>

  <video id="video" controls autoplay muted playsinline></video>

  <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
  <script>
    let hls = null;

    function hlsUrl() {{
      const p = document.getElementById('path').value.trim().replace(/^\\/+/, '');
      return `http://${{location.hostname}}:{HLS_PORT}/${{p}}/index.m3u8`;
    }}

    function stop() {{
      const video = document.getElementById('video');
      if (hls) {{
        hls.destroy();
        hls = null;
      }}
      video.pause();
      video.removeAttribute('src');
      video.load();
    }}

    function play() {{
      stop();
      const video = document.getElementById('video');
      const url = hlsUrl();

      if (video.canPlayType('application/vnd.apple.mpegurl')) {{
        // Safari (native HLS)
        video.src = url;
        video.play().catch(()=>{{}});
        return;
      }}

      if (window.Hls) {{
        hls = new Hls({{ lowLatencyMode: true }});
        hls.loadSource(url);
        hls.attachMedia(video);
        hls.on(Hls.Events.ERROR, function (event, data) {{
          console.log("HLS error:", data);
        }});
      }} else {{
        alert("hls.js failed to load. Check your internet access (CDN).");
      }}
    }}
  </script>
</body>
</html>
""")


def start_mediamtx():
    global mediamtx_proc

    if not MEDIAMTX_BIN.exists():
        raise FileNotFoundError(
            f"MediaMTX binary not found: {MEDIAMTX_BIN}\n"
            f"Put mediamtx (or mediamtx.exe) in the same folder as app.py."
        )

    CFG_PATH.write_text(MEDIAMTX_CONFIG, encoding="utf-8")

    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        preexec_fn = os.setsid

    mediamtx_proc = subprocess.Popen(
        [str(MEDIAMTX_BIN), str(CFG_PATH)],
        cwd=str(HERE),
        stdout=sys.stdout,
        stderr=sys.stderr,
        creationflags=creationflags,
        preexec_fn=preexec_fn,
    )


def stop_mediamtx():
    global mediamtx_proc
    if mediamtx_proc is None:
        return

    try:
        if os.name == "nt":
            mediamtx_proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(os.getpgid(mediamtx_proc.pid), signal.SIGINT)
    except Exception:
        mediamtx_proc.terminate()

    try:
        mediamtx_proc.wait(timeout=5)
    except Exception:
        mediamtx_proc.kill()

    mediamtx_proc = None


def main():
    start_mediamtx()

    def handle_exit(*_):
        stop_mediamtx()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    print("\n=== RUNNING ===")
    print(f"RTMP ingest:   rtmp://YOUR_PC_IP:{RTMP_PORT}/live/stream")
    print(f"HLS playback:  http://YOUR_PC_IP:{HLS_PORT}/live/stream/index.m3u8")
    print(f"Web UI:        http://127.0.0.1:{WEB_PORT}\n")

    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, log_level="info")


if __name__ == "__main__":
    main()
