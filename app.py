import os
import tempfile
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yt_dlp
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str

def download_audio(url: str) -> str:
    out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.m4a")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    if not os.path.exists(out_path):
        base = out_path.rsplit(".", 1)[0]
        for ext in ["m4a", "mp3", "webm", "opus"]:
            cand = f"{base}.{ext}"
            if os.path.exists(cand):
                return cand
    return out_path

@app.post("/transcribe")
def transcribe(req: VideoRequest):
    audio_file = None
    try:
        audio_file = download_audio(req.url)
        uploaded = genai.upload_file(audio_file)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([
            "Transcribe the spoken audio in this file, then return a clean, "
            "well-punctuated, polished transcript. Fix filler words and format "
            "into readable paragraphs. Return only the transcript.",
            uploaded,
        ])
        return {"success": True, "transcript": result.text}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><head><title>Video Transcriber</title></head>
    <body style="font-family:sans-serif;max-width:600px;margin:40px auto;">
      <h2>Video Transcriber</h2>
      <input id="url" style="width:100%;padding:8px;" placeholder="Paste Instagram or other video link"/>
      <br><br>
      <button onclick="go()" style="padding:8px 16px;">Transcribe</button>
      <p id="status"></p>
      <textarea id="out" style="width:100%;height:250px;"></textarea>
      <script>
        async function go(){
          const url = document.getElementById('url').value;
          document.getElementById('status').innerText = 'Processing... please wait';
          const r = await fetch('/transcribe', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({url})
          });
          const d = await r.json();
          document.getElementById('status').innerText = d.success ? 'Done!' : 'Error: '+d.error;
          document.getElementById('out').value = d.success ? d.transcript : '';
        }
      </script>
    </body></html>
    """
