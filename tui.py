#!/usr/bin/env python3
"""
OpenVoice - Mac-Optimized Voice Assistant with Textual TUI

Controls:
  - Hold OPTION → Voice recording (release to send)
  - Hold CONTROL → Video + Voice recording (release to send)
  - Type text → Send text message
  - Esc or 'quit' → Exit
"""

import os
import sys
import threading
import queue
import argparse
import json
import time
import base64
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Silence imports
# ═══════════════════════════════════════════════════════════════════════════════
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "1"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"

import warnings
warnings.filterwarnings("ignore")

_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import mlx.core as mx
mx.set_default_device(mx.gpu)

import sounddevice as sd
import parakeet_mlx
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.generate import load_audio
from mlx_audio.tts.audio_player import AudioPlayer
import requests

try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Try to import pynput for keyboard shortcuts
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

sys.stderr = _stderr

# Textual imports
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
from textual.widgets import Static, Input, ListView, ListItem, Button, Label
from textual.reactive import reactive
from textual.message import Message
from textual import work
from rich.text import Text

import uuid
from datetime import datetime
import glob

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000
CHUNK_SIZE = 512


# ═══════════════════════════════════════════════════════════════════════════════
# Simple energy-based VAD
# ═══════════════════════════════════════════════════════════════════════════════
def trim_silence_simple(audio, threshold=0.01):
    """Trim silence using simple energy threshold."""
    if len(audio) < SAMPLE_RATE * 0.3:
        return audio
    
    chunk_size = int(SAMPLE_RATE * 0.03)
    start_idx = 0
    for i in range(0, len(audio) - chunk_size, chunk_size):
        if np.abs(audio[i:i+chunk_size]).mean() > threshold:
            start_idx = max(0, i - chunk_size)
            break
    
    end_idx = len(audio)
    for i in range(len(audio) - chunk_size, chunk_size, -chunk_size):
        if np.abs(audio[i:i+chunk_size]).mean() > threshold:
            end_idx = min(len(audio), i + chunk_size * 2)
            break
    
    return audio[start_idx:end_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Gesture Normalization for ChatterBox TTS
# ═══════════════════════════════════════════════════════════════════════════════
import re

# Mapping of common LLM gesture patterns to ChatterBox tags
GESTURE_MAP = {
    r'\(laughs?\)': '[laugh]',
    r'\*laughs?\*': '[laugh]',
    r'\(sighs?\)': '[sigh]',
    r'\*sighs?\*': '[sigh]',
    r'\(chuckles?\)': '[chuckle]',
    r'\*chuckles?\*': '[chuckle]',
    r'\(coughs?\)': '[cough]',
    r'\*coughs?\*': '[cough]',
    r'\(sniffs?\)': '[sniff]',
    r'\*sniffs?\*': '[sniff]',
    r'\(gasps?\)': '[gasp]',
    r'\*gasps?\*': '[gasp]',
    r'\(groans?\)': '[groan]',
    r'\*groans?\*': '[groan]',
    r'\(clears? throat\)': '[clear throat]',
    r'\*clears? throat\*': '[clear throat]',
}

def normalize_gestures(text: str) -> str:
    """Convert LLM gesture formats to ChatterBox TTS [tag] format."""
    result = text
    for pattern, replacement in GESTURE_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CSS Styles
# ═══════════════════════════════════════════════════════════════════════════════
CSS = """
Screen {
    background: #0d1117;
}

#main-layout {
    layout: horizontal;
    height: 1fr;
}

/* Sidebar Area */
#sidebar {
    width: 32;
    height: 100%;
    background: #161b22;
    border-right: solid #30363d;
    layout: vertical;
}

#app-header {
    width: 100%;
    height: auto;
    background: #161b22;
    border-bottom: solid #30363d;
    content-align: center middle;
    padding: 2 0;
    margin: 0;
}

.sidebar-title {
    display: none;
}

#new-chat-container {
    height: auto;
    padding: 1;
    content-align: center middle;
}

#new-chat-btn {
    width: 100%;
    height: 3;
    background: #238636;
    color: white;
    margin-bottom: 1;
}

#mute-btn {
    width: 100%;
    height: 3;
    background: #30363d;
    color: white;
}

#mute-btn.--muted {
    background: #d29922;
}

#chat-list {
    height: 1fr;
    background: #161b22;
    scrollbar-size: 1 1;
}

/* Chat Items */
ChatListItem {
    height: 3;
    background: #161b22;
    border-bottom: solid #1c2128;
    layout: horizontal;
    align: left middle;
    padding: 0 1;
    margin: 0;
}

ChatListItem:hover {
    background: #161b22;
}

ChatListItem:focus {
    background: #161b22;
}



ChatListItem.--highlight {
    border-left: solid #c9a0ff;
    background: #1c2128;
}

.chat-item-info {
    width: 1fr;
    height: auto;
    layout: vertical;
}

.chat-item-title {
    color: #c9d1d9;
    text-style: bold;
    margin-bottom: 0;
    height: 1;
}

.chat-item-date {
    color: #8b949e;
    text-style: dim;
    height: 1;
}

.delete-btn {
    width: 3;
    height: 1;
    min-width: 3;
    color: #8b949e;
    background: transparent;
    border: none;
    padding: 0;
    content-align: center middle;
    margin-right: 2;
}

.delete-btn:hover {
    color: #f85149;
    background: #30363d;
}

#mute-btn:hover {
    background: #30363d;
}

#mute-btn.--muted:hover {
    background: #d29922;
}

#mute-btn > .textual-button--label, 
#new-chat-btn > .textual-button--label {
    background: transparent;
}

/* Main Content Area */
#chat-area {
    width: 1fr;
    height: 100%;
    layout: vertical;
}

#banner-label {
    width: 100%;
    color: #c9a0ff;
    text-align: center;
}

#chat-container {
    height: 1fr;
    background: #0d1117;
    padding: 1 2;
    overflow-y: auto;
    scrollbar-size: 1 1;
    align: center top;
}

#input-area {
    height: auto;
    background: #161b22;
    border-top: solid #30363d;
    padding: 1 2;
    layout: vertical;
}

#attachment-label {
    height: 1;
    color: #8b949e;
    margin-bottom: 0;
    display: none;
    content-align: right middle;
}

#attachment-label.--visible {
    display: block;
}

#input-row {
    height: 3;
    layout: horizontal;
    align: left middle;
}

#attach-btn {
    width: 5;
    height: 3;
    min-width: 5;
    background: #0d1117;
    border: round #30363d;
    color: #8b949e;
    padding: 0;
    content-align: center middle;
    margin-right: 1;
    outline: none;
}

#attach-btn:focus {
    background: #0d1117;
    border: round #30363d;
    outline: none;
}

#attach-btn:hover {
    color: #58a6ff;
}

#user-input {
    width: 1fr;
    height: 3;
    background: #0d1117;
    border: round #30363d;
    color: #c9d1d9;
    margin: 0;
    outline: none;
}

#user-input:focus {
    border: round #30363d;
    background: #0d1117;
    outline: none;
}

#status-bar {
    height: 1;
    background: transparent;
    color: #8b949e;
    text-align: center;
}

.status-listening {
    color: #3fb950;
    text-style: bold;
}

.status-camera {
    color: #d29922;
    text-style: bold;
}

/* Messages */
.user-message {
    width: 85%;
    height: auto;
    color: #f0f6fc;
    background: #0d47a1;
    padding: 1 2;
    border-left: solid #58a6ff;
    margin-bottom: 1;
}
.ai-message {
    width: 85%;
    height: auto;
    color: #c9d1d9;
    background: #1c2128;
    padding: 1 2;
    border-left: solid #c9a0ff;
    margin-bottom: 1;
}

.system-message {
    display: none;
}
"""

OPENCLI_BANNER = """
██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗     ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║     ██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║
╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗███████╗██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Textual Widgets
# ═══════════════════════════════════════════════════════════════════════════════
from textual.widget import Widget

class ChatMessage(Static):
    """A single chat message widget."""
    
    def __init__(self, sender: str, content: str, msg_type: str = "user"):
        self.msg_type = msg_type
        # Clean text for AI messages immediately
        display_content = self._clean_text(content) if msg_type == "ai" else content
        super().__init__(display_content)
        self.add_class(f"{msg_type}-message")

    def _clean_text(self, text: str) -> str:
        """Keep original text to show character tags like (laughs)."""
        return text.strip()
    
    def update_content(self, new_content: str):
        """Update message content for streaming."""
        display_content = self._clean_text(new_content) if self.msg_type == "ai" else new_content
        if not display_content and self.msg_type == "ai":
            return
        self.update(display_content)


class StatusBar(Static):
    """Status bar showing current state."""
    
    is_listening = reactive(False)
    is_camera_on = reactive(False)
    status_text = reactive("")
    
    def render(self) -> Text:
        text = Text(justify="center")
        text.append("OPTION", style="bold #58a6ff")
        text.append("=voice   ", style="dim")
        text.append("CTRL", style="bold #58a6ff")
        text.append("=camera   ", style="dim")
        text.append("Type", style="bold #58a6ff")
        text.append("=chat   ", style="dim")
        text.append("Esc", style="bold #58a6ff")
        text.append("=quit", style="dim")
        
        if self.status_text:
            text.append("   │   ", style="dim")
            if self.is_listening:
                text.append(self.status_text, style="bold #3fb950")
            elif self.is_camera_on:
                text.append(self.status_text, style="bold #d29922")
            else:
                text.append(self.status_text, style="bold #8b949e")
        return text
    
    def watch_is_listening(self, value: bool):
        if value:
            self.add_class("status-listening")
            self.status_text = "🎤 Recording..."
        else:
            self.remove_class("status-listening")
            if not self.is_camera_on:
                self.status_text = ""
    
    def watch_is_camera_on(self, value: bool):
        if value:
            self.add_class("status-camera")
            self.status_text = "📷 Recording..."
        else:
            self.remove_class("status-camera")
            if not self.is_listening:
                self.status_text = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Backend (Audio/Video/LLM)
# ═══════════════════════════════════════════════════════════════════════════════
class ChatManager:
    """Manages chat history and persistence."""
    
    def __init__(self, storage_dir="chats"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
    def list_chats(self):
        """List all available chats sorted by recent."""
        chats = []
        for filepath in glob.glob(os.path.join(self.storage_dir, "*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    chats.append({
                        "id": data.get("id"),
                        "title": data.get("title", "Untitled Chat"),
                        "timestamp": data.get("timestamp_last", 0),
                        "filename": filepath
                    })
            except:
                pass
        return sorted(chats, key=lambda x: x["timestamp"], reverse=True)

    def load_chat(self, chat_id):
        """Load a specific chat."""
        filepath = os.path.join(self.storage_dir, f"{chat_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

    def save_chat(self, chat_id, messages):
        """Save chat to file."""
        if not messages:
            return
            
        filepath = os.path.join(self.storage_dir, f"{chat_id}.json")
        
        # Determine title from first user message if possible
        title = "New Chat"
        for msg in messages:
            if msg["role"] == "user":
                title = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                break
        
        data = {
            "id": chat_id,
            "title": title,
            "timestamp_created": time.time(), # This is a simplification; ideally preserve creation time
            "timestamp_last": time.time(),
            "messages": messages
        }
        
        # If updating existing, preserve key metadata
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing = json.load(f)
                    data["timestamp_created"] = existing.get("timestamp_created", data["timestamp_created"])
                    # Keep original title if it was custom, but here we just auto-update for now
            except:
                pass
                
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def delete_chat(self, chat_id):
        """Delete a chat file."""
        filepath = os.path.join(self.storage_dir, f"{chat_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)


class OpenVoiceBackend:
    """Backend handling STT, TTS, LLM, and recording."""
    
    def __init__(self, args):
        self.args = args
        self.chat_history = []
        self.current_chat_id = str(uuid.uuid4())
        self.chat_manager = ChatManager()
        self.is_muted = False

        
        # Models
        self.stt = None
        self.tts = None
        self.tts_queue = queue.Queue()
        self.player = None
        self.ref_audio = None
        
        # Recording state
        self.is_recording = False
        self.is_recording_video = False
        self.audio_chunks = []
        self.video_frames = []
        self.lock = threading.Lock()
        
        # Audio stream
        self.audio_stream = None
    
    def load_models(self, on_status=None):
        """Load models on Metal GPU."""
        if on_status:
            on_status("Loading STT model...")
        
        # STT
        self.stt = parakeet_mlx.from_pretrained(self.args.stt_model)
        
        # Warm up STT
        dummy = mx.zeros((SAMPLE_RATE,), dtype=mx.float32)
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            t.add_audio(dummy)
        mx.eval(mx.array([0]))
        
        if on_status:
            on_status("Loading TTS model...")
        
        # TTS
        self.tts = load_tts(self.args.tts_model)
        self.player = AudioPlayer(sample_rate=self.tts.sample_rate)
        
        # Load reference audio
        if os.path.exists(self.args.ref_audio):
            self.ref_audio = load_audio(self.args.ref_audio, self.tts.sample_rate)
            mx.eval(self.ref_audio)
        
        # Warm up TTS
        for res in self.tts.generate(text="Hi", ref_audio=self.ref_audio, verbose=False, stream=True):
            if hasattr(res, 'audio') and res.audio is not None:
                mx.eval(res.audio)
                break
        
        # Start TTS worker
        threading.Thread(target=self._tts_worker, daemon=True).start()
        
        # Start audio stream
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            callback=self._audio_callback, blocksize=CHUNK_SIZE
        )
        self.audio_stream.start()
        
        if on_status:
            on_status("Ready")
    
    def _tts_worker(self):
        """Background TTS thread."""
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                # Debug: Print the text being sent to TTS
                print(f"[TTS DEBUG] Speaking: {text}")
                for res in self.tts.generate(text=text, ref_audio=self.ref_audio, verbose=False, stream=True):
                    if hasattr(res, 'audio') and res.audio is not None:
                        mx.eval(res.audio)
                        self.player.queue_audio(res.audio)
            except Exception as e:
                print(f"[TTS DEBUG] Error: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        with self.lock:
            if self.is_recording or self.is_recording_video:
                self.audio_chunks.append(indata.copy())
    
    def speak(self, text):
        """Queue text for TTS with gesture normalization."""
        if self.is_muted:
            return
        if text and text.strip():
            # Normalize gestures for ChatterBox TTS (e.g., (laughs) -> [laugh])
            normalized_text = normalize_gestures(text)
            self.tts_queue.put(normalized_text)
    
    def transcribe(self, audio):
        """Speech to text."""
        if len(audio) < SAMPLE_RATE * 0.3:
            return ""
        
        with self.stt.transcribe_stream(context_size=(128, 128), depth=1) as t:
            for i in range(0, len(audio), SAMPLE_RATE):
                chunk = audio[i:i+SAMPLE_RATE]
                if len(chunk) > 0:
                    t.add_audio(mx.array(chunk, dtype=mx.float32))
            
            if t.result:
                mx.eval(mx.array([0]))
                return t.result.text.strip()
        return ""
    
    def chat(self, text, image_b64=None, on_token=None):
        """Send to LLM and stream response."""
        self.chat_history.append({"role": "user", "content": text})
        
        if image_b64:
            msg = {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
            messages = self.chat_history[:-1] + [msg]
        else:
            messages = self.chat_history
        
        response = ""
        buffer = ""
        
        try:
            r = requests.post(
                f"{self.args.llm_url}/chat/completions",
                json={"model": "local", "messages": messages, "stream": True, "max_tokens": 500},
                stream=True, timeout=60
            )
            
            for line in r.iter_lines():
                if not line:
                    continue
                data = line.decode().removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    token = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if token:
                        response += token
                        buffer += token
                        if on_token:
                            on_token(token)
                        # Speak when we have a complete sentence
                        if any(c in token for c in '.!?\n') and len(buffer) > 3:
                            threading.Thread(target=self.speak, args=(buffer,), daemon=True).start()
                            buffer = ""
                except:
                    pass
            
            if buffer.strip():
                threading.Thread(target=self.speak, args=(buffer,), daemon=True).start()
                
        except requests.exceptions.ConnectionError:
            response = "Error: Cannot connect to LLM. Is LM Studio running?"
            if on_token:
                on_token(response)
        except Exception as e:
            response = f"Error: {e}"
            if on_token:
                on_token(response)
        
        if response and not response.startswith("Error"):
            self.chat_history.append({"role": "assistant", "content": response})
            # Save chat after AI response
            self.chat_manager.save_chat(self.current_chat_id, self.chat_history)
        
        return response
    
    def new_session(self):
        """Start a new chat session."""
        self.chat_history = []
        self.current_chat_id = str(uuid.uuid4())
        
    def load_session(self, chat_id):
        """Load an existing session."""
        data = self.chat_manager.load_chat(chat_id)
        if data:
            self.chat_history = data.get("messages", [])
            self.current_chat_id = chat_id
            return self.chat_history
        return []
    
    def delete_session(self, chat_id):
        """Delete a session."""
        self.chat_manager.delete_chat(chat_id)
        if self.current_chat_id == chat_id:
            self.new_session()
            return True # Indicates current session was deleted
        return False
    
    def start_recording(self):
        """Start voice recording."""
        if self.is_recording or self.is_recording_video:
            return False
        
        with self.lock:
            self.is_recording = True
            self.audio_chunks = []
        return True
    
    def stop_recording(self):
        """Stop recording and return transcribed text."""
        if not self.is_recording:
            return None
        
        with self.lock:
            self.is_recording = False
            if not self.audio_chunks:
                return None
            audio = np.concatenate(self.audio_chunks).flatten().astype(np.float32)
            self.audio_chunks = []
        
        audio = trim_silence_simple(audio)
        
        if len(audio) > SAMPLE_RATE * 0.3:
            return self.transcribe(audio)
        return None
    
    def start_video(self):
        """Start video + audio recording."""
        if not HAS_OPENCV:
            return False
        
        if self.is_recording or self.is_recording_video:
            return False
        
        with self.lock:
            self.is_recording_video = True
            self.audio_chunks = []
            self.video_frames = []
        
        # Start video capture thread
        def capture():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return
            
            while True:
                with self.lock:
                    if not self.is_recording_video:
                        break
                
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.video_frames.append(cv2.resize(frame, (640, 480)))
                time.sleep(0.1)
            
            cap.release()
        
        threading.Thread(target=capture, daemon=True).start()
        return True
    
    def stop_video(self):
        """Stop video recording and return (text, image_b64)."""
        if not self.is_recording_video:
            return None, None
        
        with self.lock:
            self.is_recording_video = False
        
        time.sleep(0.2)
        
        # Get image
        image_b64 = None
        with self.lock:
            if self.video_frames:
                frame = self.video_frames[len(self.video_frames) // 2]
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_b64 = base64.b64encode(buf).decode()
            self.video_frames = []
        
        # Get audio
        text = ""
        with self.lock:
            if self.audio_chunks:
                audio = np.concatenate(self.audio_chunks).flatten().astype(np.float32)
                self.audio_chunks = []
                audio = trim_silence_simple(audio)
                if len(audio) > SAMPLE_RATE * 0.3:
                    text = self.transcribe(audio)
        
        if not text:
            text = "What do you see in this image?"
        
        return text, image_b64
    
    def shutdown(self):
        """Clean up resources."""
        if self.audio_stream:
            self.audio_stream.stop()
        self.tts_queue.put(None)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# Textual Widgets
# ═══════════════════════════════════════════════════════════════════════════════
class ChatListItem(ListItem):
    """A list item representing a chat session."""
    
    def __init__(self, chat_id, title, timestamp):
        super().__init__()
        self.chat_id = chat_id
        self.title_text = title
        self.timestamp = timestamp
        
    def compose(self) -> ComposeResult:
        dt = datetime.fromtimestamp(self.timestamp).strftime("%b %d %H:%M")
        
        with Horizontal():
            with Vertical(classes="chat-item-info"):
                yield Label(self.title_text, classes="chat-item-title")
                yield Label(dt, classes="chat-item-date")
                
            yield Button("X", id=f"del-{self.chat_id}", classes="delete-btn")


class OpenVoiceTUI(App):
    """OpenCLI Voice Assistant TUI."""
    
    CSS = CSS
    
    BINDINGS = [
        Binding("escape", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+n", "new_chat", "New Chat", show=True),
        Binding("ctrl+d", "delete_chat", "Delete Chat", show=True),
    ]
    
    attached_image_path = reactive("")

    def __init__(self, args, backend=None):
        super().__init__()
        self.args = args
        self.backend = backend
        self.title = "OpenCLI"
        self._chat_container = None
        self._current_ai_message = None
        self._pynput_listener = None
        self._chat_list = None
    
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Container(Label(OPENCLI_BANNER.strip(), id="banner-label"), id="app-header")
        
        with Horizontal(id="main-layout"):
            # Sidebar
            with Vertical(id="sidebar"):
                with Container(id="new-chat-container"):
                    yield Button("New Chat", id="new-chat-btn", variant="success")
                    yield Button("Mute Audio", id="mute-btn")
                
                yield ListView(id="chat-list")
            
            # Chat Area
            with Vertical(id="chat-area"):
                yield ScrollableContainer(id="chat-container")
                
                with Vertical(id="input-area"):
                    with Horizontal(id="input-row"):
                        yield Button("+", id="attach-btn")
                        yield Input(placeholder="Type message and press Enter...", id="user-input")
                    yield Label("", id="attachment-label")
                    yield StatusBar(id="status-bar")
    
    def on_mount(self):
        """Called when app is mounted."""
        self._chat_container = self.query_one("#chat-container", ScrollableContainer)
        self._chat_list = self.query_one("#chat-list", ListView)
        self.query_one("#user-input", Input).focus()
        
        # Start pynput listener
        self._start_pynput_listener()
        
        if self.backend:
            # Refresh chat list
            self._refresh_chat_list()
            self._add_system_message(f"Ready! HAS_PYNPUT={HAS_PYNPUT}. Hold OPTION=voice, CTRL=camera, or type.")
        else:
            self._add_system_message("Demo mode - no backend loaded.")
            
    def _refresh_chat_list(self):
        """Reload chat list from backend."""
        if not self.backend:
            return
            
        self._chat_list.clear()
        chats = self.backend.chat_manager.list_chats()
        
        for chat in chats:
            item = ChatListItem(chat["id"], chat["title"], chat["timestamp"])
            self._chat_list.append(item)
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        btn_id = str(event.button.id)
        
        if btn_id == "new-chat-btn":
            self.action_new_chat()
            
        elif btn_id == "mute-btn":
            self.action_toggle_mute()
            
        elif btn_id == "attach-btn":
            self.action_attach_image()
            
        elif btn_id.startswith("del-"):
            # Delete specific chat
            chat_id = btn_id.replace("del-", "")
            if self.backend:
                is_current = self.backend.delete_session(chat_id)
                if is_current:
                    self._chat_container.remove_children()
                    self._add_system_message("Chat deleted. Started new session.")
                self._refresh_chat_list()
            event.stop()
            
    def on_list_view_selected(self, event: ListView.Selected):
        """Handle chat selection."""
        item = event.item
        if isinstance(item, ChatListItem) and self.backend:
            self._load_chat(item.chat_id)
            
    def _load_chat(self, chat_id):
        """Load a chat session into the UI."""
        if not self.backend:
            return
            
        history = self.backend.load_session(chat_id)
        
        # Clear UI
        self._chat_container.remove_children()
        
        # Re-populate UI
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                 # Handle image messages
                if isinstance(content, list):
                    text_content = ""
                    has_image = False
                    for part in content:
                        if part.get("type") == "text":
                            text_content = part.get("text", "")
                        elif part.get("type") == "image_url":
                            has_image = True
                    self._add_user_message(text_content + (" (+image)" if has_image else ""))
                else:
                    self._add_user_message(content)
            elif role == "assistant":
                self._add_ai_message(content)
        
        self.query_one("#user-input", Input).focus()

    def action_new_chat(self):
        """Start a new chat."""
        if self.backend:
            self.backend.new_session()
            self._chat_container.remove_children()
            self._add_system_message("New chat started.")
            self._refresh_chat_list()
            self.query_one("#user-input", Input).focus()

    def action_delete_chat(self):
        """Delete current chat (hotkey)."""
        if not self.backend:
            return
        if self._chat_list.index is not None:
             item = self._chat_list.children[self._chat_list.index]
             if isinstance(item, ChatListItem):
                 if self.backend.delete_session(item.chat_id):
                     self._chat_container.remove_children()
                 self._refresh_chat_list()

    def action_toggle_mute(self):
        """Toggle audio mute state."""
        if self.backend:
            self.backend.is_muted = not self.backend.is_muted
            btn = self.query_one("#mute-btn", Button)
            if self.backend.is_muted:
                btn.label = "Unmute Audio"
                btn.add_class("--muted")
                self._add_system_message("Audio muted.")
            else:
                btn.label = "Mute Audio"
                btn.remove_class("--muted")
                self._add_system_message("Audio unmuted.")

    def action_attach_image(self):
        """Manually attach an image using Mac file picker."""
        import subprocess
        
        # AppleScript to choose an image file
        script = 'POSIX path of (choose file with prompt "Select an image to attach:" of type {"public.image"})'
        try:
            # We run this in a separate process to avoid blocking the TUI event loop
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()
                if path and os.path.exists(path):
                    self.attached_image_path = path
                    label = self.query_one("#attachment-label", Label)
                    label.update(f"📎 Attached: {os.path.basename(path)}")
                    label.add_class("--visible")
                    self._update_status(f"Attached: {os.path.basename(path)}")
        except Exception as e:
            self._add_system_message(f"Error selecting image: {e}")

    def _start_pynput_listener(self):
        """Start pynput keyboard listener for PTT."""
        if not HAS_PYNPUT:
            self._add_system_message("pynput not available - use text commands: 'r'=record, 'v'=video")
            return
        
        def on_press(key):
            try:
                # Debugging: Show all keys to verify listener is working
                self.call_from_thread(self._add_system_message, f"Debug: Key pressed: {key}")
                
                if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    if self.backend and self.backend.start_recording():
                        self.call_from_thread(self._on_recording_start)
                elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    if self.backend and self.backend.start_video():
                        self.call_from_thread(self._on_video_start)
            except Exception as e:
                self.call_from_thread(self._add_system_message, f"Listener Error: {e}")
        
        def on_release(key):
            try:
                if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    if self.backend:
                        self.call_from_thread(self._on_recording_stop)
                elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    if self.backend:
                        self.call_from_thread(self._on_video_stop)
            except Exception:
                pass
        
        self._pynput_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._pynput_listener.start()
    
    def _on_recording_start(self):
        """Handle recording start."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.is_listening = True
    
    def _on_recording_stop(self):
        """Handle recording stop."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.is_listening = False
        status_bar.status_text = "Processing..."
        self._process_voice_recording()
    
    @work(thread=True)
    def _process_voice_recording(self):
        """Process voice recording in background."""
        if not self.backend:
            return
        
        text = self.backend.stop_recording()
        if text:
            self.call_from_thread(self._add_user_message, text)
            self.call_from_thread(self._start_ai_response)
            
            full_response = ""
            def on_token(token):
                nonlocal full_response
                full_response += token
                self.call_from_thread(self._update_ai_response, full_response)
            
            self.backend.chat(text, on_token=on_token)
            self.call_from_thread(self._finish_ai_response)
            self.call_from_thread(self._refresh_chat_list)
        else:
            self.call_from_thread(self._add_system_message, "(No speech detected)")
        
        self.call_from_thread(self._update_status, "Ready")
    
    def _on_video_start(self):
        """Handle video start."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.is_camera_on = True
    
    def _on_video_stop(self):
        """Handle video stop."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.is_camera_on = False
        status_bar.status_text = "Processing..."
        self._process_video_recording()
    
    @work(thread=True)
    def _process_video_recording(self):
        """Process video recording in background."""
        if not self.backend:
            return
        
        text, image_b64 = self.backend.stop_video()
        if image_b64:
            self.call_from_thread(self._add_user_message, f"{text} (+image)")
            self.call_from_thread(self._start_ai_response)
            
            full_response = ""
            def on_token(token):
                nonlocal full_response
                full_response += token
                self.call_from_thread(self._update_ai_response, full_response)
            
            self.backend.chat(text, image_b64=image_b64, on_token=on_token)
            self.call_from_thread(self._finish_ai_response)
            self.call_from_thread(self._refresh_chat_list)
        else:
            self.call_from_thread(self._add_system_message, "No image captured")
        
        self.call_from_thread(self._update_status, "Ready")
    
    def _add_user_message(self, content: str):
        """Add a user message to chat."""
        msg = ChatMessage("User", content, msg_type="user")
        self._chat_container.mount(msg)
        self._chat_container.scroll_end(animate=False)
    
    def _add_ai_message(self, content: str):
        """Add an AI message to chat."""
        msg = ChatMessage("AI", content, msg_type="ai")
        self._chat_container.mount(msg)
        self._chat_container.scroll_end(animate=False)
    
    def _add_system_message(self, content: str):
        """Add a system message to chat."""
        msg = ChatMessage("System", content, msg_type="system")
        self._chat_container.mount(msg)
        self._chat_container.scroll_end(animate=False)
    
    def _start_ai_response(self):
        """Start a new streaming AI response."""
        self._current_ai_message = ChatMessage("AI", "", msg_type="ai")
        self._chat_container.mount(self._current_ai_message)
    
    def _update_ai_response(self, content: str):
        """Update the current streaming AI response."""
        if self._current_ai_message:
            self._current_ai_message.update_content(content)
            self._chat_container.scroll_end(animate=False)
    
    def _finish_ai_response(self):
        """Finish the current AI response."""
        self._current_ai_message = None
    
    def _update_status(self, status: str):
        """Update status bar."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.status_text = status
    
    async def on_input_submitted(self, event: Input.Submitted):
        """Handle text input submission."""
        user_input = event.value.strip()
        if not user_input and not self.attached_image_path:
            return
        
        event.input.value = ""
        
        if user_input.lower() == "quit":
            self.exit()
            return
        
        image_b64 = None
        if self.attached_image_path:
            try:
                with open(self.attached_image_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode()
            except Exception as e:
                self._add_system_message(f"Error reading image: {e}")
            
            # Reset attachment UI
            self.attached_image_path = ""
            label = self.query_one("#attachment-label", Label)
            label.update("")
            label.remove_class("--visible")

        # Regular text input
        self._add_user_message(user_input if user_input else "[Image Attached]")
        self._process_text_input(user_input, image_b64=image_b64)
    
    @work(thread=True)
    def _process_text_input(self, text: str, image_b64: str = None):
        """Process text input in background."""
        if not self.backend:
            self.call_from_thread(self._add_system_message, "Backend not ready")
            return
        
        self.call_from_thread(self._start_ai_response)
        
        full_response = ""
        def on_token(token):
            nonlocal full_response
            full_response += token
            self.call_from_thread(self._update_ai_response, full_response)
        
        self.backend.chat(text, image_b64=image_b64, on_token=on_token)
        self.call_from_thread(self._finish_ai_response)
        self.call_from_thread(self._refresh_chat_list)  # Update list for title change/new chat

    
    def action_quit(self):
        """Quit the application."""
        if self._pynput_listener:
            self._pynput_listener.stop()
        if self.backend:
            self.backend.shutdown()
        self.exit()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--tts-model", default="mlx-community/chatterbox-turbo-fp16")
    parser.add_argument("--llm-url", default="http://localhost:1234/v1")
    parser.add_argument("--ref-audio", default="reference.wav")
    parser.add_argument("--demo", action="store_true", help="Run without loading models")
    args = parser.parse_args()
    
    backend = None
    if not args.demo:
        # Load models BEFORE starting TUI to avoid multiprocessing conflicts
        print("\n\033[2mLoading models on Metal GPU...\033[0m")
        print("\033[2m(This may take a moment for first download)\033[0m\n")
        
        try:
            backend = OpenVoiceBackend(args)
            backend.load_models()
            print("\n\033[92m✓ Models loaded! Starting TUI...\033[0m\n")
        except Exception as e:
            print(f"\n\033[91mError loading models: {e}\033[0m")
            print("\033[93mStarting in demo mode...\033[0m\n")
            backend = None
    
    app = OpenVoiceTUI(args, backend=backend)
    app.run()


if __name__ == "__main__":
    main()