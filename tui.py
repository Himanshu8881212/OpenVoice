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
from textual.containers import Container, ScrollableContainer
from textual.widgets import Static, Input
from textual.reactive import reactive
from textual import work
from rich.text import Text

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
# CSS Styles
# ═══════════════════════════════════════════════════════════════════════════════
CSS = """
Screen {
    background: #0d1117;
}

#app-container {
    height: 100%;
    width: 100%;
    layout: vertical;
}

#header-banner {
    height: 8;
    content-align: center middle;
    background: #161b22;
    color: #c9a0ff;
}

#chat-container {
    height: 1fr;
    border: solid #30363d;
    padding: 1;
    overflow-y: auto;
    background: #0d1117;
}

.user-message {
    margin-bottom: 1;
    color: #58a6ff;
}

.ai-message {
    margin-bottom: 1;
    color: #c9a0ff;
}

.system-message {
    margin-bottom: 1;
    color: #8b949e;
}

#input-container {
    height: auto;
    min-height: 4;
    padding: 1 2;
    background: #161b22;
    border-top: solid #30363d;
}

#user-input {
    width: 100%;
    height: 3;
    background: #21262d;
    color: #c9d1d9;
    border: solid #30363d;
}

#status-bar {
    height: 2;
    background: #161b22;
    color: #8b949e;
    padding: 0 1;
    content-align: center middle;
    text-align: center;
}

.status-listening {
    background: #238636;
    color: #ffffff;
}

.status-camera {
    background: #9e6a03;
    color: #ffffff;
}
"""

OPENVOICE_BANNER = """
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║   ██║██╔═══██╗██║██╔════╝██╔════╝
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║   ██║██║   ██║██║██║     █████╗  
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  
╚██████╔╝██║     ███████╗██║ ╚████║ ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Textual Widgets
# ═══════════════════════════════════════════════════════════════════════════════
class ChatMessage(Static):
    """A single chat message widget."""
    
    def __init__(self, sender: str, content: str, msg_type: str = "user"):
        super().__init__()
        self.sender = sender
        self.content = content
        self.msg_type = msg_type
        self.add_class(f"{msg_type}-message")
    
    def compose(self) -> ComposeResult:
        text = Text()
        if self.msg_type == "user":
            text.append(f"{self.sender}: ", style="bold #58a6ff")
        elif self.msg_type == "ai":
            text.append(f"{self.sender}: ", style="bold #c9a0ff")
        else:
            text.append(f"{self.sender}: ", style="dim")
        text.append(self.content)
        yield Static(text)
    
    def update_content(self, new_content: str):
        """Update message content for streaming."""
        self.content = new_content
        try:
            static = self.query_one(Static)
            text = Text()
            if self.msg_type == "user":
                text.append(f"{self.sender}: ", style="bold #58a6ff")
            elif self.msg_type == "ai":
                text.append(f"{self.sender}: ", style="bold #c9a0ff")
            else:
                text.append(f"{self.sender}: ", style="dim")
            text.append(new_content)
            static.update(text)
        except Exception:
            pass


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
class OpenVoiceBackend:
    """Backend handling STT, TTS, LLM, and recording."""
    
    def __init__(self, args):
        self.args = args
        self.chat_history = []
        
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
                for res in self.tts.generate(text=text, ref_audio=self.ref_audio, verbose=False, stream=True):
                    if hasattr(res, 'audio') and res.audio is not None:
                        mx.eval(res.audio)
                        self.player.queue_audio(res.audio)
            except Exception:
                pass
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        with self.lock:
            if self.is_recording or self.is_recording_video:
                self.audio_chunks.append(indata.copy())
    
    def speak(self, text):
        """Queue text for TTS."""
        if text and text.strip():
            self.tts_queue.put(text)
    
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
        
        return response
    
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
# Textual TUI Application
# ═══════════════════════════════════════════════════════════════════════════════
class OpenVoiceTUI(App):
    """OpenVoice Voice Assistant TUI."""
    
    CSS = CSS
    
    BINDINGS = [
        Binding("escape", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]
    
    def __init__(self, args, backend=None):
        super().__init__()
        self.args = args
        self.backend = backend  # Pre-loaded backend
        self._chat_container = None
        self._current_ai_message = None
        self._pynput_listener = None
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        with Container(id="app-container"):
            yield Static(OPENVOICE_BANNER, id="header-banner")
            yield ScrollableContainer(id="chat-container")
            with Container(id="input-container"):
                yield Input(placeholder="Type message and press Enter...", id="user-input")
            yield StatusBar(id="status-bar")
    
    def on_mount(self):
        """Called when app is mounted."""
        self._chat_container = self.query_one("#chat-container", ScrollableContainer)
        self.query_one("#user-input", Input).focus()
        
        # Start pynput listener for PTT (backend already loaded)
        self._start_pynput_listener()
    
    def _start_pynput_listener(self):
        """Start pynput keyboard listener for PTT."""
        if not HAS_PYNPUT:
            self._add_system_message("pynput not available - use text commands: 'r'=record, 'v'=video")
            return
        
        def on_press(key):
            try:
                if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    if self.backend and self.backend.start_recording():
                        self.call_from_thread(self._on_recording_start)
                elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    if self.backend and self.backend.start_video():
                        self.call_from_thread(self._on_video_start)
            except Exception:
                pass
        
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
        if not user_input:
            return
        
        event.input.value = ""
        
        if user_input.lower() == "quit":
            self.exit()
            return
        
        # Manual record command
        if user_input.lower() in ('r', 'record'):
            self._add_system_message("Use OPTION key to record, or type your message")
            return
        
        # Manual video command
        if user_input.lower() in ('v', 'video'):
            self._add_system_message("Use CTRL key to record video, or type your message")
            return
        
        # Regular text input
        self._add_user_message(user_input)
        self._process_text_input(user_input)
    
    @work(thread=True)
    def _process_text_input(self, text: str):
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
        
        self.backend.chat(text, on_token=on_token)
        self.call_from_thread(self._finish_ai_response)
    
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