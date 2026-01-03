# OpenVoice

A Mac-optimized voice assistant with a modern Textual TUI.

## Features

- 🎤 **Push-to-Talk** - Hold OPTION key to record voice
- 📷 **Push-to-Record** - Hold CTRL key to record video + audio
- ⌨️ **Text Chat** - Type messages and press Enter
- 🔊 **Streaming Audio** - Text and audio play simultaneously
- 🖥️ **Beautiful TUI** - Modern terminal interface with Textual

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- LM Studio or compatible LLM server running on localhost:1234

## Installation

```bash
# Clone the repository
git clone https://github.com/Himanshu8881212/OpenVoice.git
cd OpenVoice

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the TUI (requires LM Studio running)
python tui.py

# Run in demo mode (without models)
python tui.py --demo
```

## Controls

| Key | Action |
|-----|--------|
| **OPTION** (hold) | Voice recording - release to send |
| **CTRL** (hold) | Video + voice recording - release to send |
| **Type + Enter** | Send text message |
| **Esc** | Quit |

## Configuration

```bash
python tui.py --help

Options:
  --stt-model    STT model (default: mlx-community/parakeet-tdt-0.6b-v3)
  --tts-model    TTS model (default: mlx-community/chatterbox-turbo-fp16)
  --llm-url      LLM API URL (default: http://localhost:1234/v1)
  --ref-audio    Reference audio for voice cloning (default: reference.wav)
  --demo         Run without loading models
```

## License

MIT
