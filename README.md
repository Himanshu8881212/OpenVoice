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
- **LM Studio (Required)** - You must have [LM Studio](https://lmstudio.ai/) running with a local server enabled on localhost:1234. This provides the LLM reasoning capability.

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

You can manually change the models by passing arguments when starting the application:

```bash
# Example: Use a different STT or TTS model
python tui.py --stt-model "mlx-community/whisper-large-v3-turbo" --tts-model "mlx-community/chatterbox-turbo-fp16"
```

### Command Line Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--stt-model` | `mlx-community/parakeet-tdt-0.6b-v3` | MLX-optimized ASR model for voice recognition |
| `--tts-model` | `mlx-community/chatterbox-turbo-fp16` | MLX-optimized TTS model for voice output |
| `--llm-url` | `http://localhost:1234/v1` | URL for the LLM API server |
| `--ref-audio` | `reference.wav` | Reference audio file for voice cloning |
| `--demo` | - | Run without loading models (UI-only mode) |

## License

MIT
