# Local Answer Assistant

A lightweight desktop prototype, inspired by Parakeet AI, that listens to your microphone, transcribes speech in real time, and generates a suggested answer to the question it heard. **Speech-to-text and the LLM both run entirely on your machine.** There are no API keys, no cloud calls, and no usage costs.

## Features

- **Simple desktop UI** with Start, Stop and Suggest Answer buttons
- **Live microphone transcription** using a local Whisper model
- **One-click answer suggestions** from a local LLM
- **Progressive output**: the answer streams into the window token by token as it is generated
- **Fully offline** after the one-time model downloads
- Works on **Windows and macOS**

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Python, Tkinter |
| Audio capture | `sounddevice`, `numpy` |
| Speech-to-text | `faster-whisper` (default model: `base.en`) |
| LLM runtime | [Ollama](https://ollama.com) (default model: `llama3.2:3b`) |
| Streaming | Ollama HTTP API via `requests` |

## How It Works

```
Microphone ──► audio blocks (100 ms) ──► silence-based utterance splitter
                                                   │
                                                   ▼
                                       faster-whisper (local STT)
                                                   │
                                                   ▼
                                          transcript pane (UI)
                                                   │   click "Suggest Answer"
                                                   ▼
                                    Ollama chat API (streaming, local)
                                                   │
                                                   ▼
                                  answer pane, updated token by token
```

1. The microphone callback pushes small audio blocks onto a queue.
2. A background worker groups them into utterances, ending one after about 0.8 s of silence (or 15 s of continuous speech), and transcribes each with Whisper.
3. When you click **Suggest Answer**, the most recent part of the transcript is sent to Ollama with a prompt asking it to answer the latest question.
4. Tokens stream back and are appended to the answer pane as they arrive.
5. Worker threads never touch widgets directly; they post messages to a queue that the UI thread polls, so the window stays responsive.

## Prerequisites

- **Python 3.10 – 3.12** (3.12 recommended). Very new Python releases may lack prebuilt wheels for dependencies such as `av`, which causes install failures.
- **Ollama**, installed from [ollama.com](https://ollama.com)
- A working microphone
- Around 3 GB of free disk space for the models and 8 GB of RAM or more recommended

## Setup

### 1. Install Ollama and pull the model

```bash
ollama pull llama3.2:3b
```

Ollama normally runs in the background after installation. If it does not, start it with `ollama serve`.

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

On the first launch, the Whisper model (about 150 MB) is downloaded once. Wait until the status changes from "Loading speech model..." to "Ready". After that, no internet connection is needed.

## Usage

1. Click **Start** and speak a question, for example: *"What is the difference between a list and a tuple in Python?"*
2. Pause briefly. The transcription appears in the top pane.
3. Click **Stop** when you are done.
4. Click **Suggest Answer**. The answer streams into the bottom pane as it is generated.

## Configuration

Edit the constants at the top of `app.py`:

| Setting | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | Use `tiny.en` for speed, `small.en` for accuracy |
| `OLLAMA_MODEL` | `llama3.2:3b` | Any model you have pulled with `ollama pull` |
| `SILENCE_RMS` | `0.01` | Raise it in noisy rooms, lower it for quiet microphones |
| `END_OF_SPEECH_BLOCKS` | `8` | Silence length (in 100 ms blocks) that ends an utterance |
| `MAX_CONTEXT_CHARS` | `3000` | How much transcript is sent to the LLM |
| `SYSTEM_PROMPT` | interview assistant | Controls the style of the suggested answer |

On a slower machine, switch to lighter models: set `WHISPER_MODEL = "tiny.en"` and run `ollama pull llama3.2:1b`, then set `OLLAMA_MODEL = "llama3.2:1b"`.

## Limitations

- **Microphone only.** It does not capture system audio. To transcribe a call or video, play it through speakers within range of the mic, or use a loopback device (BlackHole on macOS, Stereo Mix on Windows).
- **Utterance-level transcription.** Text appears after each pause rather than word by word, which keeps the implementation simple and accurate.
- **English only** with the default models.
- **Answer quality** depends on the local model size and on transcription accuracy.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails while building `av` | Use Python 3.12 instead of a newer release |
| "Can't reach Ollama" | Start the Ollama app or run `ollama serve` |
| `ollama` command not found | Restart the terminal or VS Code after installing Ollama |
| No transcription appears | Check the OS microphone permission and input device; lower `SILENCE_RMS` |
| macOS: `No module named tkinter` | Install Python from python.org or run `brew install python-tk` |
| Slow transcription or answers | Use `tiny.en` and `llama3.2:1b` |

## Project Structure

```
.
├── app.py              # UI, audio capture, transcription, LLM streaming
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Possible Improvements

- Capture system or loopback audio alongside the microphone
- Word-level live transcription with partial results
- Automatic question detection instead of a manual button
- Model and microphone pickers in the UI
- Stop and regenerate controls for answers
