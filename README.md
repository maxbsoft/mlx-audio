# MLX-Audio

A text-to-speech (TTS) and Speech-to-Speech (STS) library built on Apple's MLX framework, providing efficient speech synthesis on Apple Silicon with streaming capabilities.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Real-Time Streaming TTS](#real-time-streaming-tts)
  - [Streaming Features](#streaming-features)
  - [Basic Streaming Example](#basic-streaming-example)
  - [Callback-Based Streaming](#callback-based-streaming)
  - [WebSocket Streaming Server](#websocket-streaming-server)
  - [Streaming Parameters](#streaming-parameters)
  - [Audio Quality Modes](#audio-quality-modes)
  - [Quality Comparison Tool](#quality-comparison-tool)
  - [Known Quality Considerations](#known-quality-considerations)
  - [Best Practices](#best-practices)
- [Web Interface & API Server](#web-interface--api-server)
- [Models](#models)
- [Advanced Features](#advanced-features)
- [Requirements](#requirements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Fast inference on Apple Silicon (M series chips)
- Multiple language support
- Voice customization options
- Adjustable speech speed control (0.5x to 2.0x)
- **WebSocket streaming server**
- Interactive web interface with 3D audio visualization
- REST API for TTS generation
- Quantization support for optimized performance
- Direct access to output files via Finder/Explorer integration

## Installation

```bash
# Install the package
pip install mlx-audio

# For web interface and API dependencies
pip install -r requirements.txt
```

### Quick Start

To generate audio with an LLM use:

```bash
# Basic usage
mlx_audio.tts.generate --text "Hello, world"

# Specify prefix for output file
mlx_audio.tts.generate --text "Hello, world" --file_prefix hello

# Adjust speaking speed (0.5-2.0)
mlx_audio.tts.generate --text "Hello, world" --speed 1.4

# Real-time streaming TTS (see Streaming section for details)
mlx_audio.tts.generate --text "Hello, streaming world" --real_streaming --ultra_low_latency
```

### How to call from python

To generate audio with an LLM use:

```python
from mlx_audio.tts.generate import generate_audio

# Example: Generate an audiobook chapter as mp3 audio
generate_audio(
    text=("In the beginning, the universe was created...\n"
        "...or the simulation was booted up."),
    model_path="prince-canuma/Kokoro-82M",
    voice="af_heart",
    speed=1.2,
    lang_code="a", # Kokoro: (a)f_heart, or comment out for auto
    file_prefix="audiobook_chapter1",
    audio_format="wav",
    sample_rate=24000,
    join_audio=True,
    verbose=True  # Set to False to disable print messages
)

print("Audiobook chapter successfully generated!")

```

## Real-Time Streaming TTS

MLX-Audio supports real-time streaming TTS for applications requiring low-latency audio generation, such as conversational AI, live broadcasts, and interactive voice applications.

### Streaming Features

- **Ultra-low latency**: First audio chunk in ~0.25s vs ~6s for traditional generation
- **Real-time processing**: Audio streams as chunks without waiting for complete generation
- **Memory efficient**: No accumulation of large audio buffers
- **WebSocket support**: Perfect for real-time web applications
- **Configurable chunk sizes**: Control latency vs quality trade-offs
- **Two quality modes**: Clean (identical to normal generation) vs Processed (smoother transitions)

### Basic Streaming Example

```python
from mlx_audio.tts.generate import generate_audio_streaming

# Stream audio in real-time
for audio_chunk in generate_audio_streaming(
    text="Hello, this is streaming TTS with ultra-low latency!",
    model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
    voice="af_heart",
    streaming_chunk_tokens=35,  # ~150ms audio chunks
    temperature=0.6,
    ultra_low_latency=True  # Enable aggressive first chunk
):
    # Process each chunk as it arrives
    # - Send to audio device for immediate playback
    # - Stream over network via WebSocket
    # - Buffer for smooth playback
    print(f"Received chunk: {len(audio_chunk)} samples")
```

### Callback-Based Streaming

```python
from mlx_audio.tts.generate import generate_audio_with_callback

def audio_callback(chunk, metadata):
    """Called for each audio chunk"""
    print(f"Chunk {metadata['chunk_index']}: {len(chunk)} samples")
    # Send to audio device, WebSocket, etc.

generate_audio_with_callback(
    text="Your text here",
    callback=audio_callback,
    model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
    voice="af_heart",
    streaming_chunk_tokens=35,
    output_chunk_duration_ms=150,  # Target chunk duration
    ultra_low_latency=True
)
```

### WebSocket Streaming Server

MLX-Audio includes a WebSocket server for real-time streaming applications:

```bash
# Start the streaming server with model preloading (recommended)
python -m mlx_audio.orpheus_streaming_server --host 0.0.0.0 --port 8765

# With custom model and settings
python -m mlx_audio.orpheus_streaming_server \
    --host 127.0.0.1 \
    --port 8765 \
    --model mlx-community/orpheus-3b-0.1-ft-4bit

# Disable model preloading (load on first request)
python -m mlx_audio.orpheus_streaming_server \
    --host 0.0.0.0 \
    --port 8765 \
    --no-preload

# With verbose logging
python -m mlx_audio.orpheus_streaming_server \
    --host 0.0.0.0 \
    --port 8765 \
    --verbose
```

##### Quick local setup (server + browser client)

```bash
# 1) Start the server (local only)
python -m mlx_audio.orpheus_streaming_server --host 127.0.0.1 --port 8765 --verbose

# 2) In another terminal, serve the repo root for the browser client (AudioWorklet requires HTTP)
python3 -m http.server 8000

# 3) Open in your browser
# http://127.0.0.1:8000/websocket_test_client.html
```

#### Run the included browser client

The repository contains a minimal browser client that plays the stream via an AudioWorklet and expects binary PCM frames.

1) From the repository root, start a static HTTP server (required for AudioWorklet):

```bash
python3 -m http.server 8000
```

2) Open the client in your browser:

```
http://127.0.0.1:8000/websocket_test_client.html
```

3) Click “Connect”, then “Start”. Adjust “Playback Buffer (sec)” if you observe underflows; 0.15–0.25 is a good starting point.

Notes:
- AudioWorklets cannot be loaded from file:// URLs; serving over HTTP/HTTPS is required.
- The WebSocket server streams binary Float32 PCM frames (24 kHz, mono) with a small header: `b'PCM0' | u32 sample_rate | u32 num_samples | u32 num_channels | payload`.
- Control messages (`stream_started`, `stream_complete`, errors) are JSON.

#### Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Server host address | localhost |
| `--port` | Server port | 8765 |
| `--model` | Model path to preload | mlx-community/orpheus-3b-0.1-ft-4bit |
| `--no-preload` | Disable model preloading | False (preload enabled) |
| `--verbose` | Enable verbose logging | False |

Protocol details:
- Audio is sent as binary PCM frames (Float32, 24 kHz, mono) with a `PCM0` header. Clients should treat non‑text WebSocket messages as audio frames and parse the header before reading the Float32 payload.
- Control/status messages are JSON text frames.

**Model Preloading Benefits:**
- ⚡ Faster first response (~0.3s vs ~6s)
- 🛡️ Prevents segmentation faults from concurrent model loading
- 💾 Single model instance in memory for all clients
- 🚀 Ready to serve immediately upon connection
- 🎵 SNAC audio codec also preloaded (no download delay)
- 🔄 Thread-safe model sharing across all client sessions

#### WebSocket Client Example

Note: The server currently uses binary PCM frames. The legacy example below demonstrates a base64 text protocol and is kept for reference; for browsers, prefer the included `websocket_test_client.html` which handles binary frames and AudioWorklet playback.

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = function() {
    // Send text for streaming with emotions
    ws.send(JSON.stringify({
        type: 'start_orpheus_stream',
        text: "Hello! <chuckle> This is real-time streaming TTS with emotions. Amazing, right? <gasp>",
        voice: "tara",
        temperature: 0.6,
        chunk_tokens: 35,
        ultra_low_latency: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'audio_chunk') {
        // Received audio chunk as base64
        const audioBuffer = base64ToArrayBuffer(data.audio);
        // Play audio immediately
        playAudioChunk(audioBuffer);
    }
};
```

### Streaming Parameters

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|---------|
| `streaming_chunk_tokens` | Tokens per chunk (must be multiple of 7) | 21 | Smaller = lower latency, more chunks |
| `ultra_low_latency` | Aggressive first chunk generation | True | Reduces first chunk time to ~0.25s |
| `temperature` | Generation randomness | 0.6 | Higher = more variation |
| `output_chunk_duration_ms` | Target output chunk duration | 150ms | Controls buffering behavior |
| `respect_source_boundaries` | Preserve model chunk boundaries | False | Reduces audio artifacts |

### Available Voices

| Voice | Gender | Description |
|-------|--------|-------------|
| `tara` | Female | Natural, warm voice (default) |
| `leah` | Female | Clear, professional voice |
| `jess` | Female | Friendly, energetic voice |
| `leo` | Male | Deep, authoritative voice |
| `dan` | Male | Casual, approachable voice |
| `mia` | Female | Soft, expressive voice |
| `zac` | Male | Young, dynamic voice |
| `zoe` | Female | Bright, cheerful voice |

### Emotion Tags

Add emotional expressions to your text using these tags:

| Emotion | Tag | Usage Example |
|---------|-----|---------------|
| Laugh | `<laugh>` | "That's hilarious! `<laugh>`" |
| Chuckle | `<chuckle>` | "Well, `<chuckle>` that's interesting." |
| Sigh | `<sigh>` | "I suppose so. `<sigh>`" |
| Cough | `<cough>` | "Excuse me. `<cough>` As I was saying..." |
| Sniffle | `<sniffle>` | "I'm a bit emotional. `<sniffle>`" |
| Groan | `<groan>` | "Oh no, not again! `<groan>`" |
| Yawn | `<yawn>` | "I'm getting tired. `<yawn>`" |
| Gasp | `<gasp>` | "Wow! `<gasp>` That's amazing!" |

**💡 Tip**: Place emotion tags where you want the expression to occur in the speech. The emotion will be rendered at that specific point in the audio.

### Audio Quality Modes

#### Clean Mode (Recommended for Quality)
```python
# Generate without audio processing (identical to normal generation)
for chunk in generate_audio_streaming(text, ...):
    chunks.append(chunk.astype(np.float32))
total_audio = np.concatenate(chunks)  # Simple concatenation
```

#### Processed Mode (Smoother Transitions)
```python
from mlx_audio.tts.audio_utils import StreamingAudioProcessor

# Generate with crossfading and smoothing
processor = StreamingAudioProcessor(crossfade_ms=10.0)
for chunk in generate_audio_streaming(text, ...):
    processed_chunk = processor.process_chunk(chunk)
total_audio = processor.get_concatenated_audio()
```

### Quality Comparison Tool

Use the included comparison tool to evaluate audio quality across different generation methods:

```bash
# Generate all variants for comparison with consistent voice
python simple_streaming_example.py

# This will generate:
# - consistent_normal_000.wav (baseline quality - traditional generation)
# - consistent_clean_streaming.wav (streaming without processing)
# - consistent_processed_streaming.wav (streaming with crossfading)
# - consistent_callback_streaming.wav (callback-based streaming)
```

#### What the Comparison Tool Tests

The `simple_streaming_example.py` script generates audio using the same voice (`af_heart`) and text across all methods to provide fair quality comparison:

1. **Normal Generation**: Traditional non-streaming TTS (baseline quality)
2. **Clean Streaming**: Raw chunk concatenation without processing (should match baseline)
3. **Processed Streaming**: Crossfading and smoothing applied (smoother but potentially muffled)
4. **Callback Streaming**: Event-driven streaming (useful for real-time applications)

All tests use consistent parameters:
- Model: `mlx-community/orpheus-3b-0.1-ft-4bit`
- Voice: `af_heart` 
- Chunk size: 35 tokens (~150ms audio)
- Temperature: 0.6

Listen to all generated files to understand the quality trade-offs for your specific use case.

### Known Quality Considerations

#### Audio Quality Comparison

The streaming implementation provides acceptable quality with some trade-offs compared to traditional generation:

1. **Clean Streaming (Unprocessed)**:
   - ✅ **Quality**: Identical to normal generation
   - ✅ **Clarity**: Full voice dynamics preserved
   - ⚠️ **Artifacts**: May have rare audio clicks between chunks during concatenation
   - 🎯 **Use Case**: Maximum quality applications where occasional clicks are acceptable

2. **Processed Streaming (With Audio Processing)**:
   - ✅ **Smoothness**: Eliminates clicks with crossfading
   - ✅ **Continuity**: Seamless audio transitions
   - ⚠️ **Quality**: Slightly muffled/softer voice due to crossfading and smoothing
   - ⚠️ **Dynamics**: Reduced dynamic range from processing
   - 🎯 **Use Case**: Real-time applications requiring smooth playback

3. **Latency vs Quality Trade-offs**:
   - Smaller chunk sizes (7-14 tokens) = Lower latency but more potential artifacts
   - Larger chunk sizes (21-35 tokens) = Better quality but higher latency
   - Ultra-low latency mode = ~0.25s first chunk but may affect initial quality

4. **Model Compatibility**:
   - Optimized for Orpheus models (`mlx-community/orpheus-3b-0.1-ft-4bit`)
   - Other models may have different streaming behavior and quality characteristics
   - Kokoro and other models may require different parameter tuning

### Best Practices

#### Quality Optimization
1. **For Maximum Quality**: Use clean streaming without processing
   ```python
   # Simple concatenation preserves original quality
   chunks = [chunk.astype(np.float32) for chunk in generate_audio_streaming(...)]
   audio = np.concatenate(chunks)
   ```

2. **For Smooth Playback**: Use minimal crossfading (1-3ms) to reduce artifacts
   ```python
   processor = StreamingAudioProcessor(crossfade_ms=3.0)  # Minimal processing
   ```

3. **For Real-time Applications**: 
   - Start with `streaming_chunk_tokens=35` (good balance)
   - Reduce to 21 or 14 for lower latency if needed
   - Use `ultra_low_latency=True` for conversational AI

#### Performance Tuning
4. **WebSocket Streaming**: Enable `ultra_low_latency=True` for responsive user experience
5. **Memory Efficiency**: Use callback-based streaming for long texts
6. **Network Streaming**: Consider chunk buffering to handle network jitter

#### Testing and Development
7. **Quality Testing**: Always compare with normal generation using same voice and text
8. **Use Comparison Tool**: Run `python simple_streaming_example.py` to evaluate quality
9. **Voice Consistency**: Use same voice across all tests for fair comparison
10. **Parameter Experimentation**: Test different `streaming_chunk_tokens` values for your use case

#### Production Considerations
11. **Error Handling**: Implement proper error handling for network interruptions
12. **Audio Buffering**: Buffer 2-3 chunks for smooth playback in real-time applications
13. **Quality Monitoring**: Monitor for audio artifacts in production deployments
14. **Model Selection**: Use Orpheus models for best streaming performance

## Web Interface & API Server

MLX-Audio includes a web interface with a 3D visualization that reacts to audio frequencies. The interface allows you to:

1. Generate TTS with different voices and speed settings
2. Upload and play your own audio files
3. Visualize audio with an interactive 3D orb
4. Automatically saves generated audio files to the outputs directory in the current working folder
5. Open the output folder directly from the interface (when running locally)

#### Features

- **Multiple Voice Options**: Choose from different voice styles (AF Heart, AF Nova, AF Bella, BF Emma)
- **Adjustable Speech Speed**: Control the speed of speech generation with an interactive slider (0.5x to 2.0x)
- **Real-time 3D Visualization**: A responsive 3D orb that reacts to audio frequencies
- **Audio Upload**: Play and visualize your own audio files
- **Auto-play Option**: Automatically play generated audio
- **Output Folder Access**: Convenient button to open the output folder in your system's file explorer

To start the web interface and API server:

```bash
# Using the command-line interface
mlx_audio.server

# With custom host and port
mlx_audio.server --host 0.0.0.0 --port 9000

# With verbose logging
mlx_audio.server --verbose
```

Available command line arguments:
- `--host`: Host address to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 8000)

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

#### API Endpoints

The server provides the following REST API endpoints:

- `POST /tts`: Generate TTS audio
  - Parameters (form data):
    - `text`: The text to convert to speech (required)
    - `voice`: Voice to use (default: "af_heart")
    - `speed`: Speech speed from 0.5 to 2.0 (default: 1.0)
  - Returns: JSON with filename of generated audio

- `GET /audio/{filename}`: Retrieve generated audio file

- `POST /play`: Play audio directly from the server
  - Parameters (form data):
    - `filename`: The filename of the audio to play (required)
  - Returns: JSON with status and filename

- `POST /stop`: Stop any currently playing audio
  - Returns: JSON with status

- `POST /open_output_folder`: Open the output folder in the system's file explorer
  - Returns: JSON with status and path
  - Note: This feature only works when running the server locally

> Note: Generated audio files are stored in `~/.mlx_audio/outputs` by default, or in a fallback directory if that location is not writable.

## Models

### Kokoro

Kokoro is a multilingual TTS model that supports various languages and voice styles.

#### Example Usage

```python
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
from IPython.display import Audio
import soundfile as sf

# Initialize the model
model_id = 'prince-canuma/Kokoro-82M'
model = load_model(model_id)

# Create a pipeline with American English
pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)

# Generate audio
text = "The MLX King lives. Let him cook!"
for _, _, audio in pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'):
    # Display audio in notebook (if applicable)
    display(Audio(data=audio, rate=24000, autoplay=0))

    # Save audio to file
    sf.write('audio.wav', audio[0], 24000)
```

#### Language Options

- 🇺🇸 `'a'` - American English
- 🇬🇧 `'b'` - British English
- 🇯🇵 `'j'` - Japanese (requires `pip install misaki[ja]`)
- 🇨🇳 `'z'` - Mandarin Chinese (requires `pip install misaki[zh]`)

### CSM (Conversational Speech Model)

CSM is a model from Sesame that allows you text-to-speech and to customize voices using reference audio samples.

#### Example Usage

```bash
# Generate speech using CSM-1B model with reference audio
python -m mlx_audio.tts.generate --model mlx-community/csm-1b --text "Hello from Sesame." --play --ref_audio ./conversational_a.wav
```

You can pass any audio to clone the voice from or download sample audio file from [here](https://huggingface.co/mlx-community/csm-1b/tree/main/prompts).

## Advanced Features

### Quantization

You can quantize models for improved performance:

```python
from mlx_audio.tts.utils import quantize_model, load_model
import json
import mlx.core as mx

model = load_model(repo_id='prince-canuma/Kokoro-82M')
config = model.config

# Quantize to 8-bit
group_size = 64
bits = 8
weights, config = quantize_model(model, config, group_size, bits)

# Save quantized model
with open('./8bit/config.json', 'w') as f:
    json.dump(config, f)

mx.save_safetensors("./8bit/kokoro-v1_0.safetensors", weights, metadata={"format": "mlx"})
```

## Requirements

- MLX
- Python 3.8+
- Apple Silicon Mac (for optimal performance)
- For streaming functionality:
  - soundfile (for audio I/O)
  - numpy (for audio processing)
- For the web interface and API:
  - FastAPI
  - Uvicorn
- For WebSocket streaming:
  - websockets
  
## License

[MIT License](LICENSE)

## Acknowledgements

- Thanks to the Apple MLX team for providing a great framework for building TTS and STS models.
- This project uses the Kokoro and Orpheus model architectures for text-to-speech synthesis.
- The 3D visualization uses Three.js for rendering.
- Streaming TTS implementation optimized for real-time applications and conversational AI.
- Special thanks to the open-source community for feedback on audio quality and streaming performance.
