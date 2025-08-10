import argparse
import os
import random
import sys
import time
from typing import Optional, Tuple, Generator, Callable

import mlx.core as mx
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample

from .audio_player import AudioPlayer
from .utils import load_model


def load_audio(
    audio_path: str,
    sample_rate: int = 24000,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
) -> mx.array:
    samples, orig_sample_rate = sf.read(audio_path)
    shape = samples.shape

    # Collapse multi channel as mono
    if len(shape) > 1:
        samples = samples.sum(axis=1)
        # Divide summed samples by channel count.
        samples = samples / shape[1]
    if sample_rate != orig_sample_rate:
        print(f"Resampling from {orig_sample_rate} to {sample_rate}")
        duration = samples.shape[0] / orig_sample_rate
        num_samples = int(duration * sample_rate)
        samples = resample(samples, num_samples)

    if segment_duration is not None:
        seg_length = int(sample_rate * segment_duration)
        samples = random_select_audio_segment(samples, seg_length)

    # Audio volume normalize
    if volume_normalize:
        samples = audio_volume_normalize(samples)

    if length is not None:
        assert abs(samples.shape[0] - length) < 1000
        if samples.shape[0] > length:
            samples = samples[:length]
        else:
            samples = np.pad(samples, (0, int(length - samples.shape[0])))

    audio = mx.array(samples, dtype=mx.float32)

    return audio


def audio_volume_normalize(audio: np.ndarray, coeff: float = 0.2) -> np.ndarray:
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def random_select_audio_segment(audio: np.ndarray, length: int) -> np.ndarray:
    """get an audio segment given the length

    Args:
        audio (np.ndarray):
        length (int): audio length = sampling_rate * duration
    """
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]


def detect_speech_boundaries(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = 0.1,
    energy_threshold: float = 0.01,
    margin_factor: int = 2,
) -> Tuple[int, int]:
    """Detect the start and end points of speech in an audio signal using RMS energy.

    Args:
        wav: Input audio signal array with values in [-1, 1]
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        energy_threshold: RMS energy threshold for speech detection
        margin_factor: Factor to determine extra margin around detected boundaries

    Returns:
        tuple: (start_index, end_index) of speech segment

    Raises:
        ValueError: If the audio contains only silence
    """
    window_size = int(window_duration * sample_rate)
    margin = margin_factor * window_size
    step_size = window_size // 10

    # Create sliding windows using stride tricks to avoid loops
    windows = sliding_window_view(wav, window_size)[::step_size]

    # Calculate RMS energy for each window
    energy = np.sqrt(np.mean(windows**2, axis=1))
    speech_mask = energy >= energy_threshold

    if not np.any(speech_mask):
        raise ValueError("No speech detected in audio (only silence)")

    start = max(0, np.argmax(speech_mask) * step_size - margin)
    end = min(
        len(wav),
        (len(speech_mask) - 1 - np.argmax(speech_mask[::-1])) * step_size + margin,
    )

    return start, end


def remove_silence_on_both_ends(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = 0.1,
    volume_threshold: float = 0.01,
) -> np.ndarray:
    """Remove silence from both ends of an audio signal.

    Args:
        wav: Input audio signal array
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        volume_threshold: Amplitude threshold for silence detection

    Returns:
        np.ndarray: Audio signal with silence removed from both ends

    Raises:
        ValueError: If the audio contains only silence
    """
    start, end = detect_speech_boundaries(
        wav, sample_rate, window_duration, volume_threshold
    )
    return wav[start:end]


def hertz_to_mel(pitch: float) -> float:
    """
    Converts a frequency from the Hertz scale to the Mel scale.

    Parameters:
    - pitch: float or ndarray
        Frequency in Hertz.

    Returns:
    - mel: float or ndarray
        Frequency in Mel scale.
    """
    mel = 2595 * np.log10(1 + pitch / 700)
    return mel


def generate_audio(
    text: str,
    model_path: str = "prince-canuma/Kokoro-82M",
    max_tokens: int = 1200,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    stt_model: str = "mlx-community/whisper-large-v3-turbo",
    file_prefix: str = "audio",
    audio_format: str = "wav",
    join_audio: bool = False,
    play: bool = False,
    verbose: bool = True,
    temperature: float = 0.7,
    stream: bool = False,
    streaming_interval: float = 2.0,
    **kwargs,
) -> None:
    """
    Generates audio from text using a specified TTS model.

    Parameters:
    - text (str): The input text to be converted to speech.
    - model (str): The TTS model to use.
    - voice (str): The voice style to use.
    - temperature (float): The temperature for the model.
    - speed (float): Playback speed multiplier.
    - lang_code (str): The language code.
    - ref_audio (mx.array): Reference audio you would like to clone the voice from.
    - ref_text (str): Caption for reference audio.
    - stt_model (str): A mlx whisper model to use to transcribe.
    - file_prefix (str): The output file path without extension.
    - audio_format (str): Output audio format (e.g., "wav", "flac").
    - join_audio (bool): Whether to join multiple audio files into one.
    - play (bool): Whether to play the generated audio.
    - verbose (bool): Whether to print status messages.
    Returns:
    - None: The function writes the generated audio to a file.
    """
    try:
        play = play or stream

        # Load model
        model = load_model(model_path=model_path)

        # Load reference audio for voice matching if specified
        if ref_audio:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

            normalize = False
            if hasattr(model, "model_type") and model.model_type() == "spark":
                normalize = True

            ref_audio = load_audio(
                ref_audio, sample_rate=model.sample_rate, volume_normalize=normalize
            )
            if not ref_text:
                print("Ref_text not found. Transcribing ref_audio...")
                from mlx_audio.stt.models.whisper import Model as Whisper

                stt_model = Whisper.from_pretrained(path_or_hf_repo=stt_model)
                ref_text = stt_model.generate(ref_audio).text
                print("Ref_text", ref_text)

                # clear memory
                del stt_model
                mx.clear_cache()

        # Load AudioPlayer
        player = AudioPlayer(sample_rate=model.sample_rate) if play else None

        print(
            f"\n\033[94mModel:\033[0m {model_path}\n"
            f"\033[94mText:\033[0m {text}\n"
            f"\033[94mVoice:\033[0m {voice}\n"
            f"\033[94mSpeed:\033[0m {speed}x\n"
            f"\033[94mLanguage:\033[0m {lang_code}"
        )

        results = model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            stream=stream,
            streaming_interval=streaming_interval,
            **kwargs,
        )

        audio_list = []
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)

            if join_audio:
                audio_list.append(result.audio)
            elif not stream:
                file_name = f"{file_prefix}_{i:03d}.{audio_format}"
                sf.write(file_name, result.audio, result.sample_rate)
                print(f"✅ Audio successfully generated and saving as: {file_name}")

            if verbose:

                print("==========")
                print(f"Duration:              {result.audio_duration}")
                print(
                    f"Samples/sec:           {result.audio_samples['samples-per-sec']:.1f}"
                )
                print(
                    f"Prompt:                {result.token_count} tokens, {result.prompt['tokens-per-sec']:.1f} tokens-per-sec"
                )
                print(
                    f"Audio:                 {result.audio_samples['samples']} samples, {result.audio_samples['samples-per-sec']:.1f} samples-per-sec"
                )
                print(f"Real-time factor:      {result.real_time_factor:.2f}x")
                print(f"Processing time:       {result.processing_time_seconds:.2f}s")
                print(f"Peak memory usage:     {result.peak_memory_usage:.2f}GB")

        if join_audio and not stream:
            if verbose:
                print(f"Joining {len(audio_list)} audio files")
            audio = mx.concatenate(audio_list, axis=0)
            sf.write(
                f"{file_prefix}.{audio_format}",
                audio,
                model.sample_rate,
            )
            if verbose:
                print(f"✅ Audio successfully generated and saving as: {file_name}")

        if play:
            player.wait_for_drain()
            player.stop()

    except ImportError as e:
        print(f"Import error: {e}")
        print(
            "This might be due to incorrect Python path. Check your project structure."
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()


def generate_audio_streaming(
    text: str,
    model_path: str = "mlx-community/orpheus-3b-0.1-ft-4bit",
    model=None,  # Optional pre-loaded model to avoid reloading
    voice: str = "af_heart", 
    temperature: float = 0.6,
    top_p: float = 0.8,
    max_tokens: int = 1200,
    chunk_tokens: Optional[int] = 70,  # Tokens per chunk (must be multiple of 7)
    overlap_groups: int = 0,  # Number of groups (7 tokens each) to use as left context (0 = full context, best quality)
    lookahead_depth: int = 0,  # Number of tokens to look ahead for overlap-save (0 = immediate output)
    crossfade_samples: int = 0,  # Crossfade samples (0 = disabled, 480 = ~20ms at 24kHz)
    verbose: bool = True,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    **kwargs,
) -> Generator[np.ndarray, None, None]:
    """
    Generate audio with real streaming for Orpheus models
    
    Args:
        text: Text to synthesize
        model_path: Path to the TTS model (used if model=None)
        model: Optional pre-loaded model (avoids ~6s loading time)
        voice: Voice to use
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        chunk_tokens: Tokens per chunk (must be multiple of 7)
        overlap_groups: Number of groups (7 tokens each) to use as left context (0 = full context, best quality)
        lookahead_depth: Number of tokens to look ahead for overlap-save (0 = immediate output)
        crossfade_samples: Number of samples to crossfade between chunks (0 = disabled, 480 = ~20ms at 24kHz)
        verbose: Print progress information
        ref_audio: Path to reference audio for voice cloning
        ref_text: Text for reference audio
        **kwargs: Additional model parameters
        
    Yields:
        np.ndarray: Audio chunks as they are generated
    """
    
    try:
        # Handle parameter compatibility
        if chunk_tokens is not None:
            effective_chunk_tokens = chunk_tokens
        else:
            effective_chunk_tokens = 70  # Default value
        
        
        # Check if model is Orpheus-compatible
        if "orpheus" not in model_path.lower() and "llama" not in model_path.lower():
            print(f"Warning: Model {model_path} may not support streaming. "
                  f"Consider using 'mlx-community/orpheus-3b-0.1-ft-4bit'")
        
        # Load or reuse model
        if model is None:
            # Load model if not provided
            from .utils import load_model as base_load_model
            model = base_load_model(model_path=model_path)
            if verbose:
                print(f"📦 Model loaded from: {model_path}")
        else:
            # Use provided model
            if verbose:
                print(f"🔄 Using pre-loaded model")
        
        # Load reference audio if provided
        ref_audio_mx = None
        if ref_audio:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
            
            ref_audio_mx = load_audio(ref_audio, sample_rate=model.sample_rate)
            
            if not ref_text:
                print("Ref_text not found. Transcribing ref_audio...")
                from mlx_audio.stt.models.whisper import Model as Whisper
                
                stt_model = Whisper.from_pretrained(path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
                ref_text = stt_model.generate(ref_audio_mx).text
                print(f"Ref_text: {ref_text}")
                
                # Clear memory
                del stt_model
                mx.clear_cache()
        
        if verbose:
            print(f"\n🎵 Orpheus Streaming TTS")
            print(f"📝 Text: {text}")
            print(f"🎤 Voice: {voice}")
            print(f"📊 Chunk size: {effective_chunk_tokens} tokens")
            print(f"🔄 Overlap groups: {overlap_groups} {'(full context)' if overlap_groups == 0 else '(limited overlap)'}")
            print(f"👀 Lookahead depth: {lookahead_depth}")
            print(f"🔀 Crossfade samples: {crossfade_samples}")
            print(f"🌡️  Temperature: {temperature}")
            print(f"🎯 Model: {model_path}")
        
        # Choose streaming approach based on latency requirements
       
        if verbose:
            print("🚀 Using ultra-low latency mode (~0.27s first chunk)")
        from .ultra_low_latency_streaming import generate_ultra_low_latency_streaming
        
        for audio_chunk in generate_ultra_low_latency_streaming(
            model=model,
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            lookahead_depth=lookahead_depth,
            chunk_tokens=effective_chunk_tokens,
            overlap_groups=overlap_groups,
            crossfade_samples=crossfade_samples,
            verbose=verbose,
            ref_audio=ref_audio_mx,
            ref_text=ref_text,
            debug_save_wav=kwargs.get("debug_save_wav", True),  # Pass debug parameters
            debug_wav_path=kwargs.get("debug_wav_path", "debug_ultra_low_latency.wav"),
            **kwargs
        ):
            yield audio_chunk
    
                
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()


def generate_audio_with_callback(
    text: str,
    callback: Callable[[np.ndarray, dict], None],
    model_path: str = "mlx-community/orpheus-3b-0.1-ft-4bit",
    output_chunk_duration_ms: int = 100,
    ultra_low_latency: bool = True,
    **kwargs
) -> None:
    """
    Generate audio with callback function for each chunk
    
    Args:
        text: Text to synthesize
        callback: Function called with (audio_chunk, metadata) for each chunk
        model_path: Path to TTS model
        output_chunk_duration_ms: Duration of output chunks in milliseconds
        ultra_low_latency: Use ultra-low latency mode (~0.27s vs ~2.8s first chunk)
        **kwargs: Additional TTS parameters
    """
    
    from .streaming_buffer import create_streaming_session, AudioStreamCallback
    
    # Create callback manager
    def on_chunk(chunk: np.ndarray, metadata: dict):
        callback(chunk, metadata)
    
    def on_start():
        print("🎵 Starting audio generation...")
    
    def on_complete():
        print("✅ Audio generation completed!")
    
    def on_error(error: Exception):
        print(f"❌ Error during generation: {error}")
    
    callback_manager = AudioStreamCallback(
        on_chunk=on_chunk,
        on_start=on_start,
        on_complete=on_complete,
        on_error=on_error
    )
    
    # Create streaming session
    buffer = create_streaming_session(
        text=text,
        model_path=model_path,
        output_chunk_duration_ms=output_chunk_duration_ms,
        callback=callback_manager,
        ultra_low_latency=ultra_low_latency,
        **kwargs
    )
    
    # Wait for completion with timeout protection
    max_wait_time = 120  # 2 minutes timeout
    wait_start = time.time()
    
    while buffer.is_active():
        if time.time() - wait_start > max_wait_time:
            print("⚠️  Timeout waiting for audio generation to complete")
            break
        time.sleep(0.01)
    
    callback_manager.complete()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio from text using TTS.")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Kokoro-82M-bf16",
        help="Path or repo id of the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to generate (leave blank to input via stdin)",
    )
    parser.add_argument("--voice", type=str, default=None, help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument(
        "--gender", type=str, default="male", help="Gender of the voice [male, female]"
    )
    parser.add_argument("--pitch", type=float, default=1.0, help="Pitch of the voice")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument(
        "--file_prefix", type=str, default="audio", help="Output file name prefix"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--join_audio", action="store_true", help="Join all audio files into one"
    )
    parser.add_argument("--play", action="store_true", help="Play the output audio")
    parser.add_argument(
        "--audio_format", type=str, default="wav", help="Output audio format"
    )
    parser.add_argument(
        "--ref_audio", type=str, default=None, help="Path to reference audio"
    )
    parser.add_argument(
        "--ref_text", type=str, default=None, help="Caption for reference audio"
    )
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model to use to transcribe reference audio",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the model"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for the model")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for the model")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for the model",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the audio as segments instead of saving to a file",
    )
    parser.add_argument(
        "--streaming_interval",
        type=float,
        default=2.0,
        help="The time interval in seconds for streaming segments",
    )
    parser.add_argument(
        "--streaming_chunk_tokens",
        type=int,
        default=21,
        help="Number of tokens per streaming chunk (for Orpheus models, must be multiple of 7)",
    )
    parser.add_argument(
        "--real_streaming",
        action="store_true",
        help="Use real chunked streaming (Orpheus models only)",
    )
    parser.add_argument(
        "--ultra_low_latency",
        action="store_true",
        default=True,
        help="Use ultra-low latency mode (~0.27s vs ~2.8s first chunk, default: True)",
    )
    parser.add_argument(
        "--standard_latency",
        action="store_true",
        help="Use standard latency mode (disables ultra-low latency)",
    )

    args = parser.parse_args()

    if args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            print("Please enter the text to generate:")
            args.text = input("> ").strip()

    return args


def main():
    args = parse_args()
    
    # Determine latency mode
    ultra_low_latency = args.ultra_low_latency and not args.standard_latency
    
    if args.real_streaming:
        # Use real streaming for Orpheus models
        latency_mode = "ultra-low (~0.27s)" if ultra_low_latency else "standard (~2.8s)"
        print(f"🚀 Starting real streaming mode with {latency_mode} latency...")
        
        def audio_callback(chunk: np.ndarray, metadata: dict):
            print(f"🔊 Received chunk {metadata['chunk_index']}: {len(chunk)} samples")
            # Here you could send to audio device, websocket, etc.
        
        generate_audio_with_callback(
            text=args.text,
            callback=audio_callback,
            model_path=args.model,
            voice=args.voice,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            streaming_chunk_tokens=args.streaming_chunk_tokens,
            ultra_low_latency=ultra_low_latency,  # Pass the latency mode
            verbose=args.verbose,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
    else:
        # Use original generate_audio function
        generate_audio(model_path=args.model, **vars(args))


if __name__ == "__main__":
    main()
