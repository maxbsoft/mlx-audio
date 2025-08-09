"""
Example usage of Orpheus streaming TTS functionality.
"""

import time
import numpy as np
import soundfile as sf
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def example_basic_streaming():
    """Basic streaming example"""
    
    print("🎵 Example 1: Basic Streaming")
    print("-" * 40)
    
    from tts.generate import generate_audio_streaming
    
    text = """
    Welcome to the Orpheus streaming TTS demo! 
    This system can generate audio in real-time chunks, 
    allowing for immediate playback without waiting for complete generation.
    """
    
    print(f"📝 Generating: {text.strip()}")
    
    chunks = []
    start_time = time.time()
    
    for i, audio_chunk in enumerate(generate_audio_streaming(
        text=text,
        model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
        voice="af_heart",
        streaming_chunk_tokens=21,  # 3 audio frames
        temperature=0.6,
        verbose=True
    )):
        chunks.append(audio_chunk)
        
        # Here you could play the chunk immediately
        print(f"🔊 Got chunk {i}: {len(audio_chunk)} samples "
              f"({len(audio_chunk)/24000:.3f}s)")
        
        # Simulate some processing
        time.sleep(0.01)
    
    # Combine and save
    if chunks:
        full_audio = np.concatenate(chunks)
        output_file = "example_basic_streaming.wav"
        sf.write(output_file, full_audio, 24000)
        
        total_time = time.time() - start_time
        audio_duration = len(full_audio) / 24000
        
        print(f"\n📊 Results:")
        print(f"   ✅ Saved to: {output_file}")
        print(f"   📏 Audio duration: {audio_duration:.2f}s")
        print(f"   ⏱️  Generation time: {total_time:.2f}s")
        print(f"   🚀 Real-time factor: {total_time/audio_duration:.2f}x")


def example_callback_streaming():
    """Callback-based streaming example"""
    
    print("\n🎵 Example 2: Callback Streaming")
    print("-" * 40)
    
    from tts.generate import generate_audio_with_callback
    
    # Audio accumulator
    audio_data = []
    
    def my_audio_callback(chunk: np.ndarray, metadata: dict):
        """Custom callback to handle audio chunks"""
        
        audio_data.append(chunk)
        
        duration_ms = len(chunk) / 24000 * 1000
        
        print(f"📡 Received chunk {metadata['chunk_index']}: "
              f"{len(chunk)} samples ({duration_ms:.1f}ms)")
        
        # Here you could:
        # - Send to audio device for immediate playback
        # - Stream over network (WebRTC, WebSocket, etc.)
        # - Process in real-time (effects, analysis, etc.)
        # - Buffer for smooth playback
    
    text = "This demonstrates callback-based streaming for real-time audio processing."
    
    print(f"📝 Processing: {text}")
    
    start_time = time.time()
    
    generate_audio_with_callback(
        text=text,
        callback=my_audio_callback,
        model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
        voice="af_heart",
        streaming_chunk_tokens=14,  # 2 audio frames = ~100ms
        temperature=0.6,
        output_chunk_duration_ms=100,  # 100ms output chunks
    )
    
    total_time = time.time() - start_time
    
    if audio_data:
        full_audio = np.concatenate(audio_data)
        output_file = "example_callback_streaming.wav"
        sf.write(output_file, full_audio, 24000)
        
        print(f"\n📊 Results:")
        print(f"   📞 Callbacks received: {len(audio_data)}")
        print(f"   ✅ Saved to: {output_file}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")


def example_low_latency_streaming():
    """Low-latency streaming example"""
    
    print("\n🎵 Example 3: Low-Latency Streaming")
    print("-" * 40)
    
    from tts.generate import generate_audio_streaming
    
    text = "Ultra low latency streaming test."
    
    print(f"📝 Text: {text}")
    print("🚀 Using 7-token chunks for minimum latency...")
    
    chunks = []
    first_chunk_time = None
    start_time = time.time()
    
    for i, audio_chunk in enumerate(generate_audio_streaming(
        text=text,
        model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
        voice="af_heart",
        streaming_chunk_tokens=7,  # Minimum: 1 audio frame
        temperature=0.6,
        verbose=True
    )):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start_time
            print(f"⚡ First chunk latency: {first_chunk_time:.3f}s")
        
        chunks.append(audio_chunk)
        
        duration_ms = len(audio_chunk) / 24000 * 1000
        elapsed = time.time() - start_time
        
        print(f"⚡ Chunk {i}: {len(audio_chunk)} samples, "
              f"{duration_ms:.1f}ms, elapsed: {elapsed:.3f}s")
    
    if chunks:
        full_audio = np.concatenate(chunks)
        output_file = "example_low_latency_streaming.wav"
        sf.write(output_file, full_audio, 24000)
        
        total_time = time.time() - start_time
        audio_duration = len(full_audio) / 24000
        
        print(f"\n📊 Low-Latency Results:")
        print(f"   ⚡ First chunk: {first_chunk_time:.3f}s")
        print(f"   🔢 Total chunks: {len(chunks)}")
        print(f"   📏 Audio duration: {audio_duration:.3f}s")
        print(f"   ⏱️  Generation time: {total_time:.3f}s")
        print(f"   ✅ Saved to: {output_file}")


def example_streaming_buffer():
    """Advanced streaming buffer example"""
    
    print("\n🎵 Example 4: Streaming Buffer")
    print("-" * 40)
    
    from tts.streaming_buffer import (
        OrpheusStreamingBuffer, 
        AudioStreamCallback
    )
    from tts.generate import generate_audio_streaming
    
    # Create buffer for smooth 50ms chunks
    buffer = OrpheusStreamingBuffer(
        chunk_duration_ms=50,
        buffer_size=20,
        sample_rate=24000
    )
    
    # Track received data
    received_chunks = []
    
    def on_start():
        print("🎬 Buffer streaming started")
    
    def on_chunk(chunk: np.ndarray, metadata: dict):
        received_chunks.append(chunk)
        print(f"🔄 Buffer chunk {metadata['chunk_index']}: "
              f"{len(chunk)} samples, "
              f"elapsed: {metadata['elapsed_time']:.3f}s")
    
    def on_complete():
        print("🎉 Buffer streaming completed!")
    
    def on_error(error):
        print(f"❌ Buffer error: {error}")
    
    # Create callback manager
    callback = AudioStreamCallback(
        on_start=on_start,
        on_chunk=on_chunk,
        on_complete=on_complete,
        on_error=on_error
    )
    
    # Audio generator function
    def audio_generator():
        text = "This demonstrates advanced streaming buffer functionality."
        for chunk in generate_audio_streaming(
            text=text,
            model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
            voice="af_heart",
            streaming_chunk_tokens=21,
            verbose=False
        ):
            yield chunk
    
    # Start streaming
    callback.start()
    thread = buffer.start_streaming(audio_generator, callback.chunk)
    
    # Monitor progress
    start_time = time.time()
    while buffer.is_active() and (time.time() - start_time) < 30:
        status = buffer.get_status()
        time.sleep(0.1)
    
    callback.complete()
    thread.join(timeout=5)
    
    # Save result
    if received_chunks:
        full_audio = np.concatenate(received_chunks)
        output_file = "example_streaming_buffer.wav"
        sf.write(output_file, full_audio, 24000)
        
        print(f"\n📊 Buffer Results:")
        print(f"   🔢 Buffer chunks: {len(received_chunks)}")
        print(f"   📏 Total samples: {len(full_audio)}")
        print(f"   📏 Audio duration: {len(full_audio)/24000:.3f}s")
        print(f"   ✅ Saved to: {output_file}")


def main():
    """Run all examples"""
    
    print("🎯 Orpheus Streaming TTS Examples")
    print("=" * 50)
    print("This demo shows different ways to use streaming TTS:")
    print("1. Basic streaming with generator")
    print("2. Callback-based streaming")
    print("3. Low-latency streaming")
    print("4. Advanced streaming buffer")
    print("=" * 50)
    
    try:
        example_basic_streaming()
        example_callback_streaming()
        example_low_latency_streaming()
        example_streaming_buffer()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📁 Generated files:")
        print("   - example_basic_streaming.wav")
        print("   - example_callback_streaming.wav")
        print("   - example_low_latency_streaming.wav")
        print("   - example_streaming_buffer.wav")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
