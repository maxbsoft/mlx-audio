#!/usr/bin/env python3
"""
Voice Comparison Test - Generate all audio samples with the same voice for fair comparison
"""

import sys
import os
import time
import numpy as np

# Add mlx_audio to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mlx_audio'))

# Use consistent voice across all tests
CONSISTENT_VOICE = "af_heart"
TEST_TEXT = "Hello, this is a consistent voice test to compare audio quality between normal and streaming generation methods."

def generate_normal_audio():
    """Generate normal audio with consistent voice"""
    
    print("🎯 Generating normal audio...")
    
    try:
        from tts.generate import generate_audio
        
        generate_audio(
            text=TEST_TEXT,
            model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
            voice=CONSISTENT_VOICE,
            file_prefix="consistent_normal",
            verbose=False
        )
        print("✅ Normal generation saved as: consistent_normal_000.wav")
        
    except Exception as e:
        print(f"❌ Normal generation failed: {e}")
        return False
    
    return True


def generate_clean_streaming_audio():
    """Generate streaming audio without processing (clean)"""
    
    print("\n🎯 Generating clean streaming audio...")
    
    try:
        from tts.generate import generate_audio_streaming
        
        chunks = []
        start_time = time.time()
        
        for i, audio_chunk in enumerate(generate_audio_streaming(
            text=TEST_TEXT,
            model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
            voice=CONSISTENT_VOICE,
            streaming_chunk_tokens=35,
            temperature=0.6,
            verbose=False
        )):
            # NO PROCESSING - just store raw chunks
            chunks.append(audio_chunk.astype(np.float32))
        
        # SIMPLE CONCATENATION - no crossfading, no processing
        if chunks:
            total_audio = np.concatenate(chunks)
        else:
            total_audio = np.array([])
        
        # Save clean streaming result
        import soundfile as sf
        sf.write("consistent_clean_streaming.wav", total_audio, 24000)
        print("✅ Clean streaming saved as: consistent_clean_streaming.wav")
        
    except Exception as e:
        print(f"❌ Clean streaming failed: {e}")
        return False
    
    return True


def generate_processed_streaming_audio():
    """Generate streaming audio with processing (crossfading, etc.)"""
    
    print("\n🎯 Generating processed streaming audio...")
    
    try:
        from tts.generate import generate_audio_streaming
        from tts.audio_utils import StreamingAudioProcessor
        
        chunks = []
        audio_processor = StreamingAudioProcessor(crossfade_ms=10.0)
        
        for i, audio_chunk in enumerate(generate_audio_streaming(
            text=TEST_TEXT,
            model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
            voice=CONSISTENT_VOICE,
            streaming_chunk_tokens=35,
            temperature=0.6,
            verbose=False
        )):
            # Process chunk with smooth transitions
            processed_chunk = audio_processor.process_chunk(audio_chunk)
            chunks.append(processed_chunk)
        
        # Get crossfaded audio
        total_audio = audio_processor.get_concatenated_audio() if chunks else np.array([])
        
        # Save processed streaming result
        import soundfile as sf
        sf.write("consistent_processed_streaming.wav", total_audio, 24000)
        print("✅ Processed streaming saved as: consistent_processed_streaming.wav")
        
    except Exception as e:
        print(f"❌ Processed streaming failed: {e}")
        return False
    
    return True


def generate_callback_streaming_audio():
    """Generate streaming audio using callback method"""
    
    print("\n🎯 Generating callback streaming audio...")
    
    try:
        from tts.generate import generate_audio_with_callback
        
        received_chunks = []
        
        def audio_callback(chunk: np.ndarray, metadata: dict):
            # Store raw chunks without processing
            received_chunks.append(chunk.astype(np.float32))
        
        generate_audio_with_callback(
            text=TEST_TEXT,
            callback=audio_callback,
            model_path="mlx-community/orpheus-3b-0.1-ft-4bit",
            voice=CONSISTENT_VOICE,
            streaming_chunk_tokens=35,
            temperature=0.6,
            output_chunk_duration_ms=150,
            ultra_low_latency=True
        )
        
        # Simple concatenation without processing
        if received_chunks:
            total_audio = np.concatenate(received_chunks)
        else:
            total_audio = np.array([])
        
        # Save callback streaming result
        import soundfile as sf
        sf.write("consistent_callback_streaming.wav", total_audio, 24000)
        print("✅ Callback streaming saved as: consistent_callback_streaming.wav")
        
    except Exception as e:
        print(f"❌ Callback streaming failed: {e}")
        return False
    
    return True


def main():
    """Run comprehensive voice comparison test"""
    
    print("🎤 Voice Comparison Test")
    print("=" * 50)
    print(f"🔊 Using consistent voice: {CONSISTENT_VOICE}")
    print(f"📝 Text: {TEST_TEXT}")
    print("=" * 50)
    
    results = []
    
    # Generate all variants
    results.append(("Normal Generation", generate_normal_audio()))
    results.append(("Clean Streaming", generate_clean_streaming_audio()))
    results.append(("Processed Streaming", generate_processed_streaming_audio()))
    results.append(("Callback Streaming", generate_callback_streaming_audio()))
    
    # Summary
    print(f"\n📋 Generation Summary:")
    print("=" * 30)
    
    successful = 0
    for method, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {method}")
        if success:
            successful += 1
    
    if successful > 0:
        print(f"\n🎧 Audio Files for Comparison:")
        print("   • consistent_normal_000.wav (normal generation)")
        print("   • consistent_clean_streaming.wav (streaming without processing)")
        print("   • consistent_processed_streaming.wav (streaming with crossfading)")
        print("   • consistent_callback_streaming.wav (callback-based streaming)")
        
        print(f"\n🎯 Comparison Tips:")
        print(f"   1. Normal vs Clean Streaming should sound identical")
        print(f"   2. Processed Streaming may sound slightly different (softer/muffled)")
        print(f"   3. All use the same voice ({CONSISTENT_VOICE}) for fair comparison")
        print(f"   4. Listen for differences in clarity, loudness, and quality")
    
    print(f"\n✅ Voice comparison test completed!")


if __name__ == "__main__":
    main()
