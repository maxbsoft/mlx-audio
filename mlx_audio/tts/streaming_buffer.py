"""
Streaming buffer implementation for smooth audio delivery.
"""

import queue
import threading
import time
from typing import Callable, Optional, Generator
import numpy as np


class OrpheusStreamingBuffer:
    """Buffer for smooth streaming of Orpheus audio with callback support"""
    
    def __init__(self, 
                 chunk_duration_ms: int = 100,
                 buffer_size: int = 100,  # Increased from 50 to 100 to prevent overflow
                 sample_rate: int = 24000,
                 zc_search_ms: int = 3,
                 base_step_samples: int = 320,
                 respect_source_boundaries: bool = False):
        """
        Initialize streaming buffer
        
        Args:
            chunk_duration_ms: Target duration of output chunks in milliseconds
            buffer_size: Maximum number of chunks to buffer
            sample_rate: Audio sample rate
        """
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.target_chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        # Zero-crossing search window in samples on boundaries
        self.zc_search_samples = max(0, int(sample_rate * zc_search_ms / 1000))
        # Align cut to multiples of codec step (~320 samples per audio token for Orpheus/SNAC)
        self.base_step_samples = max(1, int(base_step_samples))
        # If True, do not cut incoming chunks; forward them as-is
        self.respect_source_boundaries = respect_source_boundaries
        
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.is_generating = False
        self.generation_complete = False
        self.error = None
        
    def start_streaming(self, 
                       generate_func: Callable[[], Generator[np.ndarray, None, None]],
                       callback: Optional[Callable[[np.ndarray], None]] = None) -> threading.Thread:
        """
        Start streaming in background thread
        
        Args:
            generate_func: Function that returns generator yielding audio chunks
            callback: Optional callback function for each chunk
            
        Returns:
            threading.Thread: The generation thread
        """
        
        def generation_worker():
            self.is_generating = True
            self.error = None
            try:
                accumulated_audio = np.array([], dtype=np.float32)
                
                for audio_chunk in generate_func():
                    if audio_chunk is None or len(audio_chunk) == 0:
                        continue
                        
                    # If forwarding as-is, bypass accumulator and emit chunk directly
                    if self.respect_source_boundaries:
                        try:
                            self.buffer.put(audio_chunk, timeout=1.0)
                            if callback:
                                callback(audio_chunk)
                        except queue.Full:
                            print("Warning: Buffer overflow, dropping audio chunk")
                        continue

                    # Accumulate audio otherwise
                    accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])
                    
                    # Yield chunks of target size, snapping to codec step and cutting at nearest zero-crossing to reduce clicks
                    while len(accumulated_audio) >= self.target_chunk_size:
                        # First snap to nearest multiple of base step (codec-friendly)
                        snapped = int(round(self.target_chunk_size / self.base_step_samples) * self.base_step_samples)
                        cut_index = min(snapped, len(accumulated_audio))
                        if self.zc_search_samples > 0:
                            window_start = cut_index - self.zc_search_samples
                            window_end = cut_index + self.zc_search_samples
                            window_start = max(1, window_start)
                            window_end = min(len(accumulated_audio), window_end)
                            segment = accumulated_audio[window_start:window_end]
                            # Find sign changes (zero-crossings)
                            if segment.size > 1:
                                signs = np.sign(segment)
                                zc = np.where(signs[:-1] * signs[1:] <= 0)[0]
                                if zc.size > 0:
                                    # Choose the zero-crossing closest to target boundary
                                    rel_indices = zc + window_start
                                    cut_index = int(rel_indices[np.argmin(np.abs(rel_indices - cut_index))])
                        chunk = accumulated_audio[:cut_index]
                        accumulated_audio = accumulated_audio[cut_index:]
                        
                        try:
                            self.buffer.put(chunk, timeout=1.0)
                            if callback:
                                callback(chunk)
                        except queue.Full:
                            print("Warning: Buffer overflow, dropping audio chunk")
                
                # Handle remaining audio
                if len(accumulated_audio) > 0:
                    try:
                        self.buffer.put(accumulated_audio, timeout=1.0)
                        if callback:
                            callback(accumulated_audio)
                    except queue.Full:
                        print("Warning: Buffer overflow, dropping final chunk")
                        
            except Exception as e:
                self.error = e
                print(f"Error in audio generation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.is_generating = False
                self.generation_complete = True
        
        thread = threading.Thread(target=generation_worker, daemon=True)
        thread.start()
        return thread
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get next audio chunk from buffer
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio chunk or None if timeout/empty
        """
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            if self.generation_complete:
                return None
            return np.array([], dtype=np.float32)  # Return empty array if still generating
    
    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio chunks as they become available
        
        Yields:
            Audio chunks from the buffer
        """
        while self.is_generating or not self.buffer.empty():
            chunk = self.get_audio_chunk(timeout=0.1)
            if chunk is not None and len(chunk) > 0:
                yield chunk
            elif self.generation_complete:
                break
    
    def is_active(self) -> bool:
        """Check if generation is still active or buffer has data"""
        return self.is_generating or (not self.generation_complete and not self.buffer.empty())
    
    def get_status(self) -> dict:
        """Get current buffer status"""
        return {
            'is_generating': self.is_generating,
            'generation_complete': self.generation_complete,
            'buffer_size': self.buffer.qsize(),
            'error': str(self.error) if self.error else None
        }


class AudioStreamCallback:
    """Helper class for managing audio streaming callbacks"""
    
    def __init__(self, 
                 on_chunk: Optional[Callable[[np.ndarray, dict], None]] = None,
                 on_start: Optional[Callable[[], None]] = None,
                 on_complete: Optional[Callable[[], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
        """
        Initialize callback manager
        
        Args:
            on_chunk: Called for each audio chunk with (chunk, metadata)
            on_start: Called when streaming starts
            on_complete: Called when streaming completes
            on_error: Called if error occurs
        """
        self.on_chunk = on_chunk
        self.on_start = on_start
        self.on_complete = on_complete
        self.on_error = on_error
        
        self.chunk_count = 0
        self.start_time = None
        self.total_samples = 0
    
    def start(self):
        """Signal start of streaming"""
        self.start_time = time.time()
        self.chunk_count = 0
        self.total_samples = 0
        if self.on_start:
            self.on_start()
    
    def chunk(self, audio_chunk: np.ndarray, extra_metadata: dict = None):
        """Process audio chunk"""
        if self.on_chunk:
            metadata = {
                'chunk_index': self.chunk_count,
                'chunk_size': len(audio_chunk),
                'total_samples': self.total_samples,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0,
                'timestamp': time.time()
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            
            self.on_chunk(audio_chunk, metadata)
        
        self.chunk_count += 1
        self.total_samples += len(audio_chunk)
    
    def complete(self):
        """Signal completion of streaming"""
        if self.on_complete:
            self.on_complete()
    
    def error(self, exception: Exception):
        """Signal error during streaming"""
        if self.on_error:
            self.on_error(exception)
        else:
            print(f"Streaming error: {exception}")


def create_streaming_session(
    text: str,
    model_path: str = "mlx-community/orpheus-3b-0.1-ft-4bit",
    voice: str = "af_heart",
    streaming_chunk_tokens: int = 21,
    output_chunk_duration_ms: int = 100,
    callback: Optional[AudioStreamCallback] = None,
    ultra_low_latency: bool = True,
    respect_source_boundaries: bool = False,
    base_step_samples: Optional[int] = None,
    **tts_kwargs
) -> OrpheusStreamingBuffer:
    """
    Create a complete streaming session with buffer and callbacks
    
    Args:
        text: Text to synthesize
        model_path: Path to TTS model
        voice: Voice to use
        streaming_chunk_tokens: Tokens per generation chunk
        output_chunk_duration_ms: Duration of output chunks in ms
        callback: Callback manager for handling events
        ultra_low_latency: Use ultra-low latency mode (~0.27s vs ~2.8s first chunk)
        **tts_kwargs: Additional TTS arguments
        
    Returns:
        OrpheusStreamingBuffer: Configured buffer ready for streaming
    """
    # Estimate codec step if not provided (7 audio tokens per group for SNAC)
    if base_step_samples is None:
        try:
            from .models.llama.llama import decode_audio_from_codes
            codes = [0, 4096, 8192, 12288, 16384, 20480, 24576]  # one 7-token group
            audio = decode_audio_from_codes(codes)
            base_step_samples = int(audio.shape[-1])
        except Exception:
            base_step_samples = 2055  # sensible default observed empirically

    buffer = OrpheusStreamingBuffer(
        chunk_duration_ms=output_chunk_duration_ms,
        sample_rate=24000,
        base_step_samples=base_step_samples,
        respect_source_boundaries=respect_source_boundaries,
    )
    
    def generate_audio():
        # Import here to avoid circular imports
        from .generate import generate_audio_streaming
        
        for chunk in generate_audio_streaming(
            text=text,
            model_path=model_path,
            voice=voice,
            streaming_chunk_tokens=streaming_chunk_tokens,
            ultra_low_latency=ultra_low_latency,
            **tts_kwargs
        ):
            yield chunk
    
    def chunk_callback(chunk: np.ndarray):
        if callback:
            callback.chunk(chunk)
    
    # Start streaming
    if callback:
        callback.start()
    
    try:
        buffer.start_streaming(generate_audio, chunk_callback)
    except Exception as e:
        if callback:
            callback.error(e)
        raise
    
    return buffer
