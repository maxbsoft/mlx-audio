"""
Audio utilities for smooth streaming TTS
"""

import numpy as np
from typing import List, Optional


def crossfade_chunks(chunks: List[np.ndarray], 
                    crossfade_ms: float = 10.0,
                    sample_rate: int = 24000,
                    preserve_length: bool = True) -> np.ndarray:
    """
    Smoothly concatenate audio chunks with crossfading to eliminate clicks
    
    Args:
        chunks: List of audio chunks (numpy arrays)
        crossfade_ms: Crossfade duration in milliseconds
        sample_rate: Audio sample rate
        preserve_length: If True, preserve total audio length by using additive crossfading
        
    Returns:
        Smoothly concatenated audio array
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)
    
    if not preserve_length:
        # Original overlapping crossfade (shortens audio)
        return crossfade_chunks_overlapping(chunks, crossfade_ms, sample_rate)
    
    # New length-preserving crossfade - smooth connections without overlap
    if crossfade_ms <= 0:
        # No crossfading, just concatenate
        return np.concatenate(chunks)
    
    # Calculate crossfade samples (smaller for edge smoothing)
    crossfade_samples = int(crossfade_ms * sample_rate / 1000.0)
    edge_fade_samples = min(crossfade_samples // 2, 240)  # Max 10ms edge fade
    
    # Start with first chunk
    result = chunks[0].astype(np.float32)
    
    for i in range(1, len(chunks)):
        chunk = chunks[i].astype(np.float32)
        
        # Apply smooth edge transitions without overlap (preserves length)
        fade_samples = min(edge_fade_samples, len(result), len(chunk))
        
        if fade_samples > 0:
            # Create smooth fade curves for edges
            fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_samples)))
            fade_in = 0.5 * (1 + np.cos(np.pi * np.linspace(1, 0, fade_samples)))
            
            # Apply fade to end of result (fade out)
            result[-fade_samples:] *= fade_out
            
            # Apply fade to beginning of chunk (fade in)  
            chunk[:fade_samples] *= fade_in
        
        # Concatenate without overlap (preserves total length)
        result = np.concatenate([result, chunk])
    
    return result


def crossfade_chunks_overlapping(chunks: List[np.ndarray], 
                               crossfade_ms: float = 10.0,
                               sample_rate: int = 24000) -> np.ndarray:
    """
    Original overlapping crossfade method (shortens audio)
    """
    # Calculate crossfade samples
    crossfade_samples = int(crossfade_ms * sample_rate / 1000.0)
    
    # Start with first chunk
    result = chunks[0].astype(np.float32)
    
    for i in range(1, len(chunks)):
        chunk = chunks[i].astype(np.float32)
        
        # Determine overlap region
        overlap_samples = min(crossfade_samples, len(result), len(chunk))
        
        if overlap_samples > 0:
            # Create smooth fade curves using cosine interpolation for smoother transitions
            fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, overlap_samples)))
            fade_in = 0.5 * (1 + np.cos(np.pi * np.linspace(1, 0, overlap_samples)))
            
            # Apply crossfade to overlapping region
            result_tail = result[-overlap_samples:] * fade_out
            chunk_head = chunk[:overlap_samples] * fade_in
            crossfaded_region = result_tail + chunk_head
            
            # Concatenate: result (without tail) + crossfaded region + chunk (without head)
            result = np.concatenate([
                result[:-overlap_samples],
                crossfaded_region,
                chunk[overlap_samples:]
            ])
        else:
            # No overlap possible, just concatenate
            result = np.concatenate([result, chunk])
    
    return result


def smooth_chunk_transition(prev_chunk: Optional[np.ndarray], 
                           current_chunk: np.ndarray,
                           crossfade_ms: float = 5.0,
                           sample_rate: int = 24000) -> np.ndarray:
    """
    Apply smooth transition to a single chunk based on previous chunk
    
    Args:
        prev_chunk: Previous audio chunk (None for first chunk)
        current_chunk: Current audio chunk to process
        crossfade_ms: Crossfade duration in milliseconds
        sample_rate: Audio sample rate
        
    Returns:
        Processed current chunk with smooth transition
    """
    if prev_chunk is None or len(prev_chunk) == 0:
        return current_chunk.astype(np.float32)
    
    current = current_chunk.astype(np.float32)
    crossfade_samples = int(crossfade_ms * sample_rate / 1000.0)
    
    # Only apply fade-in to beginning of current chunk
    fade_samples = min(crossfade_samples, len(current))
    
    if fade_samples > 0:
        # Use smooth cosine fade-in
        fade_in = 0.5 * (1 + np.cos(np.pi * np.linspace(1, 0, fade_samples)))
        current[:fade_samples] *= fade_in
    
    return current


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio to prevent clicks
    
    Args:
        audio: Input audio array
        
    Returns:
        Audio with DC offset removed
    """
    if len(audio) == 0:
        return audio
    
    # Remove DC component
    return audio - np.mean(audio)


def apply_gentle_smoothing(audio: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply gentle smoothing to reduce micro-artifacts
    
    Args:
        audio: Input audio array
        window_size: Size of smoothing window (should be small, e.g., 3-5)
        
    Returns:
        Smoothed audio
    """
    if len(audio) <= window_size:
        return audio
    
    # Simple moving average for very gentle smoothing
    smoothed = np.copy(audio)
    half_window = window_size // 2
    
    for i in range(half_window, len(audio) - half_window):
        start = i - half_window
        end = i + half_window + 1
        smoothed[i] = np.mean(audio[start:end])
    
    return smoothed


def apply_soft_clipping(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Apply soft clipping to prevent harsh distortion
    
    Args:
        audio: Input audio array
        threshold: Clipping threshold (0.0 to 1.0)
        
    Returns:
        Soft-clipped audio
    """
    if len(audio) == 0:
        return audio
    
    # Apply soft clipping using tanh
    sign = np.sign(audio)
    abs_audio = np.abs(audio)
    
    # Apply soft clipping only to samples above threshold
    mask = abs_audio > threshold
    abs_audio[mask] = threshold + (1.0 - threshold) * np.tanh((abs_audio[mask] - threshold) / (1.0 - threshold))
    
    return sign * abs_audio


class StreamingAudioProcessor:
    """
    Stateful audio processor for streaming TTS with smooth transitions
    """
    
    def __init__(self, 
                 crossfade_ms: float = 10.0,
                 sample_rate: int = 24000,
                 remove_dc: bool = True,
                 soft_clip: bool = True,
                 gentle_smoothing: bool = True,
                 max_transition_fade_ms: float = 5.0):
        """
        Initialize streaming audio processor
        
        Args:
            crossfade_ms: Final crossfade duration used when concatenating all processed chunks
            sample_rate: Audio sample rate
            remove_dc: Whether to remove DC offset
            soft_clip: Whether to apply soft clipping
            gentle_smoothing: Whether to apply gentle smoothing
            max_transition_fade_ms: Upper bound for per-chunk fade-in used during streaming to avoid clicks.
                This prevents excessive per-chunk attenuation when chunks are very short.
        """
        self.crossfade_ms = crossfade_ms
        self.sample_rate = sample_rate
        self.remove_dc = remove_dc
        self.soft_clip = soft_clip
        self.gentle_smoothing = gentle_smoothing
        self.max_transition_fade_ms = max_transition_fade_ms
        
        self.prev_chunk = None
        self.processed_chunks = []
    
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a single audio chunk with smooth transitions
        
        Args:
            chunk: Raw audio chunk
            
        Returns:
            Processed audio chunk
        """
        if len(chunk) == 0:
            return chunk
        
        # Convert to float32
        processed = chunk.astype(np.float32)
        
        # Remove DC offset
        if self.remove_dc:
            processed = remove_dc_offset(processed)
        
        # Apply smooth transition with a conservative fade on boundaries to avoid pumping
        transition_fade_ms = min(self.crossfade_ms, self.max_transition_fade_ms)
        processed = smooth_chunk_transition(
            self.prev_chunk,
            processed,
            transition_fade_ms,
            self.sample_rate
        )
        
        # Apply soft clipping
        if self.soft_clip:
            processed = apply_soft_clipping(processed)
        
        # Apply gentle smoothing to reduce micro-artifacts
        if self.gentle_smoothing:
            processed = apply_gentle_smoothing(processed)
        
        # Store for next iteration
        self.prev_chunk = processed.copy() if len(processed) > 0 else None
        self.processed_chunks.append(processed)
        
        return processed
    
    def get_concatenated_audio(self) -> np.ndarray:
        """
        Get final concatenated audio with crossfading
        
        Returns:
            Smoothly concatenated audio
        """
        if not self.processed_chunks:
            return np.array([], dtype=np.float32)
        
        return crossfade_chunks(
            self.processed_chunks,
            self.crossfade_ms,
            self.sample_rate,
            preserve_length=True  # NEW: Preserve audio length
        )
    
    def reset(self):
        """Reset processor state"""
        self.prev_chunk = None
        self.processed_chunks = []
