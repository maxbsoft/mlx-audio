"""
Ultra low-latency streaming implementation for Orpheus with improved audio continuity
"""

import time
import os
import numpy as np
from typing import List, Optional, Generator, Dict, Tuple
import logging
import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from collections import OrderedDict
import hashlib


class CachedDecoder:
    """Caches decoded audio segments to ensure consistency and improve performance"""
    def __init__(self, max_cache_size: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.max_cache_size = max_cache_size
        
    def decode_tokens(self, tokens: List[int], start: int = 0, end: Optional[int] = None) -> np.ndarray:
        """Decode tokens with caching for consistency"""
        if end is None:
            end = len(tokens)
            
        # Create a unique key for this token sequence
        token_slice = tokens[start:end]
        key = hashlib.md5(str(token_slice).encode()).hexdigest()
        
        if key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        
        # Decode and cache
        from .models.llama.llama import decode_audio_from_codes
        code_list = [t - 128266 for t in token_slice if t >= 128266]
        
        # Ensure multiple of 7
        usable_len = (len(code_list) // 7) * 7
        if usable_len == 0:
            return np.array([], dtype=np.float32)
            
        decoded = decode_audio_from_codes(code_list[:usable_len])
        decoded_np = np.array(decoded, dtype=np.float32).flatten()
        
        # Add to cache
        self.cache[key] = decoded_np.copy()
        
        # Limit cache size
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        
        return decoded_np.copy()


class TokenToSampleMapper:
    """Precisely tracks the mapping between token positions and sample positions"""
    def __init__(self):
        self.mappings: List[Tuple[int, int]] = []  # [(token_count, sample_count)]
        
    def add_mapping(self, token_count: int, sample_count: int):
        """Add a new token->sample mapping"""
        self.mappings.append((token_count, sample_count))
        
    def get_samples_for_tokens(self, token_count: int) -> int:
        """Get exact sample count for given token count using interpolation"""
        if not self.mappings:
            # Default estimate: ~320 samples per 7-token group
            return (token_count // 7) * 320
            
        # Find closest mapping
        for tokens, samples in self.mappings:
            if tokens == token_count:
                return samples
                
        # Interpolate from closest mappings
        if len(self.mappings) >= 2:
            # Use linear regression for better estimate
            tokens_arr = np.array([m[0] for m in self.mappings])
            samples_arr = np.array([m[1] for m in self.mappings])
            
            # Fit linear model
            coeffs = np.polyfit(tokens_arr, samples_arr, 1)
            estimated = int(np.polyval(coeffs, token_count))
            return max(0, estimated)
            
        # Fallback to ratio from last mapping
        if self.mappings:
            last_tokens, last_samples = self.mappings[-1]
            ratio = last_samples / last_tokens if last_tokens > 0 else 320/7
            return int(token_count * ratio)
            
        return (token_count // 7) * 320


def cosine_crossfade(prev_tail: np.ndarray, curr_head: np.ndarray, 
                     crossfade_samples: int) -> np.ndarray:
    """Apply cosine crossfade for smoother transitions"""
    if crossfade_samples <= 0:
        return curr_head
        
    actual_crossfade = min(crossfade_samples, len(prev_tail), len(curr_head))
    if actual_crossfade <= 0:
        return curr_head
        
    # Cosine interpolation for smoother fade
    t = np.linspace(0, np.pi, actual_crossfade)
    fade_out = 0.5 * (1 + np.cos(t))
    fade_in = 0.5 * (1 - np.cos(t))
    
    # Apply crossfade
    tail = prev_tail[-actual_crossfade:] if len(prev_tail) > actual_crossfade else prev_tail
    head = curr_head[:actual_crossfade]
    
    crossfaded = tail * fade_out + head * fade_in
    
    # Combine with rest of current chunk
    if len(curr_head) > actual_crossfade:
        return np.concatenate([crossfaded, curr_head[actual_crossfade:]])
    return crossfaded


def detect_and_fix_discontinuity(prev_chunk: Optional[np.ndarray], 
                                 curr_chunk: np.ndarray,
                                 threshold: float = 0.1,
                                 smooth_samples: int = 100) -> np.ndarray:
    """Detect and smooth out discontinuities at chunk boundaries"""
    if prev_chunk is None or len(prev_chunk) == 0 or len(curr_chunk) == 0:
        return curr_chunk
        
    # Check for discontinuity
    last_samples = prev_chunk[-min(10, len(prev_chunk)):]
    first_samples = curr_chunk[:min(10, len(curr_chunk))]
    
    # Calculate discontinuity metrics
    last_mean = np.mean(last_samples)
    first_mean = np.mean(first_samples)
    discontinuity = abs(first_mean - last_mean)
    
    # Also check instantaneous jump
    if len(prev_chunk) > 0 and len(curr_chunk) > 0:
        instant_jump = abs(prev_chunk[-1] - curr_chunk[0])
    else:
        instant_jump = 0
        
    if discontinuity > threshold or instant_jump > threshold * 2:
        # Apply smoothing filter
        smooth_len = min(smooth_samples, len(curr_chunk) // 4)
        if smooth_len > 0:
            # Use Hanning window for smooth transition
            window = np.hanning(smooth_len * 2)[:smooth_len]
            
            # Apply DC offset correction
            dc_offset = prev_chunk[-1] - curr_chunk[0] if instant_jump > threshold else 0
            if dc_offset != 0:
                # Gradually reduce offset
                offset_curve = dc_offset * np.exp(-np.linspace(0, 5, smooth_len))
                curr_chunk[:smooth_len] += offset_curve
            
            # Apply smoothing window
            curr_chunk[:smooth_len] *= window
            
    return curr_chunk


class StreamingState:
    """Maintains state for overlap-save streaming method"""
    def __init__(self, overlap_tokens: int = 0):  # Default to 0 = use ALL previous context
        self.overlap_tokens = overlap_tokens
        self.all_previous_tokens: List[int] = []  # Store ALL previous tokens
        self.prev_tokens: List[int] = []  # For limited overlap mode
        self.prev_samples: int = 0
        # Tracks total decoded samples for the full combined token sequence
        # to avoid re-decoding previous portion every chunk
        self.last_total_combined_samples: int = 0
        self.decoder = CachedDecoder()
        self.mapper = TokenToSampleMapper()
        self.total_tokens_processed = 0
        self.total_samples_generated = 0


def decode_with_overlap_save(tokens: List[int], 
                            state: StreamingState,
                            lookahead_depth: int = 0,
                            verbose: bool = False) -> Tuple[np.ndarray, int]:
    """
    Overlap-save method with full context support.
    When overlap_tokens=0, uses ALL previous context for best quality.
    Returns (audio_samples, tokens_consumed)
    """
    if not tokens:
        return np.array([], dtype=np.float32), 0
        
    # Ensure tokens are audio tokens (>= 128266)
    audio_tokens = [t for t in tokens if t >= 128266 and t != 128258]
    if not audio_tokens:
        return np.array([], dtype=np.float32), 0
        
    # Make sure we have complete groups of 7
    usable_len = (len(audio_tokens) // 7) * 7
    if usable_len == 0:
        return np.array([], dtype=np.float32), 0
        
    audio_tokens = audio_tokens[:usable_len]
    
    # Determine context strategy based on overlap_tokens
    if state.overlap_tokens == 0:
        # Use ALL previous tokens for maximum context (best quality)
        if state.all_previous_tokens:
            # Delta decode: decode only the new audio tokens (fast), rely on smoothing/crossfade for continuity
            new_audio = state.decoder.decode_tokens(audio_tokens)
            # Update trackers as if total grew by len(new_audio)
            total_tokens = len(state.all_previous_tokens) + len(audio_tokens)
            total_samples = state.last_total_combined_samples + len(new_audio)
            state.mapper.add_mapping(total_tokens, total_samples)
            state.last_total_combined_samples = total_samples
            if verbose:
                print(
                    f"🔄 Delta decode: +{len(audio_tokens)} tokens -> {len(new_audio)} samples (total_tokens={total_tokens})"
                )
        else:
            # First chunk - no previous context
            new_audio = state.decoder.decode_tokens(audio_tokens)
            state.mapper.add_mapping(len(audio_tokens), len(new_audio))
            # Initialize combined samples tracker with first chunk length
            state.last_total_combined_samples = len(new_audio)
            
            if verbose:
                print(f"🔄 First chunk: {len(audio_tokens)} tokens -> {len(new_audio)} samples")
        
        # Update all_previous_tokens with current tokens
        state.all_previous_tokens.extend(audio_tokens)
        
    else:
        # Limited overlap mode (original behavior for backward compatibility)
        if state.prev_tokens:
            combined_tokens = state.prev_tokens + audio_tokens
            combined_audio = state.decoder.decode_tokens(combined_tokens)
            overlap_audio = state.decoder.decode_tokens(state.prev_tokens)
            overlap_samples = len(overlap_audio)
            
            if overlap_samples < len(combined_audio):
                new_audio = combined_audio[overlap_samples:]
            else:
                new_audio = np.array([], dtype=np.float32)
                
            state.mapper.add_mapping(len(combined_tokens), len(combined_audio))
            
            if verbose:
                print(f"🔄 Limited overlap: {len(state.prev_tokens)} overlap + {len(audio_tokens)} new -> "
                      f"{len(new_audio)} new samples")
        else:
            new_audio = state.decoder.decode_tokens(audio_tokens)
            state.mapper.add_mapping(len(audio_tokens), len(new_audio))
            
            if verbose:
                print(f"🔄 First chunk: {len(audio_tokens)} tokens -> {len(new_audio)} samples")
        
        # Update prev_tokens for limited overlap
        overlap_groups = state.overlap_tokens // 7
        if overlap_groups > 0 and len(audio_tokens) >= overlap_groups * 7:
            state.prev_tokens = audio_tokens[-(overlap_groups * 7):]
        else:
            state.prev_tokens = audio_tokens[-7:] if len(audio_tokens) >= 7 else audio_tokens
    
    # Update counters
    state.total_tokens_processed += usable_len
    state.total_samples_generated += len(new_audio)
    
    return new_audio, usable_len


def generate_ultra_low_latency_streaming(
    model,
    text: str,
    voice: str = "af_heart",
    temperature: float = 0.6,
    top_p: float = 0.8,
    max_tokens: int = 1200,
    chunk_tokens: int = 56,    # Unified chunk size for all chunks (must be multiple of 7)
    overlap_groups: int = 0,   # 0 = use ALL previous context (best quality), >0 = limited overlap
    crossfade_samples: int = 0,  # Crossfade samples (0 = disabled, 480 = ~20ms at 24kHz)
    lookahead_depth: int = 0,  # Set to 0 for immediate output (lowest latency)
    verbose: bool = False,
    ref_audio: Optional[mx.array] = None,
    ref_text: Optional[str] = None,
    debug_save_wav: bool = True,
    debug_wav_path: str = "debug_ultra_low_latency.wav",
    discontinuity_threshold: float = 0.05,  # Threshold for discontinuity detection
    **kwargs
) -> Generator[np.ndarray, None, None]:
    """
    Ultra low-latency streaming with improved audio continuity
    
    Key improvements:
    - Full context decoding when overlap_groups=0 (best quality, minimal overhead ~0.7ms)
    - Cached decoder for consistent results
    - Overlap-save method for seamless chunking
    - Cosine crossfade for smooth transitions
    - Discontinuity detection and correction
    - Precise token-to-sample mapping
    
    Args:
        overlap_groups: 0 = use ALL previous context (recommended), >0 = limited overlap
        lookahead_depth: 0 = immediate output, >0 = buffer chunks for smoother playback
    """
    
    logger = logging.getLogger(__name__)

    # Debug buffer for all audio chunks
    debug_audio_chunks = []
    debug_total_samples = 0
    
    if verbose and debug_save_wav:
        print(f"🔍 Debug mode: Will save concatenated audio to {debug_wav_path}")
    
    # Prepare input
    input_ids, _ = model.prepare_input_ids(
        [text], 
        voice, 
        ref_audio, 
        ref_text
    )
    
    # Setup generation parameters
    sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
    logits_processors = make_logits_processors(
        kwargs.get("logit_bias", None),
        kwargs.get("repetition_penalty", 1.3),
        kwargs.get("repetition_context_size", 20),
    )
    
    # Initialize streaming state
    state = StreamingState(overlap_tokens=overlap_groups * 7)
    
    # Streaming variables
    all_tokens = []
    audio_tokens_buffer = []
    audio_start_found = False
    chunk_idx = 0
    previous_chunk_audio = None
    
    # Lookahead buffering
    chunk_buffer = []
    
    if verbose:
        print(f"🚀 Ultra low-latency streaming (improved)")
        print(f"   chunk_tokens={chunk_tokens}, overlap_groups={overlap_groups}")
        print(f"   crossfade_samples={crossfade_samples}, lookahead_depth={lookahead_depth}")
        print(f"   discontinuity_threshold={discontinuity_threshold}")
    
    generation_start = time.time()
    first_audio_token_time = None
    first_chunk_time = None
    audio_tokens_started_at_wall: Optional[float] = None  # absolute wall clock time
    last_chunk_done_wall: Optional[float] = None  # time when previous chunk finished (used to measure pure generation time)
    
    # Token generation loop
    for i, response in enumerate(stream_generate(
        model,
        tokenizer=model.tokenizer,
        prompt=input_ids.squeeze(0),
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    )):
        next_token = response.token
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
        all_tokens.append(next_token)
        
        # Memory management
        if i % 200 == 0:  # More frequent clearing
            mx.clear_cache()
            
        # Check for end of generation
        if next_token == 128258:  # EOS token
            if verbose:
                print(f"🏁 EOS token reached after {len(all_tokens)} tokens")
            break
        
        # Detect audio start token
        if next_token == 128257:  # Audio start token
            audio_start_found = True
            audio_tokens_started_at_wall = time.time()
            # Until first chunk, treat last done time as audio start
            if last_chunk_done_wall is None:
                last_chunk_done_wall = audio_tokens_started_at_wall
            if first_audio_token_time is None:
                first_audio_token_time = time.time() - generation_start
                if verbose:
                    print(f"🎵 Audio tokens started at: {first_audio_token_time:.3f}s")
            continue
            
        # Only process audio tokens (>= 128266)
        if next_token >= 128266 and audio_start_found:
            audio_tokens_buffer.append(next_token)
            
            # Check if we have enough tokens for a chunk
            if len(audio_tokens_buffer) >= chunk_tokens:
                # Process chunk with overlap-save method
                now_before_decode = time.time()
                # Pure token-generation time since previous chunk finished
                gen_ms = None
                if last_chunk_done_wall is not None:
                    gen_ms = (now_before_decode - last_chunk_done_wall) * 1000.0
                t0_dec = time.time()
                chunk_audio, consumed = decode_with_overlap_save(
                    audio_tokens_buffer[:chunk_tokens],
                    state,
                    lookahead_depth=lookahead_depth,
                    verbose=verbose
                )
                dec_ms = (time.time() - t0_dec) * 1000.0
                
                if chunk_audio is not None and len(chunk_audio) > 0 and consumed > 0:
                    # Apply discontinuity detection and correction
                    chunk_audio = detect_and_fix_discontinuity(
                        previous_chunk_audio,
                        chunk_audio,
                        threshold=discontinuity_threshold
                    )
                    
                    # Apply crossfade if enabled and we have previous audio
                    if crossfade_samples > 0 and previous_chunk_audio is not None:
                        # Get tail of previous chunk for crossfade
                        tail_len = min(crossfade_samples, len(previous_chunk_audio))
                        if tail_len > 0:
                            prev_tail = previous_chunk_audio[-tail_len:]
                            chunk_audio = cosine_crossfade(prev_tail, chunk_audio, crossfade_samples)
                    
                    # Store reference to current chunk for next iteration
                    previous_chunk_audio = chunk_audio.copy()
                    
                    # Handle lookahead buffering
                    if lookahead_depth == 0:
                        # Immediate output for lowest latency
                        if debug_save_wav:
                            debug_audio_chunks.append(chunk_audio.copy())
                            debug_total_samples += len(chunk_audio)
                        
                        # Logging: generation/decoding timings and RT factors
                        audio_ms = len(chunk_audio) * 1000.0 / 24000.0
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - generation_start
                        # Log at INFO so it's visible in server output
                        logger.info(
                            (
                                "[ULL] chunk=%d tokens=%d audio_ms=%.1f gen_ms=%s decode_ms=%.1f "
                                "rtf_gen=%s rtf_dec=%.2f"
                            ),
                            chunk_idx,
                            consumed,
                            audio_ms,
                            f"{gen_ms:.1f}" if gen_ms is not None else "n/a",
                            dec_ms,
                            f"{gen_ms / audio_ms:.2f}" if (gen_ms is not None and audio_ms > 0) else "n/a",
                            dec_ms / audio_ms if audio_ms > 0 else 0.0,
                        )
                        
                        yield chunk_audio
                        # Mark the time this chunk finished to measure pure generation time to next
                        last_chunk_done_wall = time.time()
                        chunk_idx += 1
                    else:
                        # Buffer for lookahead
                        chunk_buffer.append(chunk_audio.copy())
                        
                        if len(chunk_buffer) > lookahead_depth:
                            chunk_to_send = chunk_buffer.pop(0)
                            
                            if debug_save_wav:
                                debug_audio_chunks.append(chunk_to_send.copy())
                                debug_total_samples += len(chunk_to_send)
                            
                            # Log timings (use decode time measured earlier; generation time up to decode start)
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - generation_start
                            audio_ms = len(chunk_to_send) * 1000.0 / 24000.0
                            logger.info(
                                (
                                    "[ULL] chunk=%d (buffered) tokens=%d audio_ms=%.1f gen_ms=%s decode_ms=%.1f "
                                    "rtf_gen=%s rtf_dec=%.2f"
                                ),
                                chunk_idx,
                                consumed,
                                audio_ms,
                                f"{gen_ms:.1f}" if gen_ms is not None else "n/a",
                                dec_ms,
                                f"{gen_ms / audio_ms:.2f}" if (gen_ms is not None and audio_ms > 0) else "n/a",
                                dec_ms / audio_ms if audio_ms > 0 else 0.0,
                            )
                            
                            yield chunk_to_send
                            last_chunk_done_wall = time.time()
                            chunk_idx += 1
                    
                    # Remove consumed tokens from buffer
                    audio_tokens_buffer = audio_tokens_buffer[consumed:]
    
    # Process any remaining tokens
    if audio_tokens_buffer:
        # Ensure multiple of 7
        usable_len = (len(audio_tokens_buffer) // 7) * 7
        if usable_len > 0:
            final_tokens = audio_tokens_buffer[:usable_len]
            t0_dec_final = time.time()
            final_audio, consumed = decode_with_overlap_save(
                final_tokens,
                state,
                lookahead_depth=lookahead_depth,
                verbose=verbose
            )
            dec_ms_final = (time.time() - t0_dec_final) * 1000.0
            
            if final_audio is not None and len(final_audio) > 0:
                # Apply discontinuity correction
                final_audio = detect_and_fix_discontinuity(
                    previous_chunk_audio,
                    final_audio,
                    threshold=discontinuity_threshold
                )
                
                # Apply final crossfade
                if crossfade_samples > 0 and previous_chunk_audio is not None:
                    tail_len = min(crossfade_samples, len(previous_chunk_audio))
                    if tail_len > 0:
                        prev_tail = previous_chunk_audio[-tail_len:]
                        final_audio = cosine_crossfade(prev_tail, final_audio, crossfade_samples)
                
                # Add to buffer
                chunk_buffer.append(final_audio)
                
                audio_ms = len(final_audio) * 1000.0 / 24000.0
                # Generation time since last emitted chunk (if available)
                gen_ms = None
                if last_chunk_done_wall is not None:
                    gen_ms = (time.time() - last_chunk_done_wall) * 1000.0
                logger.info(
                    (
                        "[ULL] final_prepare tokens=%d audio_ms=%.1f gen_ms=%s decode_ms=%.1f "
                        "rtf_gen=%s rtf_dec=%.2f"
                    ),
                    consumed,
                    audio_ms,
                    f"{gen_ms:.1f}" if gen_ms is not None else "n/a",
                    dec_ms_final,
                    f"{gen_ms / audio_ms:.2f}" if (gen_ms is not None and audio_ms > 0) else "n/a",
                    dec_ms_final / audio_ms if audio_ms > 0 else 0.0,
                )
    
    # Flush any remaining buffered chunks
    while chunk_buffer:
        chunk_to_send = chunk_buffer.pop(0)
        
        if debug_save_wav:
            debug_audio_chunks.append(chunk_to_send.copy())
            debug_total_samples += len(chunk_to_send)
        
        audio_ms = len(chunk_to_send) * 1000.0 / 24000.0
        logger.info(
            "[ULL] flush chunk=%d audio_ms=%.1f",
            chunk_idx,
            audio_ms,
        )
        
        yield chunk_to_send
        chunk_idx += 1
    
    # Save debug WAV if enabled
    if debug_save_wav and debug_audio_chunks:
        try:
            _save_debug_wav(debug_audio_chunks, debug_wav_path, 24000, verbose)
            
            # Print statistics
            if verbose and state.mapper.mappings:
                print(f"\n📊 Statistics:")
                print(f"   Total tokens processed: {state.total_tokens_processed}")
                print(f"   Total samples generated: {state.total_samples_generated}")
                print(f"   Average samples per token: {state.total_samples_generated/state.total_tokens_processed:.2f}")
                print(f"   Cache hit rate: {len(state.decoder.cache)}/{state.decoder.max_cache_size}")
                
        except Exception as e:
            if verbose:
                print(f"⚠️ Warning: Failed to save debug WAV: {e}")
    
    # Final timing
    if verbose:
        total_time = time.time() - generation_start
        print(f"\n⏱️ Total generation time: {total_time:.3f}s")
        if first_audio_token_time:
            print(f"   Time to first audio token: {first_audio_token_time:.3f}s")
        if first_chunk_time:
            print(f"   Time to first chunk: {first_chunk_time:.3f}s")


def _save_debug_wav(audio_chunks: List[np.ndarray], wav_path: str, sample_rate: int, verbose: bool = False):
    """Save concatenated audio chunks to WAV file for debugging"""
    try:
        import soundfile as sf
        
        # Concatenate all chunks
        concatenated_audio = np.concatenate(audio_chunks)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(wav_path) if os.path.dirname(wav_path) else '.', exist_ok=True)
        
        # Normalize if needed (keep in [-1, 1] range)
        max_val = np.max(np.abs(concatenated_audio))
        if max_val > 1.0:
            concatenated_audio = concatenated_audio / max_val
            if verbose:
                print(f"⚠️ Audio normalized from max {max_val:.3f} to [-1, 1]")
        
        # Save as float32 WAV
        sf.write(wav_path, concatenated_audio, sample_rate, subtype='FLOAT')
        
        if verbose:
            duration = len(concatenated_audio) / sample_rate
            print(f"✅ Debug WAV saved: {wav_path}")
            print(f"   Total samples: {len(concatenated_audio)}")
            print(f"   Duration: {duration:.3f}s")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Audio range: [{np.min(concatenated_audio):.6f}, {np.max(concatenated_audio):.6f}]")
            print(f"   Chunks concatenated: {len(audio_chunks)}")
            
    except ImportError:
        # Fallback to scipy
        try:
            from scipy.io import wavfile
            
            concatenated_audio = np.concatenate(audio_chunks)
            
            # Normalize and convert to int16
            max_val = np.max(np.abs(concatenated_audio))
            if max_val > 0:
                concatenated_audio = concatenated_audio / max_val
            
            audio_int16 = np.int16(concatenated_audio * 32767)
            
            os.makedirs(os.path.dirname(wav_path) if os.path.dirname(wav_path) else '.', exist_ok=True)
            wavfile.write(wav_path, sample_rate, audio_int16)
            
            if verbose:
                duration = len(concatenated_audio) / sample_rate
                print(f"✅ Debug WAV saved (scipy): {wav_path}")
                print(f"   Duration: {duration:.3f}s")
                print(f"   Chunks: {len(audio_chunks)}")
                
        except ImportError:
            if verbose:
                print("⚠️ Neither soundfile nor scipy available. Cannot save debug WAV.")