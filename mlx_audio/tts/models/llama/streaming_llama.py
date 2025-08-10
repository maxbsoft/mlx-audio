"""
Streaming implementation for Orpheus TTS model with real chunked audio generation.
"""

import time
import os
from typing import Generator, Callable, Optional, List
import mlx.core as mx
import numpy as np
from .llama import Model, decode_audio_from_codes, ModelConfig
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from ..base import GenerationResult


class StreamingOrpheusModel(Model):
    """Orpheus model with real-time chunked streaming support"""
    
    @classmethod 
    def from_model(cls, base_model):
        """Create streaming model from existing model"""
        # Create new instance and copy everything
        streaming_model = cls.__new__(cls)
        streaming_model.__dict__.update(base_model.__dict__)
        
        # Ensure we have all necessary attributes
        if not hasattr(streaming_model, 'model'):
            streaming_model.model = base_model
            
        return streaming_model
    
    def generate_streaming(
        self,
        text: str,
        voice: str,
        temperature: float = 0.6,
        top_p: float = 0.8,
        max_tokens: int = 1200,
        streaming_chunk_tokens: int = 21,  # 3 audio frames = 21 tokens
        verbose: bool = False,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        debug_save_wav: bool = True,  # ← New parameter for debugging
        debug_wav_path: str = "debug_streaming_output.wav",  # ← Path for saving
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate audio with real streaming every streaming_chunk_tokens
        
        Args:
            streaming_chunk_tokens: Number of tokens per chunk (must be multiple of 7)
            debug_save_wav: Save all chunks concatenated to WAV file for debugging
            debug_wav_path: Path to save debug WAV file
        
        Yields:
            GenerationResult: Audio chunks as they are generated
        """
        
        # Ensure chunk size is multiple of 7 (required for SNAC decoder)
        if streaming_chunk_tokens % 7 != 0:
            streaming_chunk_tokens = ((streaming_chunk_tokens // 7) + 1) * 7
            if verbose:
                print(f"Adjusted chunk size to {streaming_chunk_tokens} tokens (multiple of 7)")
        
        # Prepare input_ids
        input_ids, _ = self.prepare_input_ids([text], voice, ref_audio, ref_text)
        
        sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.3),
            kwargs.get("repetition_context_size", 20),
        )
        
        # Streaming accumulators
        accumulated_tokens = []
        yielded_chunks_count = 0
        segment_start_time = time.time()
        total_start_time = time.time()
        
        # ← Debug buffer for all audio chunks
        debug_audio_chunks = []
        debug_total_samples = 0
        
        if verbose:
            print(f"Starting streaming generation with {streaming_chunk_tokens} tokens per chunk")
            if debug_save_wav:
                print(f"🔍 Debug mode: Will save concatenated audio to {debug_wav_path}")
        
        # Token generation loop
        for i, response in enumerate(stream_generate(
            self,
            tokenizer=self.tokenizer,
            prompt=input_ids.squeeze(0),
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )):
            next_token = response.token
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
            accumulated_tokens.append(next_token)
            
            # Memory management
            if i % 50 == 0:
                mx.clear_cache()
                
            # Check for end of generation
            if next_token == 128258:  # EOS token
                if verbose:
                    print("EOS token reached, finishing generation")
                break
                
            # Stream every streaming_chunk_tokens tokens
            if len(accumulated_tokens) >= streaming_chunk_tokens:
                # Decode accumulated tokens
                audio_chunk = self._decode_partial_tokens(
                    accumulated_tokens[:streaming_chunk_tokens]
                )
                
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # ← Save chunk to debug buffer
                    if debug_save_wav:
                        debug_audio_chunks.append(audio_chunk.copy())
                        debug_total_samples += len(audio_chunk)
                        if verbose:
                            print(f"�� Debug: Added chunk {yielded_chunks_count} with {len(audio_chunk)} samples")
                    
                    segment_time = time.time() - segment_start_time
                    audio_duration = len(audio_chunk) / self.sample_rate
                    
                    yield GenerationResult(
                        audio=audio_chunk,
                        samples=len(audio_chunk),
                        sample_rate=self.sample_rate,
                        segment_idx=yielded_chunks_count,
                        token_count=streaming_chunk_tokens,
                        audio_duration=self._format_duration(audio_duration),
                        real_time_factor=segment_time / audio_duration if audio_duration > 0 else 0,
                        prompt={
                            "tokens": streaming_chunk_tokens,
                            "tokens-per-sec": streaming_chunk_tokens / segment_time if segment_time > 0 else 0,
                        },
                        audio_samples={
                            "samples": len(audio_chunk),
                            "samples-per-sec": len(audio_chunk) / segment_time if segment_time > 0 else 0,
                        },
                        processing_time_seconds=segment_time,
                        peak_memory_usage=mx.get_peak_memory() / 1e9,
                    )
                    
                    if verbose:
                        rtf = segment_time / audio_duration if audio_duration > 0 else 0
                        print(f"Chunk {yielded_chunks_count}: {len(audio_chunk)} samples, "
                              f"{audio_duration:.3f}s, RTF: {rtf:.2f}x")
                
                # Update counters
                yielded_chunks_count += 1
                accumulated_tokens = accumulated_tokens[streaming_chunk_tokens:]
                segment_start_time = time.time()
        
        # Final chunk with remaining tokens
        if len(accumulated_tokens) > 0:
            audio_chunk = self._decode_partial_tokens(accumulated_tokens)
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Save final chunk to debug buffer
                if debug_save_wav:
                    debug_audio_chunks.append(audio_chunk.copy())
                    debug_total_samples += len(audio_chunk)
                    if verbose:
                        print(f"�� Debug: Added final chunk with {len(audio_chunk)} samples")
                
                segment_time = time.time() - segment_start_time
                audio_duration = len(audio_chunk) / self.sample_rate
                
                yield GenerationResult(
                    audio=audio_chunk,
                    samples=len(audio_chunk),
                    sample_rate=self.sample_rate,
                    segment_idx=yielded_chunks_count,
                    token_count=len(accumulated_tokens),
                    audio_duration=self._format_duration(audio_duration),
                    real_time_factor=segment_time / audio_duration if audio_duration > 0 else 0,
                    prompt={
                        "tokens": len(accumulated_tokens),
                        "tokens-per-sec": len(accumulated_tokens) / segment_time if segment_time > 0 else 0,
                    },
                    audio_samples={
                        "samples": len(audio_chunk),
                        "samples-per-sec": len(audio_chunk) / segment_time if segment_time > 0 else 0,
                    },
                    processing_time_seconds=segment_time,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                )
                
                if verbose:
                    rtf = segment_time / audio_duration if audio_duration > 0 else 0
                    print(f"Final chunk {yielded_chunks_count}: {len(audio_chunk)} samples, "
                          f"{audio_duration:.3f}s, RTF: {rtf:.2f}x")
        
        total_time = time.time() - total_start_time
        if verbose:
            print(f"Total generation time: {total_time:.2f}s, "
                  f"Generated {yielded_chunks_count + 1} chunks")
        
        # ← Отладочное сохранение: конкатенируем все чанки и сохраняем в WAV
        if debug_save_wav and debug_audio_chunks:
            try:
                self._save_debug_wav(debug_audio_chunks, debug_wav_path, verbose)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Warning: Failed to save debug WAV: {e}")
    
    def _save_debug_wav(self, audio_chunks: List[mx.array], wav_path: str, verbose: bool = False):
        """Save concatenated audio chunks to WAV file for debugging"""
        try:
            import soundfile as sf
            
            # Конвертируем MLX массивы в numpy и конкатенируем
            numpy_chunks = [np.array(chunk, dtype=np.float32) for chunk in audio_chunks]
            concatenated_audio = np.concatenate(numpy_chunks)
            
            # Создаем директорию если нужно
            os.makedirs(os.path.dirname(wav_path) if os.path.dirname(wav_path) else '.', exist_ok=True)
            
            # Сохраняем как есть (без преобразований)
            sf.write(wav_path, concatenated_audio, self.sample_rate, subtype='FLOAT')
            
            if verbose:
                duration = len(concatenated_audio) / self.sample_rate
                print(f"�� Debug WAV saved: {wav_path}")
                print(f"🔍 Total samples: {len(concatenated_audio)}")
                print(f"🔍 Duration: {duration:.3f}s")
                print(f"�� Sample rate: {self.sample_rate} Hz")
                print(f"🔍 Audio range: {np.min(concatenated_audio):.6f} to {np.max(concatenated_audio):.6f}")
                
        except ImportError:
            # Fallback без soundfile - используем scipy
            try:
                from scipy.io import wavfile
                
                numpy_chunks = [np.array(chunk, dtype=np.float32) for chunk in audio_chunks]
                concatenated_audio = np.concatenate(numpy_chunks)
                
                # Нормализуем для WAV (float32 -> int16)
                audio_int16 = np.int16(concatenated_audio * 32767)
                
                os.makedirs(os.path.dirname(wav_path) if os.path.dirname(wav_path) else '.', exist_ok=True)
                wavfile.write(wav_path, self.sample_rate, audio_int16)
                
                if verbose:
                    duration = len(concatenated_audio) / self.sample_rate
                    print(f"🔍 Debug WAV saved (scipy): {wav_path}")
                    print(f"🔍 Total samples: {len(concatenated_audio)}")
                    print(f"🔍 Duration: {duration:.3f}s")
                    print(f"�� Sample rate: {self.sample_rate} Hz")
                    print(f"�� Note: Converted to int16 for WAV compatibility")
                    
            except ImportError:
                if verbose:
                    print("⚠️ Warning: Neither soundfile nor scipy available. Cannot save debug WAV.")
                    print("   Install: pip install soundfile (preferred) or scipy")
    
    def _decode_partial_tokens(self, tokens: List[int]) -> Optional[mx.array]:
        """Decode partial list of tokens to audio"""
        try:
            # Filter out special tokens and prepare for decoding
            filtered_tokens = [t - 128266 for t in tokens if t >= 128266]
            
            # Check we have enough tokens for decoding (multiple of 7)
            if len(filtered_tokens) < 7:
                return None
                
            # Trim to multiple of 7
            usable_length = (len(filtered_tokens) // 7) * 7
            code_list = filtered_tokens[:usable_length]
            
            if len(code_list) == 0:
                return None
                
            # Decode to audio
            audio = decode_audio_from_codes(code_list)
            return audio
            
        except Exception as e:
            print(f"Warning: Failed to decode partial tokens: {e}")
            return None
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"