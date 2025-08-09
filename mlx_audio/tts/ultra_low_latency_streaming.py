"""
Ultra low-latency streaming implementation for Orpheus
"""

import time
import numpy as np
from typing import List, Optional, Generator
import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler


def generate_ultra_low_latency_streaming(
    model,
    text: str,
    voice: str = "af_heart",
    temperature: float = 0.6,
    top_p: float = 0.8,
    max_tokens: int = 1200,
    min_tokens_for_first_chunk: int = 7,  # Minimum SNAC requirement
    subsequent_chunk_tokens: int = 14,    # Size for subsequent chunks
    verbose: bool = False,
    ref_audio: Optional[mx.array] = None,
    ref_text: Optional[str] = None,
    **kwargs
) -> Generator[np.ndarray, None, None]:
    """
    Ultra low-latency streaming with aggressive early decoding
    
    Strategy:
    1. Decode first chunk as soon as we have 7 tokens (minimum for SNAC)
    2. Continue with regular chunk sizes for subsequent chunks
    3. Use incremental decoding to avoid re-processing
    """
    
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
    
    # Streaming state
    all_tokens = []
    audio_tokens_found = False
    audio_start_found = False
    first_chunk_generated = False
    chunk_idx = 0
    last_decoded_audio_tokens = 0
    
    if verbose:
        print(f"🚀 Ultra low-latency streaming: first={min_tokens_for_first_chunk}, subsequent={subsequent_chunk_tokens}")
    
    generation_start = time.time()
    first_audio_token_time = None
    first_chunk_time = None
    
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
        if i % 50 == 0:
            mx.clear_cache()
            
        # Check for end of generation
        if next_token == 128258:  # EOS token
            if verbose:
                print(f"🏁 EOS token reached after {len(all_tokens)} tokens")
            break
        
        # Detect audio start token
        if next_token == 128257:  # Audio start token
            audio_start_found = True
            if first_audio_token_time is None:
                first_audio_token_time = time.time() - generation_start
                if verbose:
                    print(f"🎵 Audio tokens started at: {first_audio_token_time:.3f}s")
            continue
            
        # Only process audio tokens (>= 128266)
        if next_token >= 128266 and audio_start_found:
            if not audio_tokens_found:
                audio_tokens_found = True
                if verbose:
                    print(f"🎯 First audio token received")
            
            # Count audio tokens so far
            audio_tokens = [t for t in all_tokens if t >= 128266 and t != 128258]
            
            # Strategy: Aggressive first chunk, then regular chunks
            if not first_chunk_generated:
                # First chunk: decode as soon as we have minimum tokens
                if len(audio_tokens) >= min_tokens_for_first_chunk:
                    audio_chunk = decode_audio_tokens_incremental(
                        audio_tokens, 
                        last_decoded_audio_tokens,
                        min_tokens_for_first_chunk
                    )
                    
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - generation_start
                            
                        audio_flat = np.array(audio_chunk, dtype=np.float32).flatten()
                        
                        if verbose:
                            duration = len(audio_flat) / 24000
                            print(f"🔊 FIRST chunk {chunk_idx}: {len(audio_flat)} samples, {duration:.3f}s, latency: {first_chunk_time:.3f}s")
                        
                        yield audio_flat
                        chunk_idx += 1
                        first_chunk_generated = True
                        last_decoded_audio_tokens = len(audio_tokens)
                        
            else:
                # Subsequent chunks: use regular chunk size
                if len(audio_tokens) >= last_decoded_audio_tokens + subsequent_chunk_tokens:
                    audio_chunk = decode_audio_tokens_incremental(
                        audio_tokens,
                        last_decoded_audio_tokens, 
                        subsequent_chunk_tokens
                    )
                    
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        audio_flat = np.array(audio_chunk, dtype=np.float32).flatten()
                        
                        if verbose:
                            duration = len(audio_flat) / 24000
                            print(f"🔊 Chunk {chunk_idx}: {len(audio_flat)} samples, {duration:.3f}s")
                        
                        yield audio_flat
                        chunk_idx += 1
                        last_decoded_audio_tokens = len(audio_tokens)
    
    # Final chunk with any remaining tokens
    if audio_tokens_found:
        audio_tokens = [t for t in all_tokens if t >= 128266 and t != 128258]
        if len(audio_tokens) > last_decoded_audio_tokens:
            final_audio = decode_audio_tokens_final(all_tokens)
            if final_audio is not None and len(final_audio) > 0:
                # Only yield the new part
                if len(final_audio) > last_decoded_audio_tokens * 320:  # Rough estimate of samples per token
                    new_audio = final_audio[last_decoded_audio_tokens * 320:]
                    audio_flat = np.array(new_audio, dtype=np.float32).flatten()
                    
                    if verbose:
                        duration = len(audio_flat) / 24000
                        print(f"🔊 FINAL chunk {chunk_idx}: {len(audio_flat)} samples, {duration:.3f}s")
                    
                    yield audio_flat


def decode_audio_tokens_incremental(audio_tokens: List[int], skip_tokens: int, chunk_size: int) -> Optional[mx.array]:
    """Decode audio tokens incrementally"""
    try:
        from .models.llama.llama import decode_audio_from_codes
        
        # Take only the tokens we want to decode
        tokens_to_decode = audio_tokens[skip_tokens:skip_tokens + chunk_size]
        
        # Ensure we have enough and it's multiple of 7
        if len(tokens_to_decode) < 7:
            return None
            
        # Trim to multiple of 7
        usable_length = (len(tokens_to_decode) // 7) * 7
        if usable_length == 0:
            return None
            
        code_list = [t - 128266 for t in tokens_to_decode[:usable_length]]
        
        # Decode to audio
        audio = decode_audio_from_codes(code_list)
        return audio
        
    except Exception as e:
        print(f"Warning: Failed to decode incremental tokens: {e}")
        return None


def decode_audio_tokens_final(all_tokens: List[int]) -> Optional[mx.array]:
    """Decode final tokens using full model logic"""
    try:
        from .models.llama.llama import decode_audio_from_codes
        import mlx.core as mx
        
        # Convert to mx.array and use the model's parse_output logic
        input_ids = mx.array([all_tokens])
        
        # Replicate the parse_output logic from the original model
        token_to_find = 128257  # Audio start token
        token_to_remove = 128258  # EOS token
        
        # Find audio start token
        mask = input_ids == token_to_find
        indices = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    indices.append((i, j))
        
        if len(indices) > 0:
            # Get audio tokens after the last audio start token
            last_start_idx = indices[-1][1]
            cropped_tensor = input_ids[:, last_start_idx + 1:]
        else:
            cropped_tensor = input_ids
        
        # Remove EOS tokens and process
        processed_tokens = []
        for token in cropped_tensor[0].tolist():
            if token != token_to_remove and token >= 128266:
                processed_tokens.append(token - 128266)
        
        # Ensure multiple of 7
        usable_length = (len(processed_tokens) // 7) * 7
        if usable_length > 0:
            code_list = processed_tokens[:usable_length]
            audio = decode_audio_from_codes(code_list)
            return audio
        
        return None
        
    except Exception as e:
        print(f"Warning: Failed to decode final tokens: {e}")
        return None
