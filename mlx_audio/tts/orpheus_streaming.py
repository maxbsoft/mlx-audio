"""
Simple Orpheus streaming implementation using existing model structure.
"""

import time
from typing import Generator, Optional
import mlx.core as mx
import numpy as np
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from .models.base import GenerationResult


def generate_audio_streaming_simple(
    model,
    text: str,
    voice: str,
    temperature: float = 0.6,
    top_p: float = 0.8,
    max_tokens: int = 1200,
    streaming_chunk_tokens: int = 21,
    verbose: bool = False,
    ref_audio: Optional[mx.array] = None,
    ref_text: Optional[str] = None,
    **kwargs,
) -> Generator[np.ndarray, None, None]:
    """
    Simple streaming implementation for Orpheus models
    """
    
    # Ensure chunk size is multiple of 7
    if streaming_chunk_tokens % 7 != 0:
        streaming_chunk_tokens = ((streaming_chunk_tokens // 7) + 1) * 7
        if verbose:
            print(f"Adjusted chunk size to {streaming_chunk_tokens} tokens (multiple of 7)")
    
    # Prepare input_ids using the model's existing method
    input_ids, _ = model.prepare_input_ids([text], voice, ref_audio, ref_text)
    
    sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
    logits_processors = make_logits_processors(
        kwargs.get("logit_bias", None),
        kwargs.get("repetition_penalty", 1.3),
        kwargs.get("repetition_context_size", 20),
    )
    
    # Streaming accumulators
    all_tokens = []  # Keep all tokens
    chunk_idx = 0
    last_decoded_length = 0
    
    if verbose:
        print(f"Starting streaming generation with {streaming_chunk_tokens} tokens per chunk")
    
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
                print("EOS token reached, finishing generation")
            break
        
        # Try to decode when we have enough new tokens
        if len(all_tokens) >= last_decoded_length + streaming_chunk_tokens:
            # Use all tokens accumulated so far for better context
            audio_chunk = decode_partial_tokens_with_context(all_tokens, last_decoded_length)
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Flatten the audio chunk to 1D
                audio_flat = np.array(audio_chunk, dtype=np.float32).flatten()
                
                if verbose:
                    duration = len(audio_flat) / 24000
                    print(f"🔊 Chunk {chunk_idx}: {len(audio_flat)} samples, {duration:.3f}s")
                
                yield audio_flat
                chunk_idx += 1
                
                # Update the last decoded position
                last_decoded_length = len(all_tokens)
    
    # Final chunk with any remaining tokens
    if len(all_tokens) > last_decoded_length:
        final_audio = decode_final_tokens(all_tokens)
        if final_audio is not None and len(final_audio) > 0:
            # Flatten the final audio chunk to 1D
            audio_flat = np.array(final_audio, dtype=np.float32).flatten()
            
            if verbose:
                duration = len(audio_flat) / 24000
                print(f"🔊 Final chunk {chunk_idx}: {len(audio_flat)} samples, {duration:.3f}s")
            
            yield audio_flat


def decode_partial_tokens_with_context(all_tokens, last_decoded_length):
    """Decode tokens with full context, returning only new audio"""
    try:
        # Import here to avoid circular imports
        from .models.llama.llama import decode_audio_from_codes
        
        # Filter audio tokens from all accumulated tokens
        filtered_tokens = []
        for t in all_tokens:
            if t >= 128266 and t != 128258:  # Audio tokens, excluding EOS
                filtered_tokens.append(t - 128266)
        
        # Need at least 7 tokens for SNAC decoding
        if len(filtered_tokens) < 7:
            return None
            
        # Trim to multiple of 7
        usable_length = (len(filtered_tokens) // 7) * 7
        code_list = filtered_tokens[:usable_length]
        
        if len(code_list) == 0:
            return None
            
        # Decode full audio
        full_audio = decode_audio_from_codes(code_list)
        
        # Calculate how much audio we've already yielded
        # Estimate based on token ratio
        if last_decoded_length > 0:
            audio_ratio = len(filtered_tokens) / len(all_tokens)
            last_audio_length = int(last_decoded_length * audio_ratio * (len(full_audio) / len(filtered_tokens)))
            
            # Return only new audio
            if len(full_audio) > last_audio_length:
                return full_audio[last_audio_length:]
            else:
                return None
        else:
            # First chunk, return beginning portion
            chunk_ratio = min(0.3, len(filtered_tokens) / 100)  # Conservative first chunk
            chunk_length = int(len(full_audio) * chunk_ratio)
            return full_audio[:chunk_length] if chunk_length > 0 else None
        
    except Exception as e:
        print(f"Warning: Failed to decode tokens with context: {e}")
        return None


def decode_final_tokens(all_tokens):
    """Decode final complete sequence using model's parse_output logic"""
    try:
        # Import here to avoid circular imports  
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
        
        # Remove EOS tokens
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


def decode_partial_tokens(tokens):
    """Legacy function - kept for compatibility"""
    return decode_final_tokens(tokens)
