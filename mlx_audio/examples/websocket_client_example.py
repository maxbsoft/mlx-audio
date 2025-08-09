"""
WebSocket client example for Orpheus streaming TTS.
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import soundfile as sf
import time
from typing import List


class OrpheusStreamingClient:
    """WebSocket client for Orpheus streaming"""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.audio_chunks: List[np.ndarray] = []
        self.session_id = None
        
    async def connect_and_stream(self, text: str, **kwargs):
        """Connect to server and stream TTS"""
        
        print(f"🔗 Connecting to {self.uri}")
        
        try:
            async with websockets.connect(self.uri) as websocket:
                # Send streaming request
                request = {
                    'type': 'start_orpheus_stream',
                    'text': text,
                    'model_path': kwargs.get('model_path', 'mlx-community/orpheus-3b-0.1-ft-4bit'),
                    'voice': kwargs.get('voice', 'af_heart'),
                    'chunk_tokens': kwargs.get('chunk_tokens', 21),
                    'temperature': kwargs.get('temperature', 0.6),
                    'verbose': kwargs.get('verbose', True)
                }
                
                print(f"📤 Sending request: {text[:50]}...")
                await websocket.send(json.dumps(request))
                
                # Process responses
                start_time = time.time()
                first_chunk_time = None
                
                async for message in websocket:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'stream_started':
                        self.session_id = data.get('session_id')
                        print(f"✅ Stream started with session {self.session_id}")
                        
                    elif message_type == 'audio_chunk':
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time
                            print(f"⚡ First chunk latency: {first_chunk_time:.3f}s")
                        
                        await self._handle_audio_chunk(data)
                        
                    elif message_type == 'stream_complete':
                        stats = data.get('stats', {})
                        print(f"🎉 Stream completed!")
                        print(f"   📊 Stats: {stats}")
                        break
                        
                    elif message_type == 'error':
                        print(f"❌ Server error: {data.get('message')}")
                        break
                
                return self.audio_chunks
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return []
    
    async def _handle_audio_chunk(self, data: dict):
        """Handle incoming audio chunk"""
        
        try:
            # Decode audio data
            audio_b64 = data.get('audio', '')
            audio_bytes = base64.b64decode(audio_b64)
            
            shape = data.get('shape', [])
            dtype = data.get('dtype', 'float32')
            
            # Convert back to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=dtype).reshape(shape)
            self.audio_chunks.append(audio_array)
            
            # Print chunk info
            chunk_index = data.get('chunk_index', 0)
            metadata = data.get('metadata', {})
            duration_ms = metadata.get('duration_ms', 0)
            elapsed_time = metadata.get('elapsed_time', 0)
            
            print(f"🔊 Chunk {chunk_index}: {len(audio_array)} samples, "
                  f"{duration_ms:.1f}ms, elapsed: {elapsed_time:.3f}s")
            
            # Here you could:
            # - Play audio immediately
            # - Stream to another service
            # - Process in real-time
            
        except Exception as e:
            print(f"❌ Error handling audio chunk: {e}")


async def example_websocket_streaming():
    """Basic WebSocket streaming example"""
    
    print("🌐 WebSocket Streaming Example")
    print("-" * 40)
    
    client = OrpheusStreamingClient()
    
    text = """
    This is a demonstration of WebSocket-based streaming TTS.
    The audio is generated on the server and streamed to the client in real-time.
    This enables applications like real-time voice assistants and interactive audio.
    """
    
    chunks = await client.connect_and_stream(
        text=text.strip(),
        voice="af_heart",
        chunk_tokens=21,
        temperature=0.6
    )
    
    if chunks:
        # Combine all chunks
        full_audio = np.concatenate(chunks)
        
        # Save to file
        output_file = "websocket_streaming_output.wav"
        sf.write(output_file, full_audio, 24000)
        
        duration = len(full_audio) / 24000
        
        print(f"\n📊 Results:")
        print(f"   🔢 Received chunks: {len(chunks)}")
        print(f"   📏 Total samples: {len(full_audio)}")
        print(f"   📏 Audio duration: {duration:.3f}s")
        print(f"   ✅ Saved to: {output_file}")
    else:
        print("❌ No audio received")


async def example_multiple_sessions():
    """Multiple concurrent streaming sessions"""
    
    print("\n🌐 Multiple Sessions Example")
    print("-" * 40)
    
    texts = [
        "First session: Quick brown fox jumps over the lazy dog.",
        "Second session: The weather is nice today.",
        "Third session: Machine learning and artificial intelligence."
    ]
    
    # Start multiple sessions concurrently
    tasks = []
    clients = []
    
    for i, text in enumerate(texts):
        client = OrpheusStreamingClient()
        clients.append(client)
        
        task = asyncio.create_task(client.connect_and_stream(
            text=text,
            voice="af_heart",
            chunk_tokens=14,  # Smaller chunks for faster response
            temperature=0.6
        ))
        tasks.append(task)
        
        print(f"🚀 Started session {i+1}: {text[:30]}...")
    
    # Wait for all sessions to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, (client, result) in enumerate(zip(clients, results)):
        if isinstance(result, Exception):
            print(f"❌ Session {i+1} failed: {result}")
        elif result:
            audio = np.concatenate(result)
            output_file = f"websocket_session_{i+1}.wav"
            sf.write(output_file, audio, 24000)
            
            duration = len(audio) / 24000
            print(f"✅ Session {i+1}: {len(result)} chunks, "
                  f"{duration:.3f}s, saved to {output_file}")


async def example_real_time_interaction():
    """Simulate real-time interaction"""
    
    print("\n🌐 Real-Time Interaction Example")
    print("-" * 40)
    
    client = OrpheusStreamingClient()
    
    # Simulate conversation
    conversation = [
        "Hello! How can I help you today?",
        "I can provide information on various topics.",
        "Feel free to ask me anything you'd like to know.",
        "Thank you for using our streaming TTS service!"
    ]
    
    all_audio = []
    
    for i, text in enumerate(conversation):
        print(f"\n💬 Turn {i+1}: {text}")
        
        chunks = await client.connect_and_stream(
            text=text,
            voice="af_heart",
            chunk_tokens=7,  # Very small chunks for responsiveness
            temperature=0.6
        )
        
        if chunks:
            turn_audio = np.concatenate(chunks)
            all_audio.append(turn_audio)
            
            # Add pause between turns
            pause = np.zeros(int(24000 * 0.5))  # 0.5 second pause
            all_audio.append(pause)
            
            print(f"   ✅ Generated {len(chunks)} chunks")
        
        # Simulate thinking time
        await asyncio.sleep(0.5)
    
    if all_audio:
        full_conversation = np.concatenate(all_audio)
        output_file = "websocket_conversation.wav"
        sf.write(output_file, full_conversation, 24000)
        
        duration = len(full_conversation) / 24000
        print(f"\n📊 Conversation Results:")
        print(f"   🗣️  Turns: {len(conversation)}")
        print(f"   📏 Total duration: {duration:.3f}s")
        print(f"   ✅ Saved to: {output_file}")


def start_server_instructions():
    """Print instructions for starting the server"""
    
    print("📋 To run these examples, first start the streaming server:")
    print("   python -m mlx_audio.orpheus_streaming_server")
    print("   (or)")
    print("   python mlx_audio/orpheus_streaming_server.py")
    print("\n🌐 Server will be available at ws://localhost:8765")
    print("=" * 50)


async def main():
    """Run WebSocket client examples"""
    
    print("🌐 Orpheus WebSocket Client Examples")
    print("=" * 50)
    
    start_server_instructions()
    
    try:
        await example_websocket_streaming()
        await example_multiple_sessions()
        await example_real_time_interaction()
        
        print("\n🎉 All WebSocket examples completed!")
        print("\n📁 Generated files:")
        print("   - websocket_streaming_output.wav")
        print("   - websocket_session_1.wav")
        print("   - websocket_session_2.wav")
        print("   - websocket_session_3.wav")
        print("   - websocket_conversation.wav")
        
    except ConnectionRefusedError:
        print("\n❌ Connection refused!")
        print("🚨 Make sure the streaming server is running:")
        print("   python -m mlx_audio.orpheus_streaming_server")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
