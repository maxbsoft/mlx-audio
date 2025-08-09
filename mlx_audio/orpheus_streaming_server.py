"""
WebSocket streaming server for Orpheus TTS with real-time audio delivery.
"""

import asyncio
import websockets
import json
import numpy as np
import base64
import time
import threading
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrpheusStreamingServer:
    """WebSocket server for real-time Orpheus TTS streaming"""
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 model_path: str = "mlx-community/orpheus-3b-0.1-ft-4bit",
                 preload_model: bool = True):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.preload_model = preload_model
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.main_loop = None
        self.model = None  # Pre-loaded model
        self.snac_model = None  # Pre-loaded SNAC codec
        self._model_lock = threading.Lock()  # Thread-safe model access
    
    def load_model(self):
        """Load the TTS model and SNAC codec into memory"""
        try:
            logger.info(f"🚀 Loading model: {self.model_path}")
            start_time = time.time()
            
            # Import with proper path handling
            import sys
            import os
            current_dir = os.path.dirname(__file__)
            sys.path.insert(0, current_dir)
            
            try:
                from tts.utils import load_model
                from mlx_audio.codec.models.snac import SNAC
            except ImportError:
                from mlx_audio.tts.utils import load_model
                from mlx_audio.codec.models.snac import SNAC
            
            with self._model_lock:
                # Load main TTS model
                logger.info(f"📦 Loading TTS model...")
                self.model = load_model(model_path=self.model_path)
                
                # Load SNAC codec
                logger.info(f"🎵 Loading SNAC codec...")
                self.snac_model = SNAC.from_pretrained("mlx-community/snac_24khz").eval()
                
                # Patch the global snac_model in llama module to use our preloaded one
                try:
                    from tts.models.llama import llama
                    llama.snac_model = self.snac_model
                except ImportError:
                    from mlx_audio.tts.models.llama import llama
                    llama.snac_model = self.snac_model
                
                logger.info(f"🔄 SNAC codec patched in llama module")
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"📦 Model ready for streaming requests")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
            self.snac_model = None
            raise
    
    def get_model(self):
        """Get the loaded model (thread-safe)"""
        with self._model_lock:
            return self.model
        
    async def handle_websocket(self, websocket):
        """Handle WebSocket connection and streaming requests"""
        
        client_id = id(websocket)
        logger.info(f"New client connected: {client_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(websocket, client_id, data)
                    
                except json.JSONDecodeError as e:
                    await self.send_error(websocket, f"Invalid JSON: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            # Clean up active session
            if str(client_id) in self.active_sessions:
                session = self.active_sessions[str(client_id)]
                if 'stop_event' in session:
                    session['stop_event'].set()
                del self.active_sessions[str(client_id)]
    
    async def process_message(self, websocket, client_id: int, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        
        message_type = data.get('type')
        
        if message_type == 'start_orpheus_stream':
            await self.start_streaming(websocket, client_id, data)
            
        elif message_type == 'stop_stream':
            await self.stop_streaming(websocket, client_id)
            
        elif message_type == 'get_status':
            await self.send_status(websocket, client_id)
            
        else:
            await self.send_error(websocket, f"Unknown message type: {message_type}")
    
    async def start_streaming(self, websocket, client_id: int, data: Dict[str, Any]):
        """Start TTS streaming session"""
        
        # Extract parameters
        text = data.get('text', '')
        if not text.strip():
            await self.send_error(websocket, "Text is required")
            return
        
        model_path = data.get('model_path', self.model_path)
        voice = data.get('voice', 'tara')
        chunk_tokens = data.get('chunk_tokens', 21)
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.8)
        max_tokens = data.get('max_tokens', 1200)
        ultra_low_latency = data.get('ultra_low_latency', True)
        enable_processing = data.get('enable_processing', False)
        verbose = data.get('verbose', False)
        repetition_penalty = data.get('repetition_penalty', 1.3)
        
        logger.info(f"Starting stream for client {client_id}: '{text[:50]}...'")
        
        # Stop any existing session
        await self.stop_streaming(websocket, client_id)
        
        # Create session tracking
        stop_event = threading.Event()
        session = {
            'stop_event': stop_event,
            'start_time': time.time(),
            'chunk_count': 0,
            'total_samples': 0
        }
        self.active_sessions[str(client_id)] = session
        
        # Send start confirmation
        await websocket.send(json.dumps({
            'type': 'stream_started',
            'session_id': str(client_id),
            'parameters': {
                'model_path': model_path,
                'voice': voice,
                'chunk_tokens': chunk_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'ultra_low_latency': ultra_low_latency,
                'enable_processing': enable_processing,
                'repetition_penalty': repetition_penalty
            }
        }))
        
        # Start streaming in background thread
        def streaming_worker():
            """Background worker for audio generation"""
            try:
                # Import with proper path handling
                import sys
                import os
                current_dir = os.path.dirname(__file__)
                sys.path.insert(0, current_dir)
                
                try:
                    from tts.generate import generate_audio_streaming
                except ImportError:
                    # Fallback for different import scenarios
                    from mlx_audio.tts.generate import generate_audio_streaming
                
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Get pre-loaded model or use model_path
                model = self.get_model()
                if model is not None:
                    logger.info(f"Using pre-loaded model for client {client_id}")
                else:
                    logger.info(f"Model not pre-loaded, will load on-demand for client {client_id}")
                
                # Use pre-loaded model if available, otherwise use model_path
                streaming_kwargs = {
                    'text': text,
                    'voice': voice,
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_tokens,
                    'streaming_chunk_tokens': chunk_tokens,
                    'ultra_low_latency': ultra_low_latency,
                    'verbose': verbose,
                    'repetition_penalty': repetition_penalty
                }
                
                if model is not None:
                    # Use pre-loaded model
                    streaming_kwargs['model'] = model
                    streaming_kwargs['model_path'] = model_path  # Keep model_path for compatibility
                else:
                    # Load model on-demand
                    streaming_kwargs['model_path'] = model_path
                
                for audio_chunk in generate_audio_streaming(**streaming_kwargs):
                    # Check if streaming should stop
                    if stop_event.is_set():
                        break
                    
                    # Send audio chunk using thread-safe method
                    asyncio.run_coroutine_threadsafe(
                        self.send_audio_chunk(websocket, client_id, audio_chunk),
                        self.main_loop
                    )
                    
                    # Update session stats
                    session['chunk_count'] += 1
                    session['total_samples'] += len(audio_chunk)
                
                # Send completion message
                if not stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(
                        self.send_completion(websocket, client_id),
                        self.main_loop
                    )
                    
            except Exception as e:
                logger.error(f"Streaming error for client {client_id}: {e}")
                asyncio.run_coroutine_threadsafe(
                    self.send_error(websocket, str(e)),
                    self.main_loop
                )
            finally:
                # Clean up session
                if str(client_id) in self.active_sessions:
                    del self.active_sessions[str(client_id)]
        
        # Start worker thread
        thread = threading.Thread(target=streaming_worker, daemon=True)
        thread.start()
    
    async def stop_streaming(self, websocket, client_id: int):
        """Stop active streaming session"""
        
        session_key = str(client_id)
        if session_key in self.active_sessions:
            session = self.active_sessions[session_key]
            session['stop_event'].set()
            
            await websocket.send(json.dumps({
                'type': 'stream_stopped',
                'session_id': session_key
            }))
            
            logger.info(f"Stopped streaming for client {client_id}")
    
    async def send_audio_chunk(self, websocket, client_id: int, audio_chunk: np.ndarray):
        """Send audio chunk to client"""
        
        try:
            # Convert audio to base64
            audio_bytes = audio_chunk.astype(np.float32).tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            session = self.active_sessions.get(str(client_id), {})
            
            message = {
                'type': 'audio_chunk',
                'session_id': str(client_id),
                'audio': audio_b64,
                'sample_rate': 24000,
                'dtype': 'float32',
                'shape': list(audio_chunk.shape),
                'chunk_index': session.get('chunk_count', 0),
                'timestamp': time.time(),
                'metadata': {
                    'duration_ms': len(audio_chunk) / 24000 * 1000,
                    'total_samples': session.get('total_samples', 0) + len(audio_chunk),
                    'elapsed_time': time.time() - session.get('start_time', time.time())
                }
            }
            
            await websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending audio chunk to client {client_id}: {e}")
    
    async def send_completion(self, websocket, client_id: int):
        """Send stream completion message"""
        
        session = self.active_sessions.get(str(client_id), {})
        total_time = time.time() - session.get('start_time', time.time())
        
        message = {
            'type': 'stream_complete',
            'session_id': str(client_id),
            'stats': {
                'total_chunks': session.get('chunk_count', 0),
                'total_samples': session.get('total_samples', 0),
                'total_duration': total_time,
                'audio_duration': session.get('total_samples', 0) / 24000
            }
        }
        
        await websocket.send(json.dumps(message))
        logger.info(f"Completed streaming for client {client_id}")
    
    async def send_status(self, websocket, client_id: int):
        """Send current session status"""
        
        session = self.active_sessions.get(str(client_id))
        
        if session:
            elapsed = time.time() - session['start_time']
            status = {
                'type': 'status',
                'session_id': str(client_id),
                'active': True,
                'chunk_count': session['chunk_count'],
                'total_samples': session['total_samples'],
                'elapsed_time': elapsed,
                'audio_duration': session['total_samples'] / 24000
            }
        else:
            status = {
                'type': 'status',
                'session_id': str(client_id),
                'active': False
            }
        
        await websocket.send(json.dumps(status))
    
    async def send_error(self, websocket, error_message: str):
        """Send error message to client"""
        
        message = {
            'type': 'error',
            'message': error_message,
            'timestamp': time.time()
        }
        
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    async def start_server_async(self):
        """Start the WebSocket server asynchronously"""
        
        # Store the main event loop for thread-safe communication
        self.main_loop = asyncio.get_running_loop()
        
        logger.info(f"🚀 Starting Orpheus Streaming Server on ws://{self.host}:{self.port}")
        
        # Pre-load model if requested
        if self.preload_model:
            logger.info(f"🔄 Pre-loading model: {self.model_path}")
            try:
                # Load model in a separate thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.load_model)
                logger.info(f"🎯 Model pre-loaded and ready!")
            except Exception as e:
                logger.error(f"❌ Failed to pre-load model: {e}")
                logger.info(f"⚠️  Server will continue without pre-loaded model")
        else:
            logger.info(f"⚠️  Model pre-loading disabled - models will load on first request")
        
        logger.info(f"📖 Usage examples:")
        logger.info(f"   - Connect to ws://{self.host}:{self.port}")
        logger.info(f"   - Send: {{'type': 'start_orpheus_stream', 'text': 'Hello world!'}}")
        logger.info(f"   - Receive: audio chunks as base64 encoded data")
        
        # Start the WebSocket server
        async with websockets.serve(
            self.handle_websocket, 
            self.host, 
            self.port,
            max_size=10**7,  # 10MB max message size
            ping_interval=20,
            ping_timeout=10
        ):
            logger.info(f"✅ Server is running on ws://{self.host}:{self.port}")
            # Keep the server running
            await asyncio.Future()  # Run forever
    
    def start_server(self):
        """Start the WebSocket server"""
        try:
            asyncio.run(self.start_server_async())
        except KeyboardInterrupt:
            logger.info("👋 Server stopped by user")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            raise


def main():
    """Main entry point for the streaming server"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Orpheus TTS Streaming Server")
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--model', default='mlx-community/orpheus-3b-0.1-ft-4bit', 
                        help='Model path to preload')
    parser.add_argument('--no-preload', action='store_true', 
                        help='Disable model preloading (load on first request)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = OrpheusStreamingServer(
        host=args.host, 
        port=args.port,
        model_path=args.model,
        preload_model=not args.no_preload
    )
    server.start_server()


if __name__ == "__main__":
    main()
