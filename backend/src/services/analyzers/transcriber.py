import threading
import queue
import os
import logging
import numpy as np
from typing import Optional
from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters
)

from src.schemas.audio_metrics import TimestampsSegment

load_dotenv()
logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.api_key = os.getenv("AssemblyAI_API_KEY")
        self._enabled = bool(self.api_key)
        
        if not self._enabled:
            logger.warning("AssemblyAI_API_KEY is not set; transcription will be disabled.")
        else:
            logger.debug(f"AssemblyAI API key found (starts with: {self.api_key[:8]}...)")
        
        self._audio_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._running = True
        self._connected = False
        self.client = None
        self.processed_samples = 0
        
        if self._enabled:
            self._thread = threading.Thread(target=self._client_loop, daemon=True)
            self._thread.start()
        else:
            self._thread = None

    def process(self, new_chunk: np.ndarray) -> Optional[TimestampsSegment]:
        if len(new_chunk) == 0 or not self._enabled:
            return None
        
        # Calculate start time before incrementing samples
        chunk_start = self.processed_samples / self.sample_rate
        
        # Convert float32 [-1, 1] to int16 bytes
        pcm_data = (new_chunk * 32767).astype(np.int16).tobytes()
        
        self._audio_queue.put((pcm_data, chunk_start))
        self.processed_samples += len(new_chunk)
        
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def get_result(self) -> Optional[TimestampsSegment]:
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def _client_loop(self):
        logger.debug("Transcriber client loop started")
        
        try:
            self.client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com",
                )
            )

            def on_turn(client, event):
                text = event.transcript
                
                if text:
                    is_final = event.end_of_turn
                    
                    if is_final:
                        logger.debug(f"Received final transcript: {text}")
                    else:
                        logger.debug(f"Received partial transcript: {text}")
                    
                    current_time = self.processed_samples / self.sample_rate
                    
                    segment = TimestampsSegment(
                        text=text,
                        start_time=max(0.0, current_time - 2.0),
                        end_time=current_time,
                        is_final=is_final,
                    )
                    self._result_queue.put(segment)

            def on_error(client, error):
                logger.error(f"AssemblyAI Error: {error}")

            def on_close(client, code, reason):
                logger.debug(f"AssemblyAI Session Closed: {code} - {reason}")
                self._connected = False

            self.client.on(StreamingEvents.Turn, on_turn)
            self.client.on(StreamingEvents.Error, on_error)

            logger.debug("Connecting to AssemblyAI...")
            self.client.connect(
                StreamingParameters(
                    sample_rate=self.sample_rate,
                    encoding="pcm_s16le", 
                    format_turns=True 
                )
            )
            logger.debug("Connected to AssemblyAI successfully!")
            self._connected = True

            logger.debug("Starting AssemblyAI stream...")
            self.client.stream(self._audio_generator())
            logger.debug("AssemblyAI stream finished normally")
            
        except TimeoutError as e:
            logger.error(f"AssemblyAI connection timed out: {e}. Check network/firewall.")
        except Exception as e:
            logger.error(f"AssemblyAI client error: {e}")
        finally:
            self._connected = False

    def _audio_generator(self):
        logger.debug("Audio generator started")
        chunks_yielded = 0
        while self._running:
            item = self._audio_queue.get()
            if item is None:
                logger.debug("Audio generator received None, stopping")
                break
            chunk, _ = item
            chunks_yielded += 1
            if chunks_yielded % 10 == 0:
                logger.debug(f"Yielded {chunks_yielded} chunks to AssemblyAI")
            yield chunk
        logger.debug("Audio generator exited")

    def stop(self):
        logger.debug("Stopping transcriber...")
        self._running = False
        self._audio_queue.put(None)
        
        if self._thread is None:
            return
        
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("Transcriber thread did not exit in time")
            if self.client:
                self.client.close()
        logger.debug("Transcriber stopped")