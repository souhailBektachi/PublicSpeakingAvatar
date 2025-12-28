import threading
import queue
import os
import logging
import time
import numpy as np
from typing import Optional, List
from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    StreamingError,
    TurnEvent
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

    def flush(self, timeout: float = 5.0) -> List[TimestampsSegment]:
        """
        Drain all pending transcripts from the result queue.
        Waits up to `timeout` seconds for any in-flight results.
        Returns list of all collected transcripts.
        """
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                # Try to get results with short timeout
                result = self._result_queue.get(timeout=0.5)
                if result:
                    results.append(result)
                    logger.info(f"Flush collected: '{result.text}'")
            except queue.Empty:
                # If queue empty and no new results for a bit, we're done
                if len(results) > 0:
                    # We got some results, wait a bit more for stragglers
                    try:
                        result = self._result_queue.get(timeout=1.0)
                        if result:
                            results.append(result)
                            logger.info(f"Flush collected (straggler): '{result.text}'")
                    except queue.Empty:
                        break
                else:
                    # No results yet, keep waiting
                    continue
        
        logger.info(f"Flush complete: {len(results)} transcripts collected")
        return results

    def _client_loop(self):
        logger.debug("Transcriber client loop started")
        
        while self._running:
            try:
                # 1. Wait for data before connecting to avoid idle timeouts
                item = self._audio_queue.get()
                if item is None:
                    break
                
                if not self.api_key:
                    logger.error("No AssemblyAI API Key found. Waiting...")
                    import time
                    time.sleep(5)
                    continue

                self.client = StreamingClient(
                    StreamingClientOptions(
                        api_key=self.api_key,
                        api_host="streaming.assemblyai.com",
                    )
                )

                def on_turn(client, event: TurnEvent):
                    if event.end_of_turn and not event.turn_is_formatted:
                        try:
                            client.set_params(
                                StreamingSessionParameters(format_turns=True)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to set params: {e}")

                    text = event.transcript
                    if text:
                        is_final = event.end_of_turn
                        current_time = self.processed_samples / self.sample_rate
                        
                        if is_final:
                            logger.info(f"Final Transcript: {text}")
                        
                        segment = TimestampsSegment(
                            text=text,
                            start_time=max(0.0, current_time - 2.0),
                            end_time=current_time,
                            is_final=is_final,
                        )
                        self._result_queue.put(segment)

                def on_error(client, error: StreamingError):
                    code = getattr(error, 'code', None)
                    if code == 1006 or str(code) == '1006':
                        logger.info(f"AssemblyAI Connection Closed (Idle): {error}")
                    else:
                        logger.error(f"AssemblyAI Streaming Error: {error}")

                def on_close(client, event):
                    logger.info(f"AssemblyAI Session Closed: {event}")
                    self._connected = False

                self.client.on(StreamingEvents.Turn, on_turn)
                self.client.on(StreamingEvents.Error, on_error)
                self.client.on(StreamingEvents.Termination, on_close)

                logger.debug("Connecting to AssemblyAI...")
                self.client.connect(
                    StreamingParameters(
                        sample_rate=self.sample_rate,
                        encoding="pcm_s16le", 
                        format_turns=True 
                    )
                )
                self._connected = True
                logger.info("Connected to AssemblyAI")

                # Stream using the item we just popped + the rest
                self.client.stream(self._audio_generator(start_item=item))
                
                logger.info("Stream finished normally")
                
            except Exception as e:
                logger.error(f"Transcriber loop error: {e}")
                import time
                time.sleep(2) 
            finally:
                self._connected = False
                if self.client:
                    try:
                        self.client.close()
                    except:
                        pass
                
                if not self._running:
                    break
        
        logger.debug("Transcriber client loop exited")

    def _audio_generator(self, start_item=None):
        logger.debug("Audio generator started")
        
        if start_item:
            chunk, _ = start_item
            yield chunk

        chunks_yielded = 0
        
        # AssemblyAI requires chunks between 50-1000ms. Use 100ms for silence.
        timeout_duration = 0.1 
        silence_chunk_size = int(self.sample_rate * 2 * timeout_duration) 
        silence_chunk = b"\x00" * silence_chunk_size
        
        while self._running:
            try:
                item = self._audio_queue.get(timeout=timeout_duration)
            except queue.Empty:
                yield silence_chunk
                continue

            if item is None:
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
                self.client.disconnect(terminate=True)
        logger.debug("Transcriber stopped")