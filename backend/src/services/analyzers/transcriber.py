import threading
import queue
import os
import logging
import numpy as np
from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters
)

load_dotenv()
logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.api_key = os.getenv("AssemblyAI_API_KEY")
        
        self._audio_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._running = True
        self.client = None
        
        self._thread = threading.Thread(target=self._client_loop, daemon=True)
        self._thread.start()

    def process(self, new_chunk: np.ndarray):
        pcm_data = (new_chunk * 32767).astype(np.int16).tobytes()
        self._audio_queue.put(pcm_data)
        
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def _client_loop(self):
        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        def on_turn(client, event):
            if event.transcript:
                self._result_queue.put(event.transcript)

        def on_error(client, error):
            logger.error(f"AssemblyAI Error: {error}")

        self.client.on(StreamingEvents.Turn, on_turn)
        self.client.on(StreamingEvents.Error, on_error)

        self.client.connect(
            StreamingParameters(
                sample_rate=self.sample_rate,
                format_turns=True,
            )
        )

        try:
            self.client.stream(self._audio_generator())
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            self.client.disconnect()

    def _audio_generator(self):
        while self._running:
            chunk = self._audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def stop(self):
        logger.info("Stopping transcriber...")
        self._running = False
        self._audio_queue.put(None)
        
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("Transcriber thread did not exit in time, forcing disconnect")
            if self.client:
                try:
                    self.client.close()
                except Exception as e:
                    logger.error(f"Error forcing disconnect: {e}")
        logger.info("Transcriber stopped")