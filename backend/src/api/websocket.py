from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
import logging
import uuid
import asyncio

from src.schemas.protocol import StreamPayload, FeedbackResponse
from src.services.session_manager import manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    await manager.connect(session_id, websocket)
    
    session = manager.get_session(session_id)
    if not session:
        return

    try: 
        while True:
            data = None
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

            if data:
                try: 
                    payload = StreamPayload.model_validate_json(data)
                    
                    prosody_metrics, transcript_segment = await run_in_threadpool(
                        session.engine.process_stream,
                        payload.audio_chunk,
                    )

                    await manager.store_results(session_id, prosody_metrics, transcript_segment)

                    if transcript_segment and transcript_segment.text and getattr(transcript_segment, "is_final", False):
                        logger.info(f"Transcript (final): '{transcript_segment.text}'")

                    if prosody_metrics:
                        feedback = session.live_analyzer.analyze(prosody_metrics)
                        audio_b64 = feedback.get("audio") if feedback else None
                        if audio_b64:
                            response = FeedbackResponse(processed_at=payload.timestamp, audio=audio_b64)
                            await websocket.send_text(response.model_dump_json())

                except ValidationError as e:
                    logger.error(f"Payload validation error: {e}")
                    continue
            
            # Check for pending transcripts (e.g. arriving after audio processing)
            extra_transcript = session.engine.get_transcript()
            if extra_transcript:
                await manager.store_results(session_id, None, extra_transcript)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await run_in_threadpool(manager.disconnect, session_id)