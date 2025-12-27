from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
import logging
import uuid

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
            data = await websocket.receive_text()
            
            try: 
                payload = StreamPayload.model_validate_json(data)
                
                prosody_metrics, transcript_segment = await run_in_threadpool(
                    session.engine.process_stream, 
                    payload.audio_chunk
                )

                manager.store_results(session_id, prosody_metrics, transcript_segment)

                if prosody_metrics or transcript_segment:
                    response = FeedbackResponse(
                        processed_at=payload.timestamp,
                        metrics=prosody_metrics,
                        transcript=transcript_segment
                    )
                    await websocket.send_text(response.model_dump_json())

            except ValidationError as e:
                logger.error(f"Payload validation error: {e}")
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await run_in_threadpool(manager.disconnect, session_id)