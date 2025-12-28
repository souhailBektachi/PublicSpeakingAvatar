from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
import logging
import uuid
import asyncio
import time
import json

from src.schemas.protocol import StreamPayload, FeedbackResponse, ReportResponse
from src.services.session_manager import manager, Session
from src.schemas.audio_metrics import TimestampsSegment

logger = logging.getLogger(__name__)
router = APIRouter()

async def orchestrate_live_feedback(session: Session, transcript: TimestampsSegment, timestamp: float, websocket: WebSocket):
    """
    Handles the full lifecycle of generating and streaming live feedback.
    1. Updates User Context
    2. Calls LLMCoach (Stream)
    3. Streams chunks to client
    4. Updates Assistant Context
    """
    # 1. Update Context (State stored in Session)
    user_msg = f"User said: \"{transcript.text}\""
    session.llm_context.append({"role": "user", "content": user_msg})
    
    # Frequency Regulation: Use monotonic clock, not payload timestamp
    current_time = time.time()
    if current_time - session.last_feedback_time < 10.0:
        logger.info(f"Skipping LLM (Cooldown): {10.0 - (current_time - session.last_feedback_time):.2f}s remaining")
        return

    session.last_feedback_time = current_time
    
    # 2. Call Service (Stateless LLMCoach)
    logger.info("Triggering LLM Coach...")
    coach_generator = session.llm_coach.generate_live_feedback(session.llm_context)
    
    full_response_text = ""
    current_emotion = None
    
    for text_chunk, emotion in coach_generator:
        if emotion:
            current_emotion = emotion
        full_response_text += text_chunk
        
        response = FeedbackResponse(
            processed_at=timestamp,
            text=text_chunk,
            emotion=emotion
        )
        await websocket.send_text(response.model_dump_json())
        await asyncio.sleep(0)

    # 3. Update Context with Assistant Response
    if current_emotion:
        stored_response = f"[{current_emotion}] {full_response_text}"
    else:
        stored_response = full_response_text

    session.llm_context.append({"role": "assistant", "content": stored_response})


async def generate_and_send_report(session: Session, websocket: WebSocket):
    """Generate final report and send to client."""
    
    # 1. Flush the transcriber to collect any pending transcripts
    logger.info("Flushing transcriber for pending results...")
    pending_transcripts = await run_in_threadpool(session.engine.flush_transcriber, 5.0)
    
    # 2. Store flushed transcripts in session history
    for transcript in pending_transcripts:
        if transcript.text and getattr(transcript, "is_final", True):
            session.transcript_history.append(transcript)
    
    # 3. Check if we have any transcripts now
    if not session.transcript_history:
        logger.info("No transcript history after flush - skipping report generation")
        return
    
    logger.info(f"Generating final report with {len(session.transcript_history)} transcripts...")
    report = await run_in_threadpool(session.generate_report)
    
    report_response = ReportResponse(report=report)
    await websocket.send_text(report_response.model_dump_json())
    logger.info("Final report sent to client")


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
                    # Parse as generic JSON to check message type
                    raw = json.loads(data)
                    msg_type = raw.get("type", "audio")
                    
                    # Handle end_session message
                    if msg_type == "end_session":
                        logger.info("Received end_session - generating final report")
                        await generate_and_send_report(session, websocket)
                        break  # Exit the loop, cleanup will happen in finally
                    
                    # Handle audio message
                    payload = StreamPayload.model_validate(raw)
                    
                    prosody_metrics, transcript_segment = await run_in_threadpool(
                        session.engine.process_stream,
                        payload.audio_chunk,
                    )

                    await manager.store_results(session_id, prosody_metrics, transcript_segment)

                    if transcript_segment and transcript_segment.text and getattr(transcript_segment, "is_final", False):
                        logger.info(f"Transcript (final): '{transcript_segment.text}'")
                        await orchestrate_live_feedback(session, transcript_segment, payload.timestamp, websocket)

                    if prosody_metrics:
                        feedback = session.live_analyzer.analyze(prosody_metrics)
                        audio_b64 = feedback.get("audio") if feedback else None
                        if audio_b64:
                            response = FeedbackResponse(processed_at=payload.timestamp, audio=audio_b64)
                            await websocket.send_text(response.model_dump_json())

                except ValidationError as e:
                    logger.error(f"Payload validation error: {e}")
                    continue
            
            extra_transcript = session.engine.get_transcript()
            if extra_transcript and extra_transcript.text and getattr(extra_transcript, "is_final", False):
                logger.info(f"Transcript (async, final): '{extra_transcript.text}'")
                await manager.store_results(session_id, None, extra_transcript)
                await orchestrate_live_feedback(session, extra_transcript, 0.0, websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await run_in_threadpool(manager.disconnect, session_id)