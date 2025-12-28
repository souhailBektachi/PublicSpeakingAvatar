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


async def orchestrate_live_feedback(session: Session, transcript: TimestampsSegment, websocket: WebSocket):
    """Generate LLM feedback with TTS and send to client."""
    user_msg = f"User said: \"{transcript.text}\""
    session.llm_context.append({"role": "user", "content": user_msg})
    
    current_time = time.time()
    if current_time - session.last_feedback_time < 10.0:
        logger.info(f"Skipping LLM (Cooldown): {10.0 - (current_time - session.last_feedback_time):.1f}s")
        return

    session.last_feedback_time = current_time
    logger.info("Triggering LLM Coach...")
    
    full_text = ""
    emotion = None
    for chunk, e in session.llm_coach.generate_live_feedback(session.llm_context):
        if e:
            emotion = e
        full_text += chunk

    audio = None
    if full_text.strip() and session.tts.is_enabled() and session.audio_coordinator.can_send_audio():
        logger.info(f"Generating TTS: '{full_text}'")
        audio = await run_in_threadpool(session.tts.synthesize, full_text)
        if audio:
            session.audio_coordinator.mark_audio_sent()

    response = FeedbackResponse(emotion=emotion, audio=audio)
    await websocket.send_text(response.model_dump_json(exclude_none=True))
    logger.info(f"Sent: [{emotion}] (audio: {'yes' if audio else 'no'})")

    stored = f"[{emotion}] {full_text}" if emotion else full_text
    session.llm_context.append({"role": "assistant", "content": stored})


async def generate_and_send_report(session: Session, websocket: WebSocket):
    """Generate final report with TTS audio for each item."""
    logger.info("Flushing transcriber...")
    pending = await run_in_threadpool(session.engine.flush_transcriber, 5.0)
    
    for t in pending:
        if t.text and getattr(t, "is_final", True):
            session.transcript_history.append(t)
    
    if not session.transcript_history:
        logger.info("No transcripts - skipping report")
        return
    
    logger.info(f"Generating report ({len(session.transcript_history)} transcripts)...")
    report = await run_in_threadpool(session.generate_report)
    
    logger.info(f"Report type: {type(report)}, TTS enabled: {session.tts.is_enabled()}")
    
    items = report.get("session_report", []) if isinstance(report, dict) else report if isinstance(report, list) else []
    logger.info(f"Found {len(items)} items for TTS")
    
    for item in items:
        text = item.get("feedback", "")
        if text and session.tts.is_enabled():
            logger.info(f"Generating TTS for: {item.get('issue', 'N/A')}")
            item["audio"] = await run_in_threadpool(session.tts.synthesize, text)
    
    await websocket.send_text(ReportResponse(report=report).model_dump_json())
    logger.info("Report sent")


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
                    raw = json.loads(data)
                    
                    if raw.get("type") == "end_session":
                        logger.info("End session requested")
                        await generate_and_send_report(session, websocket)
                        break
                    
                    payload = StreamPayload.model_validate(raw)
                    prosody, transcript = await run_in_threadpool(
                        session.engine.process_stream, payload.audio_chunk
                    )
                    await manager.store_results(session_id, prosody, transcript)

                    if prosody:
                        feedback = session.live_analyzer.analyze(prosody)
                        audio = feedback.get("audio") if feedback else None
                        if audio and session.audio_coordinator.can_send_audio():
                            await websocket.send_text(FeedbackResponse(audio=audio).model_dump_json(exclude_none=True))
                            session.audio_coordinator.mark_audio_sent()
                            logger.info("LiveAnalyzer audio sent")

                    if transcript and transcript.text and getattr(transcript, "is_final", False):
                        logger.info(f"Transcript: '{transcript.text}'")
                        await orchestrate_live_feedback(session, transcript, websocket)

                except ValidationError as e:
                    logger.error(f"Validation error: {e}")

            extra = session.engine.get_transcript()
            if extra and extra.text and getattr(extra, "is_final", False):
                logger.info(f"Async transcript: '{extra.text}'")
                await manager.store_results(session_id, None, extra)
                await orchestrate_live_feedback(session, extra, websocket)

    except WebSocketDisconnect:
        logger.info("Disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await run_in_threadpool(manager.disconnect, session_id)