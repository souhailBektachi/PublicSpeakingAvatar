from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
import logging
import uuid
import asyncio
import time
import json

from src.schemas.protocol import StreamPayload, FeedbackResponse, ReportResponse, TelemetryPayload
from src.services.session_manager import manager, Session
from src.schemas.audio_metrics import TimestampsSegment

logger = logging.getLogger(__name__)
router = APIRouter()


async def send_audio_feedback(session: Session, websocket: WebSocket, audio: str, sentiment: str = None, source: str = "unknown"):
    """Send audio feedback if coordinator allows. Returns True if sent."""
    if not audio or not session.audio_coordinator.can_send_audio(source):
        return False
    
    response = FeedbackResponse(audio=audio, sentiment=sentiment)
    await websocket.send_text(response.model_dump_json(exclude_none=True))
    session.audio_coordinator.mark_audio_sent(source)
    logger.info(f"Audio sent ({source})")
    return True


async def handle_live_analyzer(session: Session, websocket: WebSocket, prosody):
    """Analyze prosody and send audio feedback if needed."""
    if not prosody:
        return
    
    feedback = session.live_analyzer.analyze(prosody)
    audio = feedback.get("audio") if feedback else None
    await send_audio_feedback(session, websocket, audio, source="LiveAnalyzer")


async def handle_llm_feedback(session: Session, websocket: WebSocket, transcript: TimestampsSegment):
    """Generate LLM feedback with TTS and send to client."""
    session.llm_context.append({"role": "user", "content": f'User said: "{transcript.text}"'})
    
    # 10-second cooldown between LLM calls
    if time.time() - session.last_feedback_time < 10.0:
        return
    session.last_feedback_time = time.time()
    
    logger.info("LLM Coach triggered...")
    
    # Buffer LLM response
    full_text, emotion = "", None
    async for chunk, e in session.llm_coach.generate_live_feedback(session.llm_context):
        if e:
            emotion = e
        full_text += chunk

    logger.info(f"LLM buffered text: '{full_text}' [{emotion}]")

    # Generate TTS if coordinator allows
    audio = None
    if full_text.strip() and session.tts.is_enabled() and session.audio_coordinator.can_send_audio("LLM"):
        logger.info(f"Generating TTS for: '{full_text}'")
        audio = await run_in_threadpool(session.tts.synthesize, full_text)
    
    await send_audio_feedback(session, websocket, audio, emotion, source="LLM")
    
    # Store in history
    stored = f"[{emotion}] {full_text}" if emotion else full_text
    session.llm_context.append({"role": "assistant", "content": stored})


async def generate_and_send_report(session: Session, websocket: WebSocket):
    """Generate final report with TTS for each item."""
    logger.info("Flushing transcriber...")
    pending = await run_in_threadpool(session.engine.flush_transcriber, 5.0)
    
    for t in pending:
        if t.text and getattr(t, "is_final", True):
            session.transcript_history.append(t)
    
    if not session.transcript_history:
        logger.info("No transcripts - skipping report")
        return
    
    logger.info(f"Generating report ({len(session.transcript_history)} transcripts)...")
    logger.info(f"Generating report ({len(session.transcript_history)} transcripts)...")
    report = await session.generate_report()
    
    # Debug: Log report structure
    logger.info(f"TTS enabled: {session.tts.is_enabled()}")
    logger.info(f"Report type: {type(report).__name__}")
    if isinstance(report, dict):
        logger.info(f"Report keys: {list(report.keys())}")
    
    # Find items - handle different report structures
    if isinstance(report, dict):
        # Try common keys
        items = report.get("session_report") or report.get("report") or report.get("feedback") or []
        if not items and len(report) == 1:
            # If dict has single key, use its value
            items = list(report.values())[0] if isinstance(list(report.values())[0], list) else []
    elif isinstance(report, list):
        items = report
    else:
        items = []
    
    logger.info(f"Found {len(items)} items for TTS")
    
    for i, item in enumerate(items):
        feedback_list = item.get("feedback", [])
        if isinstance(feedback_list, list):
             # If feedback is a list, generate TTS for the first "message" found
            message = ""
            for fb in feedback_list:
                if isinstance(fb, dict) and fb.get("message"):
                    message = fb.get("message")
                    break
            
            if message and session.tts.is_enabled():
                logger.info(f"Generating TTS for item {i+1}: {message[:30]}")
                audio = await run_in_threadpool(session.tts.synthesize, message)
                if audio:
                    # Append audio to the first feedback item that has a message
                    for fb in feedback_list:
                         if isinstance(fb, dict) and fb.get("message") == message:
                             fb["audio"] = audio
                             break
                    logger.info(f"TTS generated for item {i+1}")
                else:
                    logger.warning(f"TTS failed for item {i+1}")
        
        elif isinstance(feedback_list, str):
             # Fallback for old string format
             text = feedback_list
             if text and session.tts.is_enabled():
                logger.info(f"Generating TTS for item {i+1}: {item.get('issue', 'N/A')[:30]}")
                audio = await run_in_threadpool(session.tts.synthesize, text)
                if audio:
                    item["audio"] = audio
                    logger.info(f"TTS generated for item {i+1}")
                else:
                    logger.warning(f"TTS failed for item {i+1}")
    
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
                        session.reset()
                        continue
                    if raw.get("type") == "telemetry":
                        telemetry = TelemetryPayload.model_validate(raw)
                        session.latest_pose = telemetry.pose_data
                        session.last_telemetry_time = telemetry.timestamp
                        
                        if telemetry.is_speaking:
                            session.last_speech_end_time = telemetry.timestamp
                            session.silence_alert_sent = False
                        else:
                            silence_duration = telemetry.timestamp - session.last_speech_end_time
                            if silence_duration > 1.0:
                                logger.info(f"Silence duration: {silence_duration:.1f}s")
                            
                            if silence_duration > 8.0 and not session.silence_alert_sent:
                                session.silence_alert_sent = True
                                logger.info(f"Silence alert triggered at {telemetry.timestamp}s")
                                response = FeedbackResponse(
                                    type="feedback",
                                    feedback="You have been silent for a while. Take a deep breath and continue.",
                                    sentiment="neutral",
                                    timestamp=telemetry.timestamp
                                )
                                await websocket.send_text(response.model_dump_json(exclude_none=True))
                        
                        continue

                    payload = StreamPayload.model_validate(raw)
                    prosody, transcript = await run_in_threadpool(
                        session.engine.process_stream, 
                        payload.audio_chunk,
                        timestamp=payload.timestamp
                    )
                    await manager.store_results(session_id, prosody, transcript)

                    # Handle feedback (LiveAnalyzer first, then LLM)
                    await handle_live_analyzer(session, websocket, prosody)
                    
                    if transcript and transcript.text and getattr(transcript, "is_final", False):
                        logger.info(f"Transcript: '{transcript.text}'")
                        await handle_llm_feedback(session, websocket, transcript)

                except ValidationError as e:
                    logger.error(f"Validation error: {e}")

            # Check for async transcripts
            extra = session.engine.get_transcript()
            if extra and extra.text and getattr(extra, "is_final", False):
                logger.info(f"Async transcript: '{extra.text}'")
                await manager.store_results(session_id, None, extra)
                await handle_llm_feedback(session, websocket, extra)

    except WebSocketDisconnect:
        logger.info("Disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await run_in_threadpool(manager.disconnect, session_id)