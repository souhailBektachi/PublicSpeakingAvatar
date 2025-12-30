from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
import logging
import uuid
import asyncio
import time
import json
from pathlib import Path
from src.services.analyzers.pose_interpreter import PoseFeedbackEngine

from src.schemas.protocol import StreamPayload, FeedbackResponse, ReportResponse, TelemetryPayload
from src.services.session_manager import manager, Session
from src.schemas.audio_metrics import TimestampsSegment

logger = logging.getLogger(__name__)
router = APIRouter()

def _summary_to_text(summary: dict) -> str:
    posture = summary.get("posture", {})
    arms = posture.get("arms", {})
    behavior = summary.get("behavior", {})
    position = summary.get("position", {})

    open_posture = posture.get("open", True)
    parts = [
        f"Posture={'open' if open_posture else 'closed'}",
        f"Arms(left_raised={arms.get('left_raised', False)},right_raised={arms.get('right_raised', False)},elbows={arms.get('elbows','neutral')},hands={arms.get('hands','apart')},crossed={arms.get('crossed', False)})",
        f"Behavior(gestures={behavior.get('gestures','medium')},facial_expression={behavior.get('facial_expression','medium')},facial_energy={behavior.get('facial_energy','medium')},movement={behavior.get('movement','static')})",
        f"Position(horizontal={position.get('horizontal','center')},depth={position.get('depth','middle')})",
    ]
    return "; ".join(parts)

async def send_audio_feedback(session: Session, websocket: WebSocket, audio: str, visemes: dict = None, sentiment: str = None, source: str = "unknown"):
    """Send audio feedback with optional visemes if coordinator allows. Returns True if sent."""
    if not audio or not session.audio_coordinator.can_send_audio(source):
        return False
    
    response = FeedbackResponse(audio=audio, visemes=visemes, sentiment=sentiment)
    await websocket.send_text(response.model_dump_json(exclude_none=True))
    session.audio_coordinator.mark_audio_sent(source)
    logger.info(f"Audio sent ({source}), visemes: {visemes is not None}")
    return True


async def handle_live_analyzer(session: Session, websocket: WebSocket, prosody):
    """Analyze prosody and send audio feedback if needed."""
    if not prosody:
        return
    
    feedback = session.live_analyzer.analyze(prosody)
    audio = feedback.get("audio") if feedback else None
    await send_audio_feedback(session, websocket, audio, visemes=None, source="LiveAnalyzer")


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

    # Generate TTS with visemes if coordinator allows
    audio = None
    visemes = None
    if full_text.strip() and session.tts.is_enabled() and session.audio_coordinator.can_send_audio("LLM"):
        logger.info(f"Generating TTS with visemes for: '{full_text}'")
        result = await run_in_threadpool(session.tts.synthesize_with_visemes, full_text)
        if result:
            audio = result.get("audio")
            visemes = result.get("visemes")
    
    await send_audio_feedback(session, websocket, audio, visemes, emotion, source="LLM")
    
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
    report = await session.generate_report()
    
    # Process items in report for TTS
    if isinstance(report, dict):
        # Try common keys
        items = report.get("session_report") or report.get("report") or report.get("feedback") or report.get("timeline") or []
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
                # Use synthesize_with_visemes to get lip sync data
                result = await run_in_threadpool(session.tts.synthesize_with_visemes, message)
                
                if result and result.get("audio"):
                    audio = result.get("audio")
                    visemes = result.get("visemes")
                    
                    # Append audio and visemes to the first feedback item that has a message
                    for fb in feedback_list:
                         if isinstance(fb, dict) and fb.get("message") == message:
                             fb["audio"] = audio
                             fb["visemes"] = visemes
                             break
                    logger.info(f"TTS + Visemes generated for item {i+1}")
                else:
                    logger.warning(f"TTS failed for item {i+1}")
        
        elif isinstance(feedback_list, str):
             # Fallback for old string format
             text = feedback_list
             if text and session.tts.is_enabled():
                logger.info(f"Generating TTS for item {i+1}: {item.get('issue', 'N/A')[:30]}")
                result = await run_in_threadpool(session.tts.synthesize_with_visemes, text)
                
                if result and result.get("audio"):
                    item["audio"] = result.get("audio")
                    item["visemes"] = result.get("visemes")
                    logger.info(f"TTS + Visemes generated for item {i+1}")
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

    # Pose interpretation setup (per session)
    pose_engine = PoseFeedbackEngine()
    BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
    POSE_DIR = BASE_DIR / "assets" / "pose"
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    POSE_OUT = POSE_DIR / "pose_interpretation.jsonl"

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

                        # Run pose interpretation and aggregate output every 1.5s
                        try:
                            result = pose_engine.process_chunk({
                                "session_id": session_id,
                                "timestamp": telemetry.timestamp,
                                "pose_data": telemetry.pose_data,
                            })

                            last_time = getattr(session, "pose_summary_last_time", 0.0)
                            if telemetry.timestamp - last_time >= 1.5:
                                setattr(session, "pose_summary_last_time", telemetry.timestamp)

                                summary_record = {
                                    "type": "pose_summary",
                                    "session_id": session_id,
                                    "timestamp": telemetry.timestamp,
                                    "summary": result.get("summary", {}),
                                }

                                # Write asynchronously to keep loop responsive
                                def _write_out(entry: dict):
                                    with POSE_OUT.open("a", encoding="utf-8") as f:
                                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                await run_in_threadpool(_write_out, summary_record)
                                logger.info(
                                    f"Pose summary written for {session_id} at {telemetry.timestamp:.2f}s"
                                )

                                # Also append to LLM context for posture-aware feedback
                                try:
                                    summary_text = _summary_to_text(summary_record["summary"]) if summary_record.get("summary") else ""
                                    if summary_text:
                                        session.llm_context.append({
                                            "role": "user",
                                            "content": f"Posture update: {summary_text}"
                                        })
                                        logger.info("Posture summary appended to LLM context")
                                        # keep context bounded
                                        if len(session.llm_context) > 200:
                                            session.llm_context = session.llm_context[-200:]
                                except Exception as e:
                                    logger.error(f"Failed to append posture summary to LLM context: {e}")
                        except Exception as e:
                            logger.error(f"Pose interpretation error: {e}")
                        
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