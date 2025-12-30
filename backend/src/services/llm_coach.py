import logging
import json
from typing import List, Optional, Dict, Generator, Tuple, AsyncGenerator
from groq import AsyncGroq
from src.core.config import settings
from src.schemas.audio_metrics import TimestampsSegment, FeedbackSummary

logger = logging.getLogger(__name__)

class LLMCoach:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        # llama-3.3-70b-versatile follows prompts better than qwen3
        self.model = "llama-3.3-70b-versatile"

    async def generate_live_feedback(self, messages: List[Dict[str, str]]) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Streamed live feedback based on conversation history.
        Yields: (text_chunk, emotion_tag)
        """
        try:
            # We assume 'messages' already contains the latest user input
            response_stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=60, # Keep it short and punchy
                stream=True
            )
            
            buffer = ""
            current_emotion = None
            
            async for chunk in response_stream:
                content = chunk.choices[0].delta.content or ""
                if not content:
                    continue
                
                # DEBUG: Log raw content from Groq
                logger.debug(f"Raw LLM chunk: '{content}'")
                
                # Filter out <think> tags if they still appear
                clean_content = content.replace("<think>", "").replace("</think>", "")
                if not clean_content.strip(): 
                    continue
                
                buffer += clean_content
                
                # Logic to extract [Emotion] tag if present at start
                if current_emotion is None:
                    if "]" in buffer and "[" in buffer:
                        start = buffer.find("[")
                        end = buffer.find("]")
                        if start != -1 and end != -1 and end > start:
                            current_emotion = buffer[start+1:end]
                            remaining_text = buffer[end+1:].lstrip()
                            if remaining_text:
                                logger.info(f"LLM yielding: '{remaining_text}' with emotion '{current_emotion}'")
                                yield remaining_text, current_emotion
                            buffer = "" 
                    else:
                         # Fallback if buffer gets too long without tag
                        if len(buffer) > 20 and "[" not in buffer:
                             logger.info(f"LLM yielding (no tag): '{buffer}'")
                             yield buffer, "Neutral" # Default emotion if omitted
                             buffer = ""
                else:
                    if buffer:
                        logger.info(f"LLM yielding: '{buffer}'")
                        yield buffer, current_emotion
                        buffer = ""
            
            if buffer:
                logger.info(f"LLM final yield: '{buffer}'")
                yield buffer, current_emotion

        except Exception as e:
            logger.error(f"Error generating live feedback: {e}")
            yield "", None

    async def generate_final_report(
        self, 
        transcript_history: List[TimestampsSegment], 
        summary_stats: FeedbackSummary,
        llm_context: List[Dict[str, str]]
    ) -> Dict:
        """
        Generates a JSON Final Report using transcript, summary stats, and conversation history.
        Returns a Python dictionary (parsed JSON).
        """
        try:
            # Build the mathematical skeleton (Speech + Silence blocks)
            timeline_skeleton = self._build_timeline_skeleton(transcript_history)
            skeleton_json = json.dumps(timeline_skeleton, indent=2)

            formatted_history = self._format_llm_history(llm_context)

            # Convert summary Pydantic model to string/dict representation
            summary_json = summary_stats.model_dump_json(indent=2)

            system_prompt = (
                "You are an expert Public Speaking Coach. Your task is to ANNOTATE an existing Session Timeline.\n"
                "You will be given a JSON 'Timeline Skeleton' containing 'speech' and 'silence' segments.\n"
                "- Do NOT change the timestamps.\n"
                "- Do NOT change the content.\n"
                "- Your job is to fill the empty \"feedback\" array for each segment based on the context.\n\n"
                "Use the provided 'Session Summary Stats' (pitch, volume, hesitation) to inform your feedback.\n\n"
                "RULES for FEEDBACK:\n"
                "1. For 'speech' segments: Analyze content quality, clarity, and delivery metrics.\n"
                "2. For 'silence' segments: If > 4s, warn about dead air. If 2-4s, check if it's a good pause or awkward.\n"
                "3. Format each feedback item as: { \"category\": \"...\", \"sentiment\": \"positive|neutral|negative|warning\", \"message\": \"...\" }\n"
                "4. Add a 'Session Overview' segment at the end of the timeline (timestamp_start=0, timestamp_end=total_duration) with general feedback.\n\n"
                "Output: The FULL JSON Timeline with your added feedback arrays."
            )
            
            user_content = (
                f"TIMELINE SKELETON (Annotate this):\n{skeleton_json}\n\n"
                f"SESSION SUMMARY STATS (Metrics):\n{summary_json}\n\n"
                f"FEEDBACK HISTORY (Past interventions):\n{formatted_history}\n\n"
                "Return the valid JSON timeline with annotations."
            )

            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.5,
                max_tokens=2000,
                stream=False,
                response_format={"type": "json_object"}
            )

            # Parse the JSON string returned by the LLM into a Python dict
            return json.loads(completion.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            return {"error": str(e)}

    def _build_timeline_skeleton(self, transcript_history: List[TimestampsSegment]) -> List[Dict]:
        """
        Mathematically constructs the timeline with Speech and Silence segments.
        """
        timeline = []
        last_end_time = 0.0
        
        # Sort by start time just in case
        sorted_segments = sorted(transcript_history, key=lambda x: x.start_time)
        
        for segment in sorted_segments:
            # Check for silence gap (> 2.0s)
            gap = segment.start_time - last_end_time
            if gap > 2.0:
                timeline.append({
                    "type": "silence",
                    "timestamp_start": round(last_end_time, 2),
                    "timestamp_end": round(segment.start_time, 2),
                    "duration": round(gap, 2),
                    "content": None,
                    "feedback": []
                })
            
            # Add speech segment
            timeline.append({
                "type": "speech",
                "timestamp_start": round(segment.start_time, 2),
                "timestamp_end": round(segment.end_time, 2),
                "content": segment.text,
                "feedback": []
            })
            
            last_end_time = max(last_end_time, segment.end_time)
            
        return timeline

    def _format_llm_history(self, llm_context: List[Dict[str, str]]) -> str:
        """Helper to format the conversation history for the final report."""
        lines = []
        for msg in llm_context:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue # Skip system prompt
            lines.append(f"[{role.upper()}]: {content}")
        return "\n".join(lines)
