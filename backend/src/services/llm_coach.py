import logging
import json
from typing import List, Optional, Dict, Generator, Tuple
from groq import Groq
from src.core.config import settings
from src.schemas.audio_metrics import TimestampsSegment, FeedbackSummary

logger = logging.getLogger(__name__)

class LLMCoach:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        # llama-3.3-70b-versatile follows prompts better than qwen3
        self.model = "llama-3.3-70b-versatile"

    def generate_live_feedback(self, messages: List[Dict[str, str]]) -> Generator[Tuple[str, Optional[str]], None, None]:
        """
        Streamed live feedback based on conversation history.
        Yields: (text_chunk, emotion_tag)
        """
        try:
            # We assume 'messages' already contains the latest user input
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=60, # Keep it short and punchy
                stream=True
            )
            
            buffer = ""
            current_emotion = None
            
            for chunk in response_stream:
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

    def generate_final_report(
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
            formatted_transcript = self._format_transcript(transcript_history)
            formatted_history = self._format_llm_history(llm_context)
            
            if not formatted_transcript:
                return {"error": "No transcript data available"}

            # Convert summary Pydantic model to string/dict representation
            summary_json = summary_stats.model_dump_json(indent=2)

            system_prompt = (
                "You are an expert Public Speaking Coach. Generate a 'Final Session Report' in JSON format.\n"
                "Use ALL the provided data:\n"
                "- 'Transcript': What the speaker said with timestamps\n"
                "- 'Session Summary Stats': Contains prosody metrics (pitch, volume, hesitation, fluency scores)\n"
                "- 'Feedback History': Your live coaching interventions during the session\n\n"
                "IMPORTANT: Analyze the SESSION SUMMARY STATS carefully and incorporate these insights:\n"
                "- Pitch dynamics and range: Was the speaker monotone or expressive?\n"
                "- Volume levels: Were they too quiet, too loud, or well-modulated?\n"
                "- Hesitation rate: Did they use filler words or pause excessively?\n"
                "- Fluency: Was their speech smooth or choppy?\n"
                "- Overall prosody score: Reference this in your Session Overview\n\n"
                "Output Format: A single JSON list:\n"
                "[\n"
                "  {\n"
                "    \"timestamp_start\": \"MM:SS\" (or null if general),\n"
                "    \"timestamp_end\": \"MM:SS\" (or null if general),\n"
                "    \"category\": \"Content\" | \"Delivery\" | \"Prosody\" | \"Engagement\",\n"
                "    \"emotion\": \"Encouraging\" | \"Impressed\" | \"Concerned\" | \"Supportive\" | \"Constructive\",\n"
                "    \"issue\": \"Short Label\",\n"
                "    \"feedback\": \"Contextualized explanation using transcript AND metrics...\",\n"
                "    \"score\": 0-100\n"
                "  }, ...\n"
                "]\n"
                "Include items for both content (what they said) AND delivery (how they said it).\n"
                "TIMESTAMPS ARE REQUIRED for all items:\n"
                "- Content items: Use timestamps from the TRANSCRIPT (format: [MM:SS - MM:SS] text)\n"
                "- Prosody/Delivery items: Use timestamps from the METRICS (start_time, end_time)\n"
                "- Only Session Overview can have null timestamps\n"
                "End with a 'Session Overview' item summarizing overall performance."
            )
            
            user_content = (
                f"TRANSCRIPT:\n{formatted_transcript}\n\n"
                f"SESSION SUMMARY STATS (PROSODY METRICS):\n{summary_json}\n\n"
                f"FEEDBACK HISTORY (Your past interventions):\n{formatted_history}\n\n"
                "Generate a comprehensive JSON report that analyzes BOTH content AND delivery metrics."
            )

            completion = self.client.chat.completions.create(
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

    def _format_transcript(self, transcript_history: List[TimestampsSegment]) -> str:
        """Helper to format transcript history."""
        lines = []
        for segment in transcript_history:
            start_str = self._seconds_to_min_sec(segment.start_time)
            end_str = self._seconds_to_min_sec(segment.end_time)
            lines.append(f"[{start_str} - {end_str}] \"{segment.text}\"")
        return "\n".join(lines)

    def _seconds_to_min_sec(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02}:{secs:02}"
