import asyncio
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.services.synthesizers.kokoro import KokoroSynthesizer

async def main():
    print("Initializing KokoroSynthesizer...")
    try:
        tts = KokoroSynthesizer(voice_id="puck")
        print("Synthesizer initialized.")
        
        text = "Hello, I am Puck. This is a test of the local text to speech system."
        print(f"Synthesizing: '{text}'")
        
        # Test basic synthesis
        audio_wav = await tts.synthesize(text)
        print(f"Synthesis complete. Generated {len(audio_wav)} bytes of audio.")
        
        with open("test_output.wav", "wb") as f:
            f.write(audio_wav)
        print("Saved to test_output.wav")
        
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
