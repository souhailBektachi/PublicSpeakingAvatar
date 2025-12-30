import sys
import os
import asyncio

# Ensure 'backend' is in path so we can import 'src'
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.services.synthesizers.kokoro import KokoroSynthesizer

async def main():
    print("Initializing KokoroSynthesizer (Puck)...")
    try:
        tts = KokoroSynthesizer(voice_id="puck")
        text = "Hello! I am Puck. I am running locally on your G P U."
        print(f"Synthesizing: '{text}'")
        
        audio = await tts.synthesize(text)
        
        out_file = "test_audio.wav"
        with open(out_file, "wb") as f:
            f.write(audio)
        
        print(f"Success! Audio saved to: {os.path.abspath(out_file)}")
        
        # Try to open the file with the default media player
        if sys.platform == "win32":
            print("Playing audio...")
            os.startfile(out_file)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
