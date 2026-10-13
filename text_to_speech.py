import asyncio
from pathlib import Path
import edge_tts

async def main():
    # Read text from the file beside this Python script
    text = Path(__file__).with_name("input.txt").read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        print("The text file is empty.")
        return

    # Hindi voice; English pronunciation may vary in mixed text
    voice = "hi-IN-SwaraNeural"

    tts = edge_tts.Communicate(
        text,
        voice=voice,
        rate="+0%"
    )

    output = Path(__file__).with_name("speech.mp3")
    await tts.save(str(output))

    print(f"Speech saved to: {output}")

asyncio.run(main())
