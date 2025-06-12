import logging
import wave
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Saves PCM audio data to a WAV file."""
    Path(filename).parent.mkdir(exist_ok=True, parents=True)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    logger.info(f"Audio content saved to '{filename}'")
