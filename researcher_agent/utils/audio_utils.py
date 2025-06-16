import logging
import wave
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_wave_file(output_filepath: str, pcm, channels=1, rate=24000, sample_width=2):
    """Saves PCM audio data to a WAV file."""
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)

    with wave.open(output_filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    logger.info(f"Audio content saved to '{output_filepath}'")
