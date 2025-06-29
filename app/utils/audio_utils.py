import logging
import wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_wave_file(
    output_filepath: str,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    """Saves PCM audio data to a WAV file.

    Args:
        output_filepath: The path to save the output WAV file.
        pcm: The raw PCM audio data as bytes.
        channels: The number of audio channels.
        rate: The sampling rate in Hz.
        sample_width: The sample width in bytes.
    """
    with wave.open(output_filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    logger.info(f"Audio content saved to '{output_filepath}'")
