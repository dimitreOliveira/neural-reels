import logging
from io import BytesIO

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_image_from_bytes(image_bytes: bytes, output_filepath: str) -> None:
    """Saves image bytes to a file using Pillow.

    Args:
        image_bytes: The raw bytes of the image.
        output_filepath: The path where the image file will be saved.
    """
    image = Image.open(BytesIO(image_bytes))
    image.save(output_filepath)
    logger.info(f"Image successfully saved to '{output_filepath}'")
