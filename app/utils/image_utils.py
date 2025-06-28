import logging
from io import BytesIO

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_image_from_bytes(image_bytes: bytes, output_filepath: str):
    """Saves image bytes to a file."""
    image = Image.open(BytesIO(image_bytes))
    image.save(output_filepath)
    logger.info(f"Image successfully saved to '{output_filepath}'")
