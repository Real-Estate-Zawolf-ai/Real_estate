from pathlib import Path
import logging

from PIL import Image

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


def load_image(file_path: str) -> Image.Image:
    """
    Load and validate an image file.

    Supported formats:
        - JPG / JPEG
        - PNG
        - WEBP
        - BMP
        - TIFF

    Args:
        file_path: Path to the image file.

    Returns:
        PIL Image object.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image format is unsupported.
        Exception: If the image cannot be opened.
    """

    logger.info("Starting image loading: %s", file_path)

    try:
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {file_path}"
            )

        # Validate extension
        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {extension}. "
                f"Supported formats: {SUPPORTED_EXTENSIONS}"
            )

        # Open image
        image = Image.open(path)

        # Force loading to detect corrupted files early
        image.load()

        logger.info(
            "Image loaded successfully | file=%s | format=%s | size=%s | mode=%s",
            file_path,
            image.format,
            image.size,
            image.mode,
        )

        return image

    except Exception:
        logger.exception(
            "Image loading failed | file=%s",
            file_path,
        )
        raise
