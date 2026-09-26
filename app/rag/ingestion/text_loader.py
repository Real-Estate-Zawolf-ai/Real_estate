from pathlib import Path
import logging

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
}


def parse_text(file_path: str) -> str:
    """
    Extract text from a plain text or Markdown file.

    Supported formats:
        - TXT
        - MD
        - MARKDOWN

    Args:
        file_path: Path to the text file.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """

    logger.info("Starting text parsing: %s", file_path)

    try:
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Text file not found: {file_path}"
            )

        # Validate file extension
        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported text format: {extension}. "
                f"Supported formats: {SUPPORTED_EXTENSIONS}"
            )

        # Read file using UTF-8
        text = path.read_text(encoding="utf-8")

        # Normalize text
        text = text.strip()

        if not text:
            logger.warning(
                "Text file is empty: %s",
                file_path,
            )
            return ""

        logger.info(
            "Text parsed successfully | file=%s | characters=%d",
            file_path,
            len(text),
        )

        return text

    except UnicodeDecodeError:
        logger.exception(
            "Failed to decode text file as UTF-8 | file=%s",
            file_path,
        )
        raise

    except Exception:
        logger.exception(
            "Text parsing failed | file=%s",
            file_path,
        )
        raise
