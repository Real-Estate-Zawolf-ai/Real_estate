from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.rag.ocr.base import BaseOCRProvider

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# Registry of known provider names → module path, class name.
# Imports are lazy so missing optional dependencies don't crash at import time.
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "paddleocr": ("app.rag.ocr.paddle_provider", "PaddleOCRProvider"),
    "easyocr": ("app.rag.ocr.extractor", "EasyOCRProvider"),
}


def create_ocr_provider(
    settings: Settings | None = None,
    provider_name: str | None = None,
) -> BaseOCRProvider:
    """
    Factory function that creates the appropriate OCR provider based on
    configuration.

    Resolution order for the provider name:
        1. Explicit ``provider_name`` argument.
        2. ``settings.OCR_PROVIDER`` (if a Settings object is supplied).
        3. Default: ``"paddleocr"``.

    Args:
        settings: Optional pydantic Settings instance.
        provider_name: Explicit override for the provider name.

    Returns:
        An initialized :class:`BaseOCRProvider` subclass.

    Raises:
        ValueError: If the provider name is not recognized.
        ImportError: If the provider's dependency is not installed.
    """
    name = provider_name
    if name is None and settings is not None:
        name = getattr(settings, "OCR_PROVIDER", "paddleocr")
    if name is None:
        name = "paddleocr"

    name = name.lower().strip()

    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(
            f"Unknown OCR provider '{name}'. Available providers: {available}"
        )

    module_path, class_name = _PROVIDER_REGISTRY[name]

    # Dynamic import to keep optional deps lazy.
    import importlib
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"OCR provider '{name}' is selected but its dependency is not installed. "
            f"Install the required package or choose another provider.\n"
            f"  Available providers: {', '.join(sorted(_PROVIDER_REGISTRY))}"
        ) from exc

    provider_cls = getattr(module, class_name)

    # Build kwargs from settings when available.
    kwargs = _build_provider_kwargs(name, settings)

    logger.info("Creating OCR provider: %s", name)
    return provider_cls(**kwargs)


def _build_provider_kwargs(
    name: str,
    settings: Settings | None,
) -> dict:
    """
    Build constructor kwargs for a provider from Settings.
    """
    if settings is None:
        return {}

    if name == "paddleocr":
        return {
            "lang": "ar",  # Arabic model handles English chars too
            "use_gpu": settings.OCR_GPU,
            "model_dir": settings.OCR_MODEL_DIR,
        }

    if name == "easyocr":
        return {
            "languages": settings.OCR_LANGUAGES,
            "gpu": settings.OCR_GPU,
            "model_storage_directory": settings.OCR_MODEL_DIR,
        }

    return {}
