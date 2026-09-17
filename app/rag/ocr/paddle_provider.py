from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.rag.ocr.base import BaseOCRProvider
from app.rag.ocr.schemas import OCRDetection

logger = logging.getLogger(__name__)


class PaddleOCRProvider(BaseOCRProvider):
    """
    OCR provider using PaddleOCR for Arabic and English recognition.

    Supports PaddleOCR >= 3.0.0.  The underlying PaddleOCR engine is
    initialized lazily on first use and reused for subsequent requests.

    Language handling:
        PaddleOCR accepts a single ``lang`` string.  For Arabic + English
        mixed documents use ``lang='ar'``, which also handles embedded
        English text and digits.  For purely English documents ``lang='en'``
        may yield better results.

    Result adapter:
        PaddleOCR returns results as::

            [  # per-image list
              [  # per-line list
                [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence) ],
                ...
              ]
            ]

        This provider maps each line into an :class:`OCRDetection` dataclass
        so the rest of the pipeline stays OCR-engine-agnostic.
    """

    def __init__(
        self,
        lang: str = "ar",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        show_log: bool = False,
        model_dir: str | None = None,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self.model_dir = model_dir
        self._engine = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy, thread-safe initialization
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Double-checked locking to ensure single-instance initialization."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    try:
                        from paddleocr import PaddleOCR
                    except ImportError as exc:
                        raise ImportError(
                            "paddleocr is not installed.  Install it with:\n"
                            "  pip install paddlepaddle paddleocr\n"
                            "See https://github.com/PaddlePaddle/PaddleOCR for details."
                        ) from exc

                    logger.info(
                        "Initializing PaddleOCR engine | lang=%s | use_gpu=%s",
                        self.lang,
                        self.use_gpu,
                    )

                    kwargs: dict[str, Any] = {
                        "lang": self.lang,
                        "use_gpu": self.use_gpu,
                        "use_angle_cls": self.use_angle_cls,
                        "show_log": self.show_log,
                    }
                    if self.model_dir:
                        kwargs["det_model_dir"] = self.model_dir
                        kwargs["rec_model_dir"] = self.model_dir

                    self._engine = PaddleOCR(**kwargs)
        return self._engine

    # ------------------------------------------------------------------
    # BaseOCRProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            return True
        except ImportError:
            return False

    def read_image(
        self,
        image_path: str | Path,
        **kwargs: Any,
    ) -> list[OCRDetection]:
        engine = self._get_engine()
        path_str = str(image_path)

        logger.debug("Running PaddleOCR on: %s", path_str)
        raw_results = engine.ocr(path_str, cls=self.use_angle_cls)

        return self._adapt_results(raw_results)

    def read_image_bytes(
        self,
        buffer: bytes,
        **kwargs: Any,
    ) -> list[OCRDetection]:
        """
        Process in-memory image bytes without writing to disk.
        Converts bytes to a numpy array and feeds it to PaddleOCR.
        """
        engine = self._get_engine()

        try:
            import numpy as np
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(buffer))
            img_array = np.array(img)
        except ImportError:
            # Fallback: use base class temp-file approach
            return super().read_image_bytes(buffer, **kwargs)

        logger.debug("Running PaddleOCR on in-memory image (%d bytes)", len(buffer))
        raw_results = engine.ocr(img_array, cls=self.use_angle_cls)

        return self._adapt_results(raw_results)

    # ------------------------------------------------------------------
    # Result adapter
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_results(
        raw_results: list | None,
    ) -> list[OCRDetection]:
        """
        Convert PaddleOCR raw output into internal OCRDetection objects.

        PaddleOCR output format::

            [  # one element per image (usually 1)
              [  # list of detected lines
                [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('recognized text', 0.95) ],
                ...
              ]
            ]

        Returns an empty list when PaddleOCR returns None or empty results.
        """
        detections: list[OCRDetection] = []

        if not raw_results:
            return detections

        for page_result in raw_results:
            if not page_result:
                continue

            for line in page_result:
                if not line or len(line) < 2:
                    continue

                bbox = line[0]
                text_conf = line[1]

                if not text_conf or not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                    continue

                text = str(text_conf[0]).strip()
                confidence = float(text_conf[1])

                if text:
                    detections.append(
                        OCRDetection(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )

        return detections
