from pathlib import Path
import logging
from unstructured.partition.auto import partition


logger = logging.getLogger(__name__)

def parse_office(file_path:str)->str:
    """
    Parse Office documents such as DOCX and PPTX using Unstructured.

    Unstructured automatically detects the document type
    and extracts its textual content.
    """
    logger.info("Starting Office document parsing: %s", file_path)
    
    try:
        path = Path(file_path)
        if not path.exists:
            raise FileNotFoundError(
                f"Office file not found: {file_path}"
            )
        if path.suffix.lower() not in {".docx", ".pptx"}:
            raise ValueError(
                f"Unsupported Office format: {path.suffix}"
            )
        elements = partition()            