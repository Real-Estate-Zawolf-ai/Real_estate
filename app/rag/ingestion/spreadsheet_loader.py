from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def parse_spreadsheet(file_path: str) -> str:
    """
    Extract spreadsheet data and convert it into structured text
    suitable for downstream RAG processing.

    Supported formats:
        - CSV
        - XLSX
        - XLS

    Args:
        file_path: Path to the spreadsheet file.

    Returns:
        Spreadsheet content as structured text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
        Exception: If parsing fails.
    """

    logger.info("Starting spreadsheet parsing: %s", file_path)

    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Spreadsheet file not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported spreadsheet format: {extension}. "
                f"Supported formats: {SUPPORTED_EXTENSIONS}"
            )

        # ---------------------------------------------------------
        # Load spreadsheet
        # ---------------------------------------------------------

        if extension == ".csv":
            dataframe = pd.read_csv(path)

        else:
            dataframe = pd.read_excel(path)

        logger.info(
            "Spreadsheet loaded successfully | file=%s | rows=%d | columns=%d",
            file_path,
            len(dataframe),
            len(dataframe.columns),
        )

        if dataframe.empty:
            logger.warning(
                "Spreadsheet is empty: %s",
                file_path,
            )
            return ""

        # ---------------------------------------------------------
        # Clean data
        # ---------------------------------------------------------

        dataframe = dataframe.dropna(
            how="all"
        )

        dataframe = dataframe.fillna("")

        # Convert column names to strings
        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        # ---------------------------------------------------------
        # Convert rows into RAG-friendly text
        # ---------------------------------------------------------

        documents = []

        for index, row in dataframe.iterrows():

            row_data = []

            for column in dataframe.columns:

                value = str(row[column]).strip()

                if not value:
                    continue

                row_data.append(
                    f"{column}: {value}"
                )

            if row_data:
                document = (
                    f"Record {index + 1}\n"
                    + "\n".join(row_data)
                )

                documents.append(document)

        full_text = "\n\n".join(documents)

        logger.info(
            "Spreadsheet parsed successfully | file=%s | characters=%d | records=%d",
            file_path,
            len(full_text),
            len(documents),
        )

        return full_text

    except Exception:
        logger.exception(
            "Spreadsheet parsing failed | file=%s",
            file_path,
        )
        raise

