"""
Spreadsheet Loader.

Responsibilities:

    CSV / XLSX / XLS
        -> load spreadsheet
        -> clean rows and columns
        -> return structured records

The loader intentionally does NOT:
    - normalize property fields
    - create embeddings
    - chunk text
    - store vectors
    - perform retrieval

Those responsibilities belong to downstream RAG components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# Supported extensions
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}


# ============================================================
# Main parser
# ============================================================

def parse_spreadsheet(
    file_path: str,
) -> list[dict[str, Any]]:
    """
    Load a spreadsheet and return structured records.

    Supported formats:
        - CSV
        - XLSX
        - XLS

    Args:
        file_path:
            Path to the spreadsheet file.

    Returns:
        A list of dictionaries where each dictionary represents
        one spreadsheet row.

        Example:

            [
                {
                    "Property ID": "P001",
                    "Type": "Apartment",
                    "Price": 5000000,
                    "Bedrooms": 3,
                    "Location": "New Cairo",
                },
                ...
            ]

    Raises:
        FileNotFoundError:
            If the file does not exist.

        ValueError:
            If the file format is unsupported.

        TypeError:
            If the loaded spreadsheet cannot be converted
            into records.

        Exception:
            If spreadsheet parsing fails.
    """

    logger.info(
        "Starting spreadsheet parsing | file=%s",
        file_path,
    )

    try:
        path = Path(file_path)

        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if not path.exists():
            raise FileNotFoundError(
                f"Spreadsheet file not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Spreadsheet path is not a file: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported spreadsheet format: {extension}. "
                f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        # ----------------------------------------------------
        # Load spreadsheet
        # ----------------------------------------------------

        if extension == ".csv":
            dataframe = pd.read_csv(path)

        else:
            dataframe = pd.read_excel(path)

        logger.info(
            "Spreadsheet loaded successfully | "
            "file=%s | rows=%d | columns=%d",
            file_path,
            len(dataframe),
            len(dataframe.columns),
        )

        # ----------------------------------------------------
        # Empty spreadsheet
        # ----------------------------------------------------

        if dataframe.empty:
            logger.warning(
                "Spreadsheet is empty | file=%s",
                file_path,
            )

            return []

        # ----------------------------------------------------
        # Clean column names
        # ----------------------------------------------------

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        # Remove completely unnamed columns that can sometimes
        # appear in exported spreadsheets.
        dataframe = dataframe.loc[
            :,
            [
                column
                for column in dataframe.columns
                if column
                and not column.lower().startswith("unnamed:")
            ],
        ]

        # ----------------------------------------------------
        # Remove completely empty rows
        # ----------------------------------------------------

        dataframe = dataframe.dropna(
            how="all"
        )

        if dataframe.empty:
            logger.warning(
                "Spreadsheet contains no non-empty rows | file=%s",
                file_path,
            )

            return []

        # ----------------------------------------------------
        # Convert NaN / missing values
        # ----------------------------------------------------

        dataframe = dataframe.where(
            pd.notna(dataframe),
            None,
        )

        # ----------------------------------------------------
        # Convert dataframe to records
        # ----------------------------------------------------

        records: list[dict[str, Any]] = []

        for row_number, row in enumerate(
            dataframe.to_dict(orient="records"),
            start=1,
        ):
            cleaned_row: dict[str, Any] = {}

            for column, value in row.items():

                column_name = str(column).strip()

                if not column_name:
                    continue

                # Preserve non-string values such as:
                # int / float / bool where possible.
                if isinstance(value, str):
                    value = value.strip()

                    if not value:
                        continue

                elif pd.isna(value):
                    continue

                cleaned_row[column_name] = value

            # Skip rows where all values are empty.
            if not cleaned_row:
                continue

            records.append(cleaned_row)

        logger.info(
            "Spreadsheet parsed successfully | "
            "file=%s | records=%d",
            file_path,
            len(records),
        )

        return records

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ):
        logger.exception(
            "Spreadsheet validation failed | file=%s",
            file_path,
        )
        raise

    except Exception:
        logger.exception(
            "Spreadsheet parsing failed | file=%s",
            file_path,
        )
        raise
