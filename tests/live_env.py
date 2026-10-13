"""Real credentials, captured once before the hermetic test setup wipes them.

Imported by ``conftest`` first. A module is only executed once however many times
conftest itself is imported (pytest and ``tests.conftest`` both load it), so the
snapshot reflects the true start-up environment.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

LIVE_ENV: dict[str, str] = {
    key: value
    for key, value in {**dotenv_values(Path(__file__).resolve().parents[1] / ".env"), **os.environ}.items()
    if value
}
