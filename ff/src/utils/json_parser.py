"""
JSON Parser Utility
====================
Robustly extracts JSON from LLM responses that may contain markdown fences.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(text: str) -> Any:
    """Try to parse JSON from an LLM response, stripping markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned).strip()
    return json.loads(cleaned)
