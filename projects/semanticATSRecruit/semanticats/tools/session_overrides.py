from __future__ import annotations

import re
from typing import Any


def parse_override(command: str) -> dict[str, Any]:
    lower = command.casefold()
    if match := re.search(r"without\s+([A-Za-z0-9+#.\- ]+)", command, re.IGNORECASE):
        return {"exclude_missing_skill": match.group(1).strip()}
    if match := re.search(r"with\s+(\d+)\+?\s+years", lower):
        return {"min_years_experience": float(match.group(1))}
    if match := re.search(r"boost candidate\s+(.+)", command, re.IGNORECASE):
        return {"boost_candidate": match.group(1).strip()}
    if match := re.search(r"with\s+([A-Za-z0-9+#.\- ]+)\s+experience", command, re.IGNORECASE):
        return {"require_skill": match.group(1).strip()}
    return {"note": command}


def apply_filters(candidates: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    output = list(candidates)
    required = filters.get("require_skill") or filters.get("exclude_missing_skill")
    if required:
        output = [
            candidate
            for candidate in output
            if required.casefold() in " ".join(candidate.get("matched_skills", [])).casefold()
            or required.casefold() in " ".join(candidate.get("evidence", [])).casefold()
        ]
    min_years = filters.get("min_years_experience")
    if min_years is not None:
        output = [
            candidate
            for candidate in output
            if float(candidate.get("years_experience") or 0.0) >= float(min_years)
        ]
    boosted = filters.get("boost_candidate")
    if boosted:
        for candidate in output:
            if boosted.casefold() in candidate.get("candidate_name", "").casefold():
                candidate["score"] = float(candidate.get("score", 0)) + 0.25
        output.sort(key=lambda item: item.get("score", 0), reverse=True)
    return output
