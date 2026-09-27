"""Resolve a user question to one lender and one authoritative manual version.

The resolver extracts explicit lender, version, and day-level date references,
then validates them against the corpus manifest. It deliberately fails closed:
ambiguous or contradictory scope produces a clarification instead of an
unfiltered search that could blend lenders or versions.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Any, TypedDict

from src.ingest.manifest import DocumentSpec, PROJECT_ROOT, load_manifest

DEFAULT_MANIFEST = "config/corpus_manifest.json"
VERSION_RE = re.compile(
    r"(?<![a-z0-9])(?:version\s*|v\s*)(\d+)(?!\d)", re.I
)
ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
MONTH_FIRST_RE = re.compile(
    rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
    re.I,
)
DAY_FIRST_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\s+(\d{{4}})\b",
    re.I,
)
HISTORICAL_WITHOUT_DATE_RE = re.compile(
    r"\b(historical|previous|prior|old|older)\s+(?:policy|manual|version|rule)",
    re.I,
)
PARTIAL_DATE_RE = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{4}}\b|(?<!\d)\d{{4}}-\d{{2}}(?!-\d)|"
    r"\b(?:in|during|as of)\s+20\d{2}\b",
    re.I,
)


class ScopeDecision(TypedDict, total=False):
    status: str
    reason: str
    lender_id: str
    lender_name: str
    document_id: str
    document_version: str
    effective_from: str
    effective_to: str | None
    filter_args: dict[str, Any]
    clarification_question: str


class ComparisonDecision(TypedDict, total=False):
    """Two independently authoritative document scopes for one comparison."""

    status: str
    reason: str
    scopes: list[ScopeDecision]
    clarification_question: str


COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|differ(?:ence|ent)?|same as|"
    r"contrast|relative to)\b",
    re.I,
)
ANTECEDENT_RE = re.compile(r"\b(?:that|this|it|previous|prior|above|other)\b", re.I)
CURRENT_RE = re.compile(r"\b(?:both\s+)?(?:current|latest|present)\b", re.I)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"manifest timestamp must include a timezone: {value}")
    return parsed


def _contains_alias(question: str, alias: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, question, flags=re.I) is not None


def _lender_aliases(specs: list[DocumentSpec]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for spec in specs:
        values = aliases.setdefault(spec.lender_id, set())
        values.update(
            {
                spec.lender_id,
                spec.lender_name,
                spec.lender_name.split()[0],
                spec.product_id.replace("_", " "),
                spec.product_name,
                spec.source_doc,
            }
        )
        camel_words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spec.lender_name.split()[0])
        values.add(camel_words)
    return aliases


def _mentioned_lenders(question: str, specs: list[DocumentSpec]) -> list[str]:
    aliases = _lender_aliases(specs)
    return sorted(
        lender_id
        for lender_id, values in aliases.items()
        if any(_contains_alias(question, alias) for alias in values)
    )


def _mentioned_lenders_in_order(question: str, specs: list[DocumentSpec]) -> list[str]:
    """Return distinct lender ids in textual order, not alphabetical order."""
    aliases = _lender_aliases(specs)
    found: list[tuple[int, str]] = []
    for lender_id, values in aliases.items():
        positions = []
        for alias in values:
            pattern = r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)"
            match = re.search(pattern, question, flags=re.I)
            if match:
                positions.append(match.start())
        if positions:
            found.append((min(positions), lender_id))
    return [lender_id for _, lender_id in sorted(found)]


def scope_key(scope: ScopeDecision) -> str:
    """Stable key used to keep comparison evidence partitioned in graph state."""
    return "::".join(
        str(scope.get(key) or "")
        for key in ("lender_id", "document_id", "document_version")
    )


def is_comparison_request(
    question: str,
    specs: list[DocumentSpec] | None = None,
) -> bool:
    """Conservative detector for explicit comparison wording, not scope."""
    specs = specs or _default_specs()
    lenders = _mentioned_lenders(question, specs)
    return bool(COMPARISON_RE.search(question) and (
        len(lenders) >= 2 or (len(lenders) == 1 and ANTECEDENT_RE.search(question))
    ))


def _extract_versions(question: str) -> list[str]:
    normalized = question.replace("_", " ")
    return sorted({f"V{number}" for number in VERSION_RE.findall(normalized)})


def _extract_dates(question: str) -> list[datetime]:
    found: dict[str, datetime] = {}
    for value in ISO_DATE_RE.findall(question):
        parsed = datetime.strptime(value, "%Y-%m-%d")
        found[parsed.date().isoformat()] = parsed
    for month, day, year in MONTH_FIRST_RE.findall(question):
        parsed = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
        found[parsed.date().isoformat()] = parsed
    for day, month, year in DAY_FIRST_RE.findall(question):
        parsed = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
        found[parsed.date().isoformat()] = parsed
    return [found[key] for key in sorted(found)]


def _clarification(reason: str, question: str) -> ScopeDecision:
    return {
        "status": "clarification",
        "reason": reason,
        "clarification_question": question,
    }


def _contains(spec: DocumentSpec, instant: datetime) -> bool:
    start = _timestamp(spec.effective_from).date()
    end = _timestamp(spec.effective_to).date() if spec.effective_to else None
    return start <= instant.date() and (end is None or instant.date() <= end)


def _resolved(
    spec: DocumentSpec,
    *,
    reason: str,
    as_of: datetime | None = None,
) -> ScopeDecision:
    filter_args: dict[str, Any] = {
        "lender_id": spec.lender_id,
        "document_id": spec.document_id,
        "document_version": spec.document_version,
    }
    if as_of is not None:
        filter_args["as_of"] = as_of.date().isoformat()
    elif reason == "default_current":
        filter_args["current_only"] = True

    return {
        "status": "resolved",
        "reason": reason,
        "lender_id": spec.lender_id,
        "lender_name": spec.lender_name,
        "document_id": spec.document_id,
        "document_version": spec.document_version,
        "effective_from": spec.effective_from,
        "effective_to": spec.effective_to,
        "filter_args": filter_args,
    }


def resolve_query_scope(
    question: str,
    specs: list[DocumentSpec] | None = None,
) -> ScopeDecision:
    """Resolve a single-document query or return a user-facing clarification."""
    specs = specs or _default_specs()
    lender_ids = _mentioned_lenders(question, specs)
    if not lender_ids:
        lenders = ", ".join(sorted({spec.lender_name for spec in specs}))
        return _clarification(
            "missing_lender",
            f"Which lender should I use: {lenders}?",
        )
    if len(lender_ids) > 1:
        return _clarification(
            "multiple_lenders",
            "This question names multiple lenders. Cross-lender comparison routing "
            "is handled separately; please ask about one lender for this query.",
        )

    lender_id = lender_ids[0]
    family = [spec for spec in specs if spec.lender_id == lender_id]
    document_ids = {spec.document_id for spec in family}
    if len(document_ids) != 1:
        return _clarification(
            "multiple_products",
            f"Which product for {family[0].lender_name} should I use?",
        )

    versions = _extract_versions(question)
    try:
        dates = _extract_dates(question)
    except ValueError:
        return _clarification(
            "invalid_date",
            "I could not interpret that date. Please provide a valid date such as 2026-06-15.",
        )
    if len(versions) > 1:
        return _clarification(
            "multiple_versions",
            "This question names multiple versions. Please ask about one version; "
            "version-comparison routing will retrieve each version separately.",
        )
    if len(dates) > 1:
        return _clarification(
            "multiple_dates",
            "Which single as-of date should govern this question?",
        )
    if not dates and PARTIAL_DATE_RE.search(question):
        return _clarification(
            "partial_date",
            "Please provide an exact as-of date, including the day.",
        )

    version = versions[0] if versions else None
    as_of = dates[0] if dates else None

    if version:
        matching = [spec for spec in family if spec.document_version == version]
        if not matching:
            available = ", ".join(sorted(spec.document_version for spec in family))
            return _clarification(
                "unknown_version",
                f"{family[0].lender_name} has {available}, not {version}. Which version should I use?",
            )
        selected = matching[0]
        if as_of is not None and not _contains(selected, as_of):
            return _clarification(
                "version_date_conflict",
                f"{selected.document_version} was not effective on {as_of.date().isoformat()}. "
                "Should I use the named version or the version active on that date?",
            )
        return _resolved(
            selected,
            reason="explicit_version_and_date" if as_of else "explicit_version",
            as_of=as_of,
        )

    if as_of is not None:
        matching = [spec for spec in family if _contains(spec, as_of)]
        if len(matching) != 1:
            return _clarification(
                "date_not_covered",
                f"No single {family[0].lender_name} manual covers "
                f"{as_of.date().isoformat()}. Please provide a version.",
            )
        return _resolved(matching[0], reason="as_of_date", as_of=as_of)

    if HISTORICAL_WITHOUT_DATE_RE.search(question):
        return _clarification(
            "underspecified_historical",
            "Which historical version or exact as-of date should I use?",
        )

    current = [spec for spec in family if spec.version_status == "current"]
    if len(current) != 1:
        return _clarification(
            "current_version_not_unique",
            f"The manifest does not identify one current manual for {family[0].lender_name}.",
        )
    return _resolved(current[0], reason="default_current")


def _comparison_clarification(reason: str, question: str) -> ComparisonDecision:
    return {"status": "clarification", "reason": reason, "clarification_question": question}


def _version_assignments(
    question: str,
    lender_ids: list[str],
    specs: list[DocumentSpec],
) -> tuple[dict[str, str], str | None]:
    """Attach explicit version mentions to their nearest named lender.

    An unattached version is accepted only with explicit "both" wording. This
    fails closed instead of guessing whether a lone V0 applies to the first,
    second, or both comparison targets.
    """
    normalized = question.replace("_", " ")
    matches = list(VERSION_RE.finditer(normalized))
    if not matches:
        return {}, None

    aliases = _lender_aliases(specs)
    lender_spans: list[tuple[int, int, str]] = []
    for lender_id in lender_ids:
        for alias in aliases[lender_id]:
            pattern = r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)"
            lender_spans.extend(
                (match.start(), match.end(), lender_id)
                for match in re.finditer(pattern, normalized, re.I)
            )

    assigned: dict[str, str] = {}
    for match in matches:
        version = f"V{match.group(1)}"
        if re.search(rf"\bboth\b[^.?!]{{0,30}}{re.escape(match.group(0))}", normalized, re.I):
            assigned.update({lender_id: version for lender_id in lender_ids})
            continue
        by_lender: dict[str, tuple[int, int]] = {}
        for start, end, lender_id in lender_spans:
            distance = min(abs(match.start() - end), abs(start - match.end()))
            candidate = (distance, start)
            if lender_id not in by_lender or candidate < by_lender[lender_id]:
                by_lender[lender_id] = candidate
        distances = sorted((distance, start, lender_id)
                           for lender_id, (distance, start) in by_lender.items())
        if distances and distances[0][0] <= 40 and (
            len(distances) == 1 or distances[0][0] < distances[1][0]
        ):
            lender_id = distances[0][2]
            if lender_id in assigned and assigned[lender_id] != version:
                return {}, "multiple_versions_for_target"
            assigned[lender_id] = version
        else:
            return {}, "ambiguous_comparison_version"
    return assigned, None


def _resolve_comparison_target(
    lender_id: str,
    family: list[DocumentSpec],
    *,
    version: str | None,
    as_of: datetime | None,
    preserve: ScopeDecision | None,
    force_current: bool,
) -> ScopeDecision | ComparisonDecision:
    if version:
        matching = [spec for spec in family if spec.document_version == version]
        if not matching:
            available = ", ".join(sorted(spec.document_version for spec in family))
            return _comparison_clarification(
                "unknown_version",
                f"{family[0].lender_name} has {available}, not {version}. Which version should I use?",
            )
        selected = matching[0]
        if as_of is not None and not _contains(selected, as_of):
            return _comparison_clarification(
                "version_date_conflict",
                f"{selected.document_version} was not effective on {as_of.date().isoformat()} for "
                f"{selected.lender_name}. Should I use the named version or the version active on that date?",
            )
        return _resolved(selected, reason="comparison_explicit_version", as_of=as_of)

    if as_of is not None:
        matching = [spec for spec in family if _contains(spec, as_of)]
        if len(matching) != 1:
            return _comparison_clarification(
                "date_not_covered",
                f"No single {family[0].lender_name} manual covers {as_of.date().isoformat()}.",
            )
        return _resolved(matching[0], reason="comparison_as_of_date", as_of=as_of)

    if (not force_current and preserve and preserve.get("status") == "resolved"
            and preserve.get("lender_id") == lender_id):
        matching = [spec for spec in family if spec.document_version == preserve.get("document_version")]
        if len(matching) == 1:
            return _resolved(matching[0], reason="prior_turn_exact_version")

    current = [spec for spec in family if spec.version_status == "current"]
    if len(current) != 1:
        return _comparison_clarification(
            "current_version_not_unique",
            f"The manifest does not identify one current manual for {family[0].lender_name}.",
        )
    return _resolved(current[0], reason="comparison_default_current")


def resolve_comparison_scopes(
    question: str,
    *,
    prior_scope: ScopeDecision | None = None,
    specs: list[DocumentSpec] | None = None,
) -> ComparisonDecision:
    """Resolve exactly two independently filtered lender/version targets."""
    specs = specs or _default_specs()
    lender_ids = _mentioned_lenders_in_order(question, specs)

    if len(lender_ids) == 1 and prior_scope and prior_scope.get("status") == "resolved":
        prior_lender = prior_scope.get("lender_id")
        if prior_lender and prior_lender not in lender_ids and COMPARISON_RE.search(question):
            lender_ids.insert(0, prior_lender)

    if len(lender_ids) < 2:
        return _comparison_clarification(
            "missing_comparison_target" if lender_ids else "missing_comparison_targets",
            "Which two lenders should I compare, and what policy topic does the comparison concern?",
        )
    if len(lender_ids) > 2:
        return _comparison_clarification(
            "too_many_comparison_targets",
            "The first comparison version supports exactly two lenders. Which two should I compare?",
        )
    if len(set(lender_ids)) != 2:
        return _comparison_clarification(
            "duplicate_comparison_target",
            "The comparison needs two different lenders. Which other lender should I use?",
        )

    try:
        dates = _extract_dates(question)
    except ValueError:
        return _comparison_clarification(
            "invalid_date", "I could not interpret that date. Please provide an exact date."
        )
    if len(dates) > 1:
        return _comparison_clarification(
            "multiple_dates", "Which single as-of date should govern the comparison?"
        )
    if not dates and PARTIAL_DATE_RE.search(question):
        return _comparison_clarification(
            "partial_date", "Please provide an exact as-of date, including the day."
        )

    versions, version_error = _version_assignments(question, lender_ids, specs)
    if version_error:
        return _comparison_clarification(
            version_error,
            "Attach each version to its lender—for example, 'NovaCred V0 versus LumenTrail V1'.",
        )

    force_current = bool(CURRENT_RE.search(question))
    as_of = dates[0] if dates else None
    scopes: list[ScopeDecision] = []
    for lender_id in lender_ids:
        family = [spec for spec in specs if spec.lender_id == lender_id]
        if not family:
            return _comparison_clarification(
                "unknown_lender", "One comparison lender is not in the corpus."
            )
        resolved = _resolve_comparison_target(
            lender_id,
            family,
            version=versions.get(lender_id),
            as_of=as_of,
            preserve=prior_scope,
            force_current=force_current,
        )
        if resolved.get("status") != "resolved":
            return resolved  # type: ignore[return-value]
        scopes.append(resolved)  # type: ignore[arg-type]

    return {"status": "resolved", "reason": "comparison", "scopes": scopes}


def _default_specs() -> list[DocumentSpec]:
    return load_manifest(DEFAULT_MANIFEST, root=PROJECT_ROOT)
