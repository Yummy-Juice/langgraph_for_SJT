"""Minimal IPIP-backed behavior-evidence resources."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sjt_system.runtime.io import write_json_atomic
from sjt_system.runtime.trace import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IPIP_SOURCE_PATH = PROJECT_ROOT / "docs" / "ipip.md"
DEFAULT_KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge_base"
DEFAULT_CORPUS_PATH = DEFAULT_KNOWLEDGE_ROOT / "items" / "ipip_neo_items.json"
CURATED_EVIDENCE_ROOT = DEFAULT_KNOWLEDGE_ROOT / "evidence_library"
CURATED_EVIDENCE_FILENAME = "stage2_evidence_library_curated.json"
# Offline mining remains available as a developer utility, but it is not a
# runtime source. Keeping its output outside knowledge_base prevents a formal
# run from silently consuming an unreviewed bundle.
DEFAULT_OFFLINE_BEHAVIOR_ROOT = PROJECT_ROOT / "outputs" / "behavior_evidence_candidates"

IPIP_SCHEMA_VERSION = "ipip-neo-items-v1"
BEHAVIOR_EVIDENCE_SCHEMA_VERSION = "behavior-evidence-v2"

NEO_FACET_CODE_TO_ID = {
    "N1": "neuroticism_anxiety",
    "N2": "neuroticism_angry_hostility",
    "N3": "neuroticism_depression",
    "N4": "neuroticism_self_consciousness",
    "N5": "neuroticism_impulsiveness",
    "N6": "neuroticism_vulnerability",
    "E1": "extraversion_warmth",
    "E2": "extraversion_gregariousness",
    "E3": "extraversion_assertiveness",
    "E4": "extraversion_activity",
    "E5": "extraversion_excitement_seeking",
    "E6": "extraversion_positive_emotions",
    "O1": "openness_fantasy",
    "O2": "openness_aesthetics",
    "O3": "openness_feelings",
    "O4": "openness_actions",
    "O5": "openness_ideas",
    "O6": "openness_values",
    "A1": "agreeableness_trust",
    "A2": "agreeableness_straightforwardness",
    "A3": "agreeableness_altruism",
    "A4": "agreeableness_compliance",
    "A5": "agreeableness_modesty",
    "A6": "agreeableness_tender_mindedness",
    "C1": "conscientiousness_competence",
    "C2": "conscientiousness_order",
    "C3": "conscientiousness_dutifulness",
    "C4": "conscientiousness_achievement_striving",
    "C5": "conscientiousness_self_discipline",
    "C6": "conscientiousness_deliberation",
}
NEO_FACET_ID_TO_CODE = {
    facet_id: code for code, facet_id in NEO_FACET_CODE_TO_ID.items()
}

_HEADER_PATTERN = re.compile(
    r"^(?P<code>[NEOAC][1-6]):\s*(?P<label>.+?)\s*"
    r"\((?:Alpha\s*=\s*)?(?P<alpha>\.\d+)\)\s*$",
    re.IGNORECASE,
)
_KEY_PATTERN = re.compile(
    r"^(?P<sign>[+\-−–])\s*keyed\s*(?P<text>.*)$",
    re.IGNORECASE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IPIPItem(StrictModel):
    item_id: str = Field(min_length=1)
    facet_code: str = Field(pattern=r"^[NEOAC][1-6]$")
    facet_id: str = Field(min_length=1)
    polarity: Literal["positive", "negative"]
    text: str = Field(min_length=1)
    source_line: int = Field(ge=1)


class IPIPFacetScale(StrictModel):
    facet_code: str = Field(pattern=r"^[NEOAC][1-6]$")
    facet_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    alpha: float = Field(gt=0, le=1)
    items: list[IPIPItem] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_polarities(self) -> "IPIPFacetScale":
        if {item.polarity for item in self.items} != {"positive", "negative"}:
            raise ValueError("每个 facet 必须同时包含正向和反向条目")
        return self


class IPIPCorpus(StrictModel):
    schema_version: Literal["ipip-neo-items-v1"]
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage_status: Literal[
        "user_supplied_source_requires_publication_rights_check"
    ]
    scales: list[IPIPFacetScale] = Field(min_length=30, max_length=30)


class BehaviorEvidenceDraft(StrictModel):
    behavior_dimension: str = Field(min_length=1, max_length=120)
    observable_behavior: str = Field(min_length=1, max_length=300)
    high_expression: str = Field(min_length=1, max_length=400)
    low_expression: str = Field(min_length=1, max_length=400)
    boundary_condition: str = Field(min_length=1, max_length=400)
    source_item_ids: list[str] = Field(min_length=1, max_length=20)


class BehaviorEvidenceAgentOutput(StrictModel):
    behavior_evidence: list[BehaviorEvidenceDraft] = Field(min_length=1)


class BehaviorEvidenceRecord(BehaviorEvidenceDraft):
    behavior_id: str = Field(min_length=1)


class BehaviorEvidenceBundle(StrictModel):
    schema_version: Literal["behavior-evidence-v2"]
    facet_code: str = Field(pattern=r"^[NEOAC][1-6]$")
    facet_id: str = Field(min_length=1)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str = Field(min_length=1)
    evidence: list[BehaviorEvidenceRecord] = Field(min_length=1)


def canonical_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def parse_ipip_markdown(path: Path = DEFAULT_IPIP_SOURCE_PATH) -> IPIPCorpus:
    source_path = Path(path)
    raw = source_path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    parsed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    polarity: Literal["positive", "negative"] | None = None
    counts: dict[tuple[str, str], int] = {}

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        header = _HEADER_PATTERN.match(line)
        if header:
            code = header.group("code").upper()
            current = {
                "facet_code": code,
                "facet_id": NEO_FACET_CODE_TO_ID.get(code),
                "source_label": header.group("label").strip(),
                "alpha": float(header.group("alpha")),
                "items": [],
            }
            if current["facet_id"] is None:
                raise ValueError(f"未知 IPIP facet code：{code}")
            parsed.append(current)
            polarity = None
            continue
        keyed = _KEY_PATTERN.match(line)
        if keyed:
            if current is None:
                raise ValueError(f"第 {line_number} 行出现在 facet 标题之前")
            polarity = "positive" if keyed.group("sign") == "+" else "negative"
            line = keyed.group("text").strip()
            if not line:
                continue
        if current is None:
            continue
        if polarity is None:
            raise ValueError(f"第 {line_number} 行缺少正反向标记")
        key = (current["facet_code"], polarity)
        counts[key] = counts.get(key, 0) + 1
        item_id = (
            f"{current['facet_code']}_"
            f"{'POS' if polarity == 'positive' else 'NEG'}_{counts[key]:02d}"
        )
        current["items"].append(
            {
                "item_id": item_id,
                "facet_code": current["facet_code"],
                "facet_id": current["facet_id"],
                "polarity": polarity,
                "text": line,
                "source_line": line_number,
            }
        )

    if [entry["facet_code"] for entry in parsed] != list(NEO_FACET_CODE_TO_ID):
        raise ValueError("IPIP facet 顺序或覆盖不完整")
    scales = [IPIPFacetScale.model_validate(entry) for entry in parsed]
    return IPIPCorpus(
        schema_version=IPIP_SCHEMA_VERSION,
        source_file=(
            str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if source_path.is_relative_to(PROJECT_ROOT)
            else str(source_path)
        ),
        source_sha256=sha256(raw).hexdigest(),
        corpus_hash=canonical_hash(
            [scale.model_dump(mode="json") for scale in scales]
        ),
        usage_status="user_supplied_source_requires_publication_rights_check",
        scales=scales,
    )


def save_ipip_corpus(
    corpus: IPIPCorpus,
    path: Path = DEFAULT_CORPUS_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, corpus.model_dump(mode="json"))
    return target


def load_ipip_corpus(path: Path = DEFAULT_CORPUS_PATH) -> IPIPCorpus:
    corpus = IPIPCorpus.model_validate_json(Path(path).read_text(encoding="utf-8"))
    expected = canonical_hash(
        [scale.model_dump(mode="json") for scale in corpus.scales]
    )
    if corpus.corpus_hash != expected:
        raise ValueError("IPIP corpus_hash 与内容不一致")
    return corpus


def get_ipip_scale(corpus: IPIPCorpus, facet_code: str) -> IPIPFacetScale:
    code = facet_code.upper()
    for scale in corpus.scales:
        if scale.facet_code == code:
            return scale
    raise ValueError(f"IPIP corpus 中不存在 facet：{facet_code}")


def behavior_source_fingerprint(
    facet: Mapping[str, Any],
    corpus: IPIPCorpus,
) -> str:
    return canonical_hash(
        {
            "facet": dict(facet),
            "corpus_hash": corpus.corpus_hash,
        }
    )


def create_behavior_evidence_bundle(
    *,
    facet_code: str,
    facet: Mapping[str, Any],
    corpus: IPIPCorpus,
    output: BehaviorEvidenceAgentOutput | Mapping[str, Any],
) -> BehaviorEvidenceBundle:
    result = (
        output
        if isinstance(output, BehaviorEvidenceAgentOutput)
        else BehaviorEvidenceAgentOutput.model_validate(output)
    )
    code = facet_code.upper()
    valid_source_ids = {
        item.item_id for item in get_ipip_scale(corpus, code).items
    }
    for draft in result.behavior_evidence:
        if len(draft.source_item_ids) != len(set(draft.source_item_ids)):
            raise ValueError("Behavior Evidence 的 source_item_ids 不得重复")
        unknown = set(draft.source_item_ids) - valid_source_ids
        if unknown:
            raise ValueError(
                "Behavior Evidence 引用了不属于当前 facet 的 IPIP 题号："
                + ", ".join(sorted(unknown))
            )
    return BehaviorEvidenceBundle(
        schema_version=BEHAVIOR_EVIDENCE_SCHEMA_VERSION,
        facet_code=code,
        facet_id=str(facet["facet_id"]),
        source_fingerprint=behavior_source_fingerprint(facet, corpus),
        generated_at=utc_timestamp(),
        evidence=[
            BehaviorEvidenceRecord(
                behavior_id=f"{code}_BE{index:02d}",
                **draft.model_dump(),
            )
            for index, draft in enumerate(result.behavior_evidence, start=1)
        ],
    )


def behavior_resource_path(
    facet_id: str,
    root: Path = DEFAULT_OFFLINE_BEHAVIOR_ROOT,
) -> Path:
    return Path(root) / f"{facet_id}.json"


def save_behavior_evidence_bundle(
    bundle: BehaviorEvidenceBundle,
    root: Path = DEFAULT_OFFLINE_BEHAVIOR_ROOT,
) -> Path:
    target = behavior_resource_path(bundle.facet_id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, bundle.model_dump(mode="json"))
    return target


def _convert_legacy_bundle(payload: Mapping[str, Any]) -> BehaviorEvidenceBundle:
    facet_id = str(payload["facet_id"])
    code = str(payload.get("facet_code") or NEO_FACET_ID_TO_CODE[facet_id])
    evidence = []
    for index, row in enumerate(payload.get("evidence") or [], start=1):
        evidence.append(
            {
                "behavior_id": str(row.get("behavior_id") or f"{code}_BE{index:02d}"),
                **{
                    field: deepcopy(row[field])
                    for field in BehaviorEvidenceDraft.model_fields
                    if field in row
                },
            }
        )
    return BehaviorEvidenceBundle(
        schema_version=BEHAVIOR_EVIDENCE_SCHEMA_VERSION,
        facet_code=code,
        facet_id=facet_id,
        source_fingerprint=str(
            payload.get("source_fingerprint")
            or payload.get("input_hash")
            or "0" * 64
        ),
        generated_at=str(payload.get("generated_at") or utc_timestamp()),
        evidence=evidence,
    )


def load_behavior_evidence_bundle(path: Path | str) -> BehaviorEvidenceBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") == BEHAVIOR_EVIDENCE_SCHEMA_VERSION:
        return BehaviorEvidenceBundle.model_validate(payload)
    return _convert_legacy_bundle(payload)


def curated_behavior_resource_path(facet_id: str) -> Path | None:
    """Return the curated source file for a registered facet, if present."""

    requested = str(facet_id).strip()
    if not requested:
        return None
    for path in sorted(
        CURATED_EVIDENCE_ROOT.glob(f"*/{CURATED_EVIDENCE_FILENAME}")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("facet_id") or "").strip() == requested:
            return path
    return None


def load_curated_behavior_evidence_bundle(
    path: Path | str,
    *,
    include_provisional: bool = False,
) -> BehaviorEvidenceBundle:
    """Convert one curated evidence-library file to the runtime bundle.

    The runtime schema intentionally keeps only behavior-level fields. Curated
    family metadata remains in the source file; stable family rows are the
    formal measurement units, while provisional rows are excluded by default.
    """

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    facet_id = str(payload.get("facet_id") or "").strip()
    facet_code = str(payload.get("neo_code") or "").strip().upper()
    if not facet_id or facet_code not in NEO_FACET_CODE_TO_ID:
        raise ValueError(
            f"curated evidence 缺少合法 facet_id/neo_code: {source_path}"
        )
    if NEO_FACET_CODE_TO_ID[facet_code] != facet_id:
        raise ValueError(
            f"curated evidence 的 neo_code 与 facet_id 不一致: {source_path}"
        )

    families = payload.get("evidence_library")
    if not isinstance(families, list):
        raise ValueError(f"curated evidence 缺少 evidence_library: {source_path}")

    selected = [
        family
        for family in families
        if isinstance(family, Mapping)
        and (
            include_provisional
            or str(family.get("status") or "").strip().lower() == "stable"
        )
    ]
    if not selected:
        raise ValueError(
            f"facet {facet_id} 没有可用于正式运行的 stable evidence family"
        )

    fingerprint = canonical_hash(payload)
    evidence: list[BehaviorEvidenceRecord] = []
    for family in selected:
        family_id = str(family.get("family_id") or "").strip()
        if not family_id:
            raise ValueError(f"curated evidence family 缺少 family_id: {source_path}")
        evidence.append(
            BehaviorEvidenceRecord(
                behavior_id=f"{facet_code}_{family_id}",
                behavior_dimension=str(family.get("family_name") or family_id),
                observable_behavior=str(family.get("definition") or ""),
                high_expression=str(family.get("high_trait_evidence") or ""),
                low_expression=str(family.get("low_trait_evidence") or ""),
                boundary_condition=str(family.get("boundary_condition") or ""),
                source_item_ids=[
                    str(item_id)
                    for item_id in family.get("supporting_item_ids") or []
                    if str(item_id).strip()
                ],
            )
        )

    if not all(evidence_row.source_item_ids for evidence_row in evidence):
        raise ValueError(
            f"curated evidence family 必须至少包含一个 supporting_item_id: {source_path}"
        )
    return BehaviorEvidenceBundle(
        schema_version=BEHAVIOR_EVIDENCE_SCHEMA_VERSION,
        facet_code=facet_code,
        facet_id=facet_id,
        source_fingerprint=fingerprint,
        generated_at=str(
            payload.get("generated_at") or f"curated:{fingerprint[:12]}"
        ),
        evidence=evidence,
    )


def find_behavior_evidence(facet_id: str) -> BehaviorEvidenceBundle | None:
    """Load only reviewed curated evidence; never fall back to legacy files."""

    path = curated_behavior_resource_path(facet_id)
    if path is None:
        return None
    return load_curated_behavior_evidence_bundle(path)


def attach_behavior_evidence(
    profile: Mapping[str, Any],
    bundles: Mapping[str, BehaviorEvidenceBundle],
) -> dict[str, Any]:
    enriched = deepcopy(dict(profile))
    for facet in enriched.get("facets") or []:
        if not isinstance(facet, dict):
            continue
        bundle = bundles.get(str(facet.get("facet_id")))
        if bundle is not None:
            facet["behavior_evidence"] = [
                row.model_dump(mode="json") for row in bundle.evidence
            ]
            facet["behavior_evidence_source"] = (
                "legacy_registry_fallback"
                if str(bundle.generated_at).startswith("legacy_registry_fallback:")
                else "curated"
            )
    snapshot = deepcopy(enriched)
    snapshot.pop("profile_hash", None)
    enriched["profile_hash"] = canonical_hash(snapshot)
    return enriched
