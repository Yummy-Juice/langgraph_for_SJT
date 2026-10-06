"""Repair-only open-action probes. Never produces formal response records."""

from __future__ import annotations

import asyncio
import errno
import json
import math
import os
import random
import shutil
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from typing_extensions import TypedDict
from sjt_system.evaluation.repair_recovery import MAX_CALLS, POLICY, RepairStopped

PROTOCOL = "scenario_detection_first_v1"
DETECTION_PROTOCOL = "adjacent_differences_v1"
EDIT_PROTOCOL = "difference_guided_options_v1"
FIELDS = ("initial_ideas", "practical_action", "when_facilitated", "when_blocked")
LEVELS = ("low", "medium_low", "medium_high", "high")
ADJACENT_LEVELS = tuple(zip(LEVELS, LEVELS[1:]))
NO_DIFFERENCE = "无明显差异"
SCENARIO_LENGTH_INSTRUCTION = (
    "简体中文题干通常为30–50个字符，且不得超过100字。题干情境字段只能包含"
    "描述情境本身、理解事件和决策机会所必需的内容；不得附加标题、问题、"
    "作答指令、选项、评分、解释、构念标签或评论。"
)
MAX_REWRITES = 5
DESIGN_LIMIT_REASON = "design_limit: scenario, skeleton and blueprint budgets exhausted"
RECOVERY_VERSION = 3
ARCHIVE_REPLACE_ATTEMPTS = 10
Invoke = Callable[[str, dict[str, Any]], Awaitable[Any]]

_ARCHIVE_SAVE_LOCK = threading.Lock()


def _replace_archive_with_retry(source: Path, destination: Path) -> None:
    """Replace a repair archive despite brief Windows sharing violations."""

    for attempt in range(ARCHIVE_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            errno_value = getattr(exc, "errno", None)
            retryable = winerror in {32, 33} or errno_value in {
                errno.EACCES,
                errno.EBUSY,
                errno.EPERM,
            }
            if not retryable or attempt + 1 >= ARCHIVE_REPLACE_ATTEMPTS:
                raise
            time.sleep(0.05 * (2 ** attempt))


class ProbeUnavailable(ValueError):
    """Missing evidence or invalid output requires a durable pause."""


class OpenAction(TypedDict):
    initial_ideas: str
    practical_action: str
    when_facilitated: str
    when_blocked: str


class AdjacentDifference(TypedDict):
    pair_id: str
    difference: str


class AdjacentDifferences(TypedDict):
    records: list[AdjacentDifference]


class KnowledgeUse(TypedDict):
    entry_id: str
    revision: int
    use: str


class ScenarioPatch(TypedDict):
    scenario: str
    reason: str
    knowledge_used: list[KnowledgeUse]


class OptionText(TypedDict):
    option_id: str
    text: str


class OptionsPatch(TypedDict):
    response_options: list[OptionText]
    reason: str
    knowledge_used: list[KnowledgeUse]


PROMPTS = {
    "open_action": """扮演给定角色，回答当前情境中的“你会怎么做”。
角色只规定一个人格facet及其表现水平，其他人格不作设定，不自行补成中等水平。
根据构念定义和行为锚点自然作答，不评判题目，不猜测测量目标。
只输出 initial_ideas（行动之前的想法）、practical_action（实际采取的行动）、
when_facilitated（遇到促进时采取的行动）、when_blocked（遇到阻碍时采取的行动）四个字段。
每格仅用一个简短短语或短句给出最核心的想法或具体行为；不铺陈背景，不写空泛评价、
等级名称、解释或重复套话。促进与阻碍下的行动要紧扣当前情境，不另编长故事。""",
    "compare_adjacent": """独立阅读每个匿名配对的left和right四列表现，
只依据提供的目标facet构念定义和行为锚点，比较双方在该目标facet上的核心表现差异。
综合四列，每个pair_id只总结一个最核心的差异，用一句简短描述表达；
不逐列罗列，不输出多个差异，不添加理由、评分、正负编码或泛泛评价。
不能把措辞、篇幅或非目标人格差异当作目标facet差异，不推测未写出的行为。
如果不存在明显差异，difference必须原样输出“无明显差异”，不要勉强制造差异。
输入不含角色组别或等级，不推断组别、等级或预期结果，不根据其他配对补齐差异。
只返回records列表，每条恰好包含pair_id和difference，完整覆盖输入配对且不重复。""",
    "rewrite_scenario": """根据开放行动检测证据修改情境文本。输出 scenario、reason 和 knowledge_used。
保留心理骨架、激活机制、构念、行为证据、蓝图约束及同一情境任务，
不能改选项、评分键、行为等级或任何身份字段。
目标组低—中低、中低—中高、中高—高三对均应自然产生目标facet上的明显核心差异；
每个非目标组至少一对在目标facet上无明显差异，不要求全部相邻对都无差异或反转端点。
依据失败相邻对的差异摘要和原始四列表现，调整情境中的机会、促进、阻碍或线索，
不把角色等级、预期答案或门槛写进情境。
不得仅改空格或返回相同情境。reason简述改变与本轮证据的关系。
必须参考failure_memory中的历次修改及其后续失败，不重复已经失败的情境。
参考repair_knowledge中适用的经验假设，不将其当作已证明机制或覆盖固定约束。
knowledge_used列出实际采用的entry_id、revision和use用途，无适用知识则为空列表。
""" + SCENARIO_LENGTH_INSTRUCTION,
    "rewrite_options": """根据已经通过的情境检测及构念约束，联合改写授权选项的文本。
只输出 response_options（option_id、text）、reason和knowledge_used；恰好覆盖 authorized_option_ids，
每个选项只出现一次。保留现有option_id、行为等级、计分键和情境，不改任何固定字段。
简体中文中，每个新写或改写的选项通常为15–25个字符，且不得超过25字；选项长度应大致可比。
必须高度依据target_adjacent_differences中的三条目标差异及选项等级对应关系修改核心行为，
以最终四列表现为直接依据，不仅润色措辞，不脱离差异另造行为梯度。
若mode为reconstruct_all，按三条差异连接四个行为等级，吸收最终开放行动及情境修改原因重构四选项。
否则一次性联合处理全部指定相邻对，强化同一目标构念的行为差异，不改变未授权选项。
若mode为joint_non_target，联合修改四选项，仍重点强化pairs中的目标行为差异，
同时参考non_target_adjacent_differences与non_target_findings避免强化非目标污染，
不强制反转端点，不要求非目标组三对全无差异，不使四选项行为相同。
参考repair_knowledge中适用的经验假设；knowledge_used列出采用的entry_id、revision和use，无适用则为空。
不要机械按是否行动计分，不引入新的非目标人格要求，不声称修改已经改善正式统计指标。""",
}
SCHEMAS = {
    "open_action": OpenAction, "compare_adjacent": AdjacentDifferences,
    "rewrite_scenario": ScenarioPatch, "rewrite_options": OptionsPatch,
}


def make_model_invoker(state: Mapping[str, Any]) -> tuple[Invoke, dict[str, Any]]:
    from sjt_system.agent.agent_factory import PSYCHOMETRIC_REASONING_ROLE_MANIFEST
    repair = PSYCHOMETRIC_REASONING_ROLE_MANIFEST["psychometric_item_repair"]
    configured_id = str((state.get("virtual_sample_config") or {}).get("model_id") or "").strip()
    thinking = os.getenv("VIRTUAL_RESPONDENT_THINKING", "enabled").strip().lower() if configured_id else None
    virtual = {
        "model_id": configured_id or os.getenv("MODEL_ID", "deepseek-v4-flash"),
        "temperature": None, "thinking_type": thinking,
        "reasoning_effort": (os.getenv("VIRTUAL_RESPONDENT_REASONING_EFFORT", "high").strip().lower()
                             if thinking == "enabled" else None),
    }
    editing = {"model_id": repair["model_id"], "temperature": repair.get("temperature"),
               "thinking_type": repair.get("thinking"), "reasoning_effort": repair.get("reasoning_effort")}
    metadata = {kind: deepcopy(virtual if kind == "open_action" else editing) for kind in SCHEMAS}

    async def invoke(kind: str, payload: dict[str, Any]) -> Any:
        from sjt_system.agent.agent_factory import create_agent
        from sjt_system.agent.client import get_model_request_timeout_seconds
        agent = create_agent(PROMPTS[kind], SCHEMAS[kind], **metadata[kind])
        # The journal owns the complete retry budget; the client has max_retries=0.
        return await asyncio.wait_for(
            agent.ainvoke({"input_data": json.dumps(payload, ensure_ascii=False)}),
            timeout=get_model_request_timeout_seconds(),
        )

    return invoke, metadata


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _text(value: Any, label: str, *, limit: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeUnavailable(f"{label}: missing text")
    if limit is not None and (len(value.strip()) > limit or "\n" in value.strip()):
        raise ProbeUnavailable(f"{label}: expected one short description")
    return value.strip()


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProbeUnavailable(f"{label}: expected fields {sorted(expected)}")


def build_detection_plan(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze groups from formal signed-rho winners, without recomputing metrics."""
    if evidence.get("planning_error"):
        raise ProbeUnavailable(str(evidence["planning_error"]))
    observations = {row["observation_id"]: row for row in evidence.get("observations", [])
                    if isinstance(row, Mapping) and row.get("observation_id")}
    failures = {}
    snapshot = {}
    # Profile target-rho and VTS values are diagnostic-only in the current
    # facet iteration contract.  Their thresholds are intentionally ``None``
    # and they must not block evidence review.  Keep only the active item
    # gates here; a failed facet form may authorize investigation even when
    # all four item gates pass.
    formal_gate_names = (
        "CITC",
        "ITEM_TARGET_HEDGES_G",
        "ITEM_TARGET_IPIP_RHO",
        "ITEM_DELTA_MIN",
    )
    for name in formal_gate_names:
        row = observations.get(f"OBS:{name}", {})
        # Historical evidence bundles predate the three IPIP item gates. They
        # remain readable, while current bundles make these rows mandatory.
        if not row and name.startswith("ITEM_"):
            continue
        value, threshold = _number(row.get("value")), _number(row.get("threshold"))
        if value is None or threshold is None:
            raise ProbeUnavailable(f"missing formal metric/threshold: {name}")
        failures[name] = value < threshold
        snapshot[name] = deepcopy(dict(row))
    facet_form_failure = evidence.get("facet_form_failure")
    if not any(failures.values()) and not isinstance(facet_form_failure, Mapping):
        raise ProbeUnavailable("all formal gates pass; repair is not authorized")
    constraints = [deepcopy(dict(row)) for row in evidence.get("normal_constraints", [])
                   if isinstance(row, Mapping)]

    def statement(prefix: str) -> str:
        texts = [row.get("statement") for row in constraints
                 if str(row.get("constraint_id", "")).startswith(prefix)]
        if not texts:
            raise ProbeUnavailable(f"missing construct material: {prefix}")
        return "\n".join(_text(text, prefix) for text in texts)

    target_id = _text((evidence.get("current_item") or {}).get("target_dimension_id"),
                      "target_dimension_id")
    target = {"definition": statement("FACET_DEFINITION:"),
              "high_anchor": statement("BE_HIGH:"), "low_anchor": statement("BE_LOW:")}
    groups = [{"group": "target", "facet_id": target_id, "construct": target}]
    for arm in ("same_domain", "cross_domain"):
        # Preserve legacy formal VTS behavior for old archives that explicitly
        # supplied a threshold, while ignoring current diagnostic-only rows.
        vts = observations.get(f"OBS:{arm.upper()}_VTS", {})
        vts_value, vts_threshold = _number(vts.get("value")), _number(vts.get("threshold"))
        vts_is_formal = vts.get("filtering_authority") is not False and vts_threshold is not None
        if not (vts_is_formal and vts_value < vts_threshold):
            continue
        selected = observations.get(f"OBS:{arm.upper()}_MAX_NON_TARGET_RHO", {})
        facet_id = _text(selected.get("dimension_id"), f"{arm} selected facet")
        if facet_id == target_id or _number(selected.get("signed_rho")) is None:
            raise ProbeUnavailable(f"invalid formal non-target winner: {arm}")
        prefix = f"NON_TARGET_{arm.upper()}:{facet_id}:"
        groups.append({"group": arm, "facet_id": facet_id,
                       "formal_winner": deepcopy(dict(selected)),
                       "construct": {"definition": statement(prefix + "DEFINITION"),
                                     "high_anchor": statement(prefix + "HIGH"),
                                     "low_anchor": statement(prefix + "LOW")}})
    return {"protocol": PROTOCOL, "groups": groups, "formal_snapshot": snapshot,
            "target_construct": target, "constraints": constraints,
            "fixed_skeleton": deepcopy(evidence.get("fixed_skeleton") or {}),
            "respondent_count": 4 * len(groups)}


def build_roles(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"group": group["group"], "facet_id": group["facet_id"], "level": level,
             "persona": {"construct": deepcopy(group["construct"]), "level": level}}
            for group in plan["groups"] for level in LEVELS]


def validate_actions(value: Any) -> dict[str, str]:
    _keys(value, set(FIELDS), "open action")
    return {field: _text(value[field], field, limit=200) for field in FIELDS}


def role_positions(roles: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    groups = {}
    for index, role in enumerate(roles):
        group, level = role["group"], role["level"]
        if group not in ("target", "same_domain", "cross_domain") or level not in LEVELS:
            raise ProbeUnavailable("unknown probe group or level")
        positions = groups.setdefault(group, {})
        if level in positions:
            raise ProbeUnavailable(f"duplicate role level: {group}/{level}")
        positions[level] = index
    if "target" not in groups or any(set(p) != set(LEVELS) for p in groups.values()):
        raise ProbeUnavailable("incomplete role levels or missing target group")
    return groups


def build_adjacent_pairs(roles: list[dict[str, Any]], rows: list[dict[str, Any]],
                         *, used_ids: set[str] | None = None) -> dict[str, Any]:
    if len(rows) != len(roles):
        raise ProbeUnavailable("incomplete open actions")
    actions = [validate_actions(row["action"]) for row in rows]
    anonymous = []
    for group, positions in role_positions(roles).items():
        for lower, higher in ADJACENT_LEVELS:
            anonymous.append(({"left": deepcopy(actions[positions[lower]]),
                               "right": deepcopy(actions[positions[higher]])},
                              {"group": group, "lower_level": lower, "higher_level": higher}))
    random.SystemRandom().shuffle(anonymous)
    records, identity = [], {}
    used = set(used_ids or ())
    for record, position in anonymous:
        pair_id = f"p{uuid4().hex[:8]}"
        while pair_id in used:
            pair_id = f"p{uuid4().hex[:8]}"
        used.add(pair_id)
        records.append({"pair_id": pair_id, **record})
        identity[pair_id] = position
    return {"records": records, "identity": identity}


def rekey_legacy_pairs(packet: Mapping[str, Any], *, used_ids: set[str] | None = None) -> dict[str, Any]:
    """Give a failed legacy comparison new opaque IDs without repeating actions."""
    records, identity = [], {}
    used = set(used_ids or ()) | set(packet["identity"])
    for row in packet["records"]:
        old_id = row["pair_id"]
        new_id = f"p{uuid4().hex[:8]}"
        while new_id in used:
            new_id = f"p{uuid4().hex[:8]}"
        used.add(new_id)
        records.append({**deepcopy(row), "pair_id": new_id})
        identity[new_id] = deepcopy(packet["identity"][old_id])
    return {"records": records, "identity": identity}


def validate_differences(value: Any, pairs: list[dict[str, Any]]) -> dict[str, str]:
    _keys(value, {"records"}, "adjacent differences")
    source = {row["pair_id"] for row in pairs}
    if len(source) != len(pairs):
        raise ProbeUnavailable("duplicate source pair ID")
    if not isinstance(value["records"], list) or len(value["records"]) != len(source):
        raise ProbeUnavailable("missing or duplicate adjacent differences")
    result = {}
    for row in value["records"]:
        _keys(row, {"pair_id", "difference"}, "adjacent difference")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or pair_id not in source or pair_id in result:
            raise ProbeUnavailable("unknown or duplicate anonymous pair ID")
        result[pair_id] = _text(row["difference"], "difference")
    return result


async def compare_actions_blindly(target_construct: Mapping[str, Any],
                                 pairs: list[dict[str, Any]], invoke: Invoke) -> dict[str, str]:
    # Identity, groups, levels and acceptance rules remain local.
    payload = {"construct": deepcopy(dict(target_construct)), "pairs": deepcopy(pairs)}
    return validate_differences(await invoke("compare_adjacent", payload), pairs)


def evaluate_differences(roles: list[dict[str, Any]], identity: Mapping[str, Any],
                         differences: Mapping[str, str]) -> dict[str, Any]:
    groups = role_positions(roles)
    if set(differences) != set(identity) or len(identity) != 3 * len(groups):
        raise ProbeUnavailable("incomplete adjacent differences")
    by_pair = {}
    for pair_id, position in identity.items():
        _keys(position, {"group", "lower_level", "higher_level"}, "pair identity")
        group = position["group"]
        adjacent = (position["lower_level"], position["higher_level"])
        key = (group, *adjacent)
        if group not in groups or adjacent not in ADJACENT_LEVELS or key in by_pair:
            raise ProbeUnavailable("invalid or duplicate adjacent pair mapping")
        difference = _text(differences[pair_id], "difference")
        by_pair[key] = {"pair_id": pair_id, "lower_level": adjacent[0], "higher_level": adjacent[1],
                        "difference": difference, "has_difference": difference != NO_DIFFERENCE}
    results, failures = {}, []
    for group in groups:
        summaries = [by_pair[(group, *pair)] for pair in ADJACENT_LEVELS]
        missing = [row for row in summaries if not row["has_difference"]]
        passed = not missing if group == "target" else bool(missing)
        results[group] = {"differences": summaries, "passes": passed,
                          "status": "pass" if passed else "fail"}
        if not passed:
            failures.append({"group": group,
                "reason": "target_difference_missing" if group == "target" else "non_target_all_different",
                "pairs": deepcopy(missing if group == "target" else summaries)})
    return {"passes": not failures, "groups": results, "failures": failures,
            "status": "fail" if failures else "pass"}


def target_option_differences(item: Mapping[str, Any], decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(item["scoring_key"], key=item["scoring_key"].get)
    summaries = decision["groups"]["target"]["differences"]
    return [{**deepcopy(row), "lower_option_id": ordered[index], "higher_option_id": ordered[index + 1]}
            for index, row in enumerate(summaries)]


def plan_option_edit(item: Mapping[str, Any], gradient: Mapping[str, Any],
                     *, scenario_changed: bool, non_target_findings=None) -> dict[str, Any]:
    options = item.get("response_options") or []
    ids = [row["option_id"] for row in options]
    scores = item.get("scoring_key") or {}
    if len(ids) != 4 or len(set(ids)) != 4 or set(scores) != set(ids):
        raise ProbeUnavailable("expected four options and matching scoring key")
    if any(_number(scores[key]) is None for key in ids) or len(set(scores.values())) != 4:
        raise ProbeUnavailable("scoring key must contain four distinct numeric scores")
    ordered = sorted(ids, key=lambda key: scores[key])
    if scenario_changed:
        return {"mode": "reconstruct_all", "authorized_option_ids": ordered, "pairs": []}
    adjacent = list(zip(ordered, ordered[1:]))
    if gradient.get("passes") is False:
        failed = gradient.get("failed_adjacent_pairs")
        if not isinstance(failed, list) or not failed:
            raise ProbeUnavailable("failed gradient has no failed adjacent pairs")
        pairs = {(row.get("lower_option_id"), row.get("higher_option_id")) for row in failed}
        if not pairs.issubset(set(adjacent)):
            raise ProbeUnavailable("failed gradient pair conflicts with scoring key")
        selected = [pair for pair in adjacent if pair in pairs]
    elif gradient.get("passes") is True:
        means = {row.get("option_id"): _number(row.get("target_facet_mean"))
                 for row in gradient.get("options", [])}
        if any(means.get(key) is None for key in ordered):
            raise ProbeUnavailable("passing gradient has missing option means")
        if any(means[high] <= means[low] for low, high in adjacent):
            raise ProbeUnavailable("passing gradient conflicts with recorded option means")
        selected = [min(adjacent, key=lambda pair: means[pair[1]] - means[pair[0]])]
    else:
        raise ProbeUnavailable("missing formal target option gradient")
    affected = {key for pair in selected for key in pair}
    if non_target_findings:
        return {"mode": "joint_non_target", "pairs": selected, "authorized_option_ids": ordered,
                "non_target_findings": deepcopy(non_target_findings)}
    return {"mode": "adjacent_union", "pairs": selected,
            "authorized_option_ids": [key for key in ordered if key in affected]}


def apply_text_patch(item: Mapping[str, Any], patch: Any, *,
                     option_ids: list[str] | None = None) -> dict[str, Any]:
    """New-protocol-only scope validator; fixed fields are never model writable."""
    candidate = deepcopy(dict(item))
    if option_ids is None:
        _keys(patch, {"scenario", "reason"} | ({"knowledge_used"} if isinstance(patch, Mapping) and "knowledge_used" in patch else set()), "scenario patch")
        _text(patch["reason"], "rewrite reason")
        scenario = _text(patch["scenario"], "scenario")
        if "".join(scenario.split()) == "".join(str(item.get("scenario", "")).split()):
            raise ProbeUnavailable("scenario rewrite is unchanged")
        candidate["scenario"] = scenario
    else:
        _keys(patch, {"response_options", "reason"} | ({"knowledge_used"} if isinstance(patch, Mapping) and "knowledge_used" in patch else set()), "option patch")
        _text(patch["reason"], "rewrite reason")
        rows = patch["response_options"]
        if not isinstance(rows, list) or len(rows) != len(option_ids):
            raise ProbeUnavailable("option patch must cover exactly the authorized options")
        texts = {}
        for row in rows:
            _keys(row, {"option_id", "text"}, "option text")
            key = row["option_id"]
            if not isinstance(key, str) or key not in option_ids or key in texts:
                raise ProbeUnavailable("duplicate or unauthorized option ID")
            texts[key] = _text(row["text"], "option text")
        for row in candidate["response_options"]:
            if row["option_id"] in texts:
                updated = texts[row["option_id"]]
                if "".join(updated.split()) == "".join(row["text"].split()):
                    raise ProbeUnavailable(f"authorized option unchanged: {row['option_id']}")
                row["text"] = updated
    return candidate


def validate_knowledge_usage(patch, snapshot):
    if snapshot is None:
        return
    uses = patch.get("knowledge_used") if isinstance(patch, Mapping) else None
    if not isinstance(uses, list):
        raise ProbeUnavailable("missing knowledge_used audit list")
    available = {(row["entry_id"], row["revision"]) for row in snapshot.get("entries", [])}
    seen = set()
    for row in uses:
        _keys(row, {"entry_id", "revision", "use"}, "knowledge use")
        if not isinstance(row["entry_id"], str) or not isinstance(row["revision"], int) or isinstance(row["revision"], bool):
            raise ProbeUnavailable("invalid knowledge citation identity/version")
        key = (row["entry_id"], row["revision"])
        if key not in available or key in seen:
            raise ProbeUnavailable("unknown or duplicate knowledge citation")
        _text(row["use"], "knowledge use")
        seen.add(key)


def validate_candidate(original: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    _text(candidate.get("scenario"), "candidate scenario")
    restored = deepcopy(dict(candidate))
    restored["scenario"] = original.get("scenario")
    old_options = original.get("response_options") or []
    new_options = restored.get("response_options") or []
    if len(old_options) != len(new_options):
        raise ProbeUnavailable("candidate changed option count")
    for old, new in zip(old_options, new_options):
        _text(new.get("text"), "candidate option")
        new["text"] = old.get("text")
    if restored != original:
        raise ProbeUnavailable("candidate changed fixed fields or option order")
    if candidate == original:
        raise ProbeUnavailable("candidate contains no text changes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepairArchive:
    """Per-item journal: flush after every response, before its next await."""

    def __init__(self, path: Path, item: Mapping[str, Any]):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data.get("protocol") != PROTOCOL or self.data.get("baseline_item") != item:
                raise ProbeUnavailable("archive protocol or baseline differs; user intervention required")
            if self.data.get("detection_protocol") != DETECTION_PROTOCOL:
                if self.data.get("detection_protocol") is not None:
                    raise ProbeUnavailable("unsupported detection protocol; user intervention required")
                backup = path.with_name(path.stem + ".legacy-" + uuid4().hex + path.suffix)
                shutil.copy2(path, backup)
                self.data.setdefault("legacy_detection_archives", []).append(str(backup))
                self.data.update(detection_protocol=DETECTION_PROTOCOL, status="running", stage="planning")
                for key in ("candidate", "option_plan", "option_protocol", "terminal_failure", "pause_reason"):
                    self.data.pop(key, None)
                self.save()
        else:
            self.data = {"protocol": PROTOCOL, "detection_protocol": DETECTION_PROTOCOL,
                         "baseline_item": deepcopy(dict(item)),
                         "working_item": deepcopy(dict(item)), "stage": "planning",
                         "rewrite_count": 0, "rounds": [], "calls": [], "errors": [],
                         "created_at": _now(), "status": "running"}
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = _now()
        # Each save gets its own source path; the lock prevents same-process
        # writers from racing on one archive while the retry handles brief
        # Windows scanner/reader sharing violations on the destination.
        tmp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        with _ARCHIVE_SAVE_LOCK:
            try:
                with tmp.open("w", encoding="utf-8") as stream:
                    json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    _replace_archive_with_retry(tmp, self.path)
                except OSError as exc:
                    winerror = getattr(exc, "winerror", None)
                    retryable = winerror in {32, 33} or exc.errno in {
                        errno.EACCES,
                        errno.EBUSY,
                        errno.EPERM,
                    }
                    if not retryable:
                        raise
                    fallback = self.path.with_name(
                        f"{self.path.stem}.lock-recovery-{uuid4().hex}{self.path.suffix}"
                    )
                    _replace_archive_with_retry(tmp, fallback)
                    self.path = fallback
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def result(self) -> dict[str, Any]:
        return {"protocol": PROTOCOL, "status": self.data["status"],
                "detection_protocol": self.data.get("detection_protocol"),
                "option_protocol": self.data.get("option_protocol"),
                "stage": self.data["stage"], "rewrite_count": self.data["rewrite_count"],
                "max_rewrites": MAX_REWRITES, "design_stage": self.data.get("design_stage", 0),
                "total_rewrite_count": self.data.get("total_rewrite_count", self.data["rewrite_count"]),
                "recovery_version": self.data.get("recovery_version"),
                "staged_design": deepcopy(self.data.get("staged_design")),
                "active_baseline_item": deepcopy(self.data.get("active_baseline_item", self.data["baseline_item"])),
                "archive_ref": str(self.path), "pause_reason": self.data.get("pause_reason"),
                "replenishment_reason": self.data.get("replenishment_reason"),
                "candidate": deepcopy(self.data.get("candidate")),
                "model_call_count": len(self.data["calls"])}


def is_current_ready(result: Mapping[str, Any]) -> bool:
    return (result.get("status") == "ready"
            and result.get("detection_protocol") == DETECTION_PROTOCOL
            and result.get("option_protocol") == EDIT_PROTOCOL)


def design_records(data):
    return [*(data.get("completed_designs") or []), data]


def failure_memory(data, archive_ref):
    memory = []
    for design in design_records(data):
        rounds = design.get("rounds") or []
        for index, row in enumerate(rounds):
            decision = row.get("decision") or {}
            if decision.get("passes") is True:
                continue
            memory.append({"design_stage": design.get("design_stage", 0), "round": row["number"],
                "detection_protocol": row.get("detection_protocol", "legacy_sign_coding"),
                "scenario": row["scenario"], "decision": deepcopy(decision),
                "rewrite": deepcopy(row.get("rewrite")),
                "subsequent_decision": deepcopy(rounds[index + 1].get("decision")) if index + 1 < len(rounds) else None,
                "archive_ref": str(archive_ref)})
    return memory


def archive_path(state: Mapping[str, Any], item: Mapping[str, Any], revision_round: int) -> Path:
    from sjt_system.runtime.output_paths import scoped_output
    root = scoped_output("scenario_detection", "outputs/scenario_detection")
    run_id = _text(state.get("run_id"), "run_id for probe archive")
    return (root / ("run-" + quote(run_id, safe=""))
            / f"analysis-{int(state.get('psychometric_analysis_round') or 0)}"
            / ("item-" + quote(str(item["item_id"]), safe=""))
            / f"v{int(item.get('version') or 0)}-repair-{revision_round}.json")


async def run_scenario_repair(*, item: Mapping[str, Any], evidence: Mapping[str, Any],
                              path: Path, invoke: Invoke,
                              model_info: Mapping[str, Any] | None = None,
                              knowledge_snapshot: Mapping[str, Any] | None = None,
                              source_context: Mapping[str, Any] | None = None,
                              rebuild: Callable | None = None,
                              progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    archive = RepairArchive(path, item)
    data = archive.data
    previous_recovery_version = data.get("recovery_version")
    if previous_recovery_version != RECOVERY_VERSION:
        data["recovery_version"] = RECOVERY_VERSION
        data.setdefault("design_stage", 0)
        data.setdefault("total_rewrite_count", data["rewrite_count"])
        data.setdefault("completed_designs", [])
        data.setdefault("active_baseline_item", deepcopy(item))
        if data.get("pause_reason") == "rewrite_limit":
            data.pop("pause_reason", None)
        if previous_recovery_version == 2 and data.get("terminal_failure") and not is_current_ready(data):
            transient_failure = next((record for record in reversed(data.get("calls") or [])
                                      if record.get("recovery_policy") == POLICY
                                      and record.get("status") == "error"
                                      and record.get("error_category") == "transient_service"), None)
            if transient_failure is not None:
                failed_key = transient_failure.get("key")
                used_calls = sum(1 for record in data.get("calls") or []
                                 if record.get("recovery_policy") == POLICY
                                 and record.get("key") == failed_key)
                if used_calls < MAX_CALLS:
                    data["terminal_failure"] = False
                    data.pop("pause_reason", None)
                    data["errors"].append({
                        "stage": transient_failure.get("kind"),
                        "message": f"explicit resume opened expanded transient retry budget ({used_calls}/{MAX_CALLS})",
                        "recorded_at": _now(),
                    })
        archive.save()
    if knowledge_snapshot is not None and data.get("knowledge_snapshot") is not None and data["knowledge_snapshot"] != knowledge_snapshot:
        message = "cannot change frozen repair knowledge on resume"
        data["errors"].append({"stage": data["stage"], "message": message, "recorded_at": _now()})
        data.update(status="paused", pause_reason=message)
        archive.save()
        return archive.result()
    id_failure = "unknown or duplicate anonymous pair ID"
    if (data.get("terminal_failure") and not is_current_ready(data)
            and data.get("stage") == "compare_adjacent"
            and id_failure in str(data.get("pause_reason") or "")
            and data.get("rounds")
            and any("differences" in str(row.get("key")) for row in data.get("calls", []))):
        current_round = (data.get("rounds") or [])[-1]
        if not current_round.get("pair_id_migration"):
            data["terminal_failure"] = False
            archive.save()
    pause_reason = str(data.get("pause_reason") or "")
    if data.get("terminal_failure") and any(
        marker in pause_reason for marker in ("[WinError 32]", "[WinError 33]")
    ):
        # A sharing violation can occur after the model response was journaled
        # but before the archive was atomically replaced. Let journaled_call
        # validate and reuse that response on resume.
        data["terminal_failure"] = False
        data.pop("pause_reason", None)
        data["status"] = "running"
        archive.save()
    if is_current_ready(data) or data.get("terminal_failure"):
        return archive.result()
    if data.get("option_protocol") != EDIT_PROTOCOL:
        data.pop("candidate", None)
    data["status"] = "running"
    data.pop("pause_reason", None)
    archive.save()

    async def call(key: str, kind: str, payload: dict[str, Any], validate: Callable[[Any], Any]) -> Any:
        from sjt_system.evaluation.repair_recovery import journaled_call
        key = f"{DETECTION_PROTOCOL}/{key}"
        if data["design_stage"]:
            key = f"design-{data['design_stage']}/{key}"
        data["stage"] = kind
        if progress is not None:
            progress({
                "status": "editing",
                "stage": kind,
                "rewrite_count": data["rewrite_count"],
                "model_call_count": len(data["calls"]),
            })
        return await journaled_call(records=data["calls"], key=key, kind=kind, payload=payload,
            invoke=invoke, validate=validate, save=archive.save,
            prompt=PROMPTS[kind], model=(model_info or {}).get(kind, {}))

    try:
        data.setdefault("source_context", deepcopy(dict(source_context or {})))
        if knowledge_snapshot is not None:
            saved_snapshot = data.setdefault("knowledge_snapshot", deepcopy(dict(knowledge_snapshot)))
            if saved_snapshot != knowledge_snapshot:
                raise ProbeUnavailable("cannot change frozen repair knowledge on resume")
        archive.save()
        if not data.get("plan"):
            data["source_evidence"] = deepcopy(dict(evidence))
            archive.save()
            data["plan"] = build_detection_plan(evidence)
            archive.save()
        plan = data["plan"]
        while True:
            number = data["rewrite_count"]
            if len(data["rounds"]) <= number:
                data["rounds"].append({"number": number, "scenario": data["working_item"]["scenario"],
                                       "detection_protocol": DETECTION_PROTOCOL,
                                       "roles": build_roles(plan), "actions": []})
                archive.save()
            detection = data["rounds"][number]
            if detection.get("detection_protocol") != DETECTION_PROTOCOL:
                # Reprobe the current working scenario, keeping counters and prior calls intact.
                detection = {"number": number, "scenario": data["working_item"]["scenario"],
                             "detection_protocol": DETECTION_PROTOCOL,
                             "roles": build_roles(plan), "actions": []}
                data["rounds"][number] = detection
                archive.save()
            actions = detection.setdefault("actions", [])
            if not isinstance(actions, list) or len(actions) > len(detection["roles"]):
                raise ProbeUnavailable("invalid cached open actions")
            actions.extend([None] * (len(detection["roles"]) - len(actions)))

            async def probe_action(index: int, role: Mapping[str, Any]) -> dict[str, Any]:
                payload = {"persona": deepcopy(role["persona"]), "scenario": detection["scenario"],
                           "instruction": "你会怎么做？"}
                action = await call(f"{number}/action/{index}", "open_action", payload, validate_actions)
                record = {"action": action}
                actions[index] = record
                archive.save()
                return record

            pending_actions = [
                probe_action(index, role)
                for index, role in enumerate(detection["roles"])
                if actions[index] is None
            ]
            if pending_actions:
                action_results = await asyncio.gather(*pending_actions, return_exceptions=True)
                archive.save()
                first_error = next(
                    (result for result in action_results if isinstance(result, BaseException)),
                    None,
                )
                if first_error is not None:
                    raise first_error
            if "adjacent_pairs" not in detection:
                used_ids = {pair_id for design in design_records(data)
                            for prior in design.get("rounds", [])
                            for pair_id in (prior.get("adjacent_pairs") or {}).get("identity", {})}
                detection["adjacent_pairs"] = build_adjacent_pairs(
                    detection["roles"], detection["actions"], used_ids=used_ids)
                archive.save()
            pairs = detection["adjacent_pairs"]
            if "differences" not in detection:
                async def invoke_comparer(kind: str, payload: dict[str, Any]) -> Any:
                    partial = detection.setdefault("partial_differences", {})
                    source_by_id = {r["pair_id"]: r for r in pairs["records"]}
                    def validate_partial(raw):
                        _keys(raw, {"records"}, "adjacent differences")
                        if not isinstance(raw["records"], list):
                            raise ProbeUnavailable("difference records must be a list")
                        rows = deepcopy(raw["records"])
                        requested = {row["pair_id"] for row in payload["pairs"]}
                        known = [row.get("pair_id") for row in rows if isinstance(row, Mapping)
                                 and row.get("pair_id") in requested]
                        unknown = [row for row in rows if not isinstance(row, Mapping)
                                   or row.get("pair_id") not in requested]
                        missing = requested.difference(known)
                        if (len(requested) >= 3 and len(rows) == len(requested)
                                and len(unknown) == len(missing) == 1
                                and len(known) == len(set(known)) == len(requested) - 1
                                and isinstance(unknown[0], Mapping)
                                and isinstance(unknown[0].get("pair_id"), str)):
                            original_id = unknown[0]["pair_id"]
                            corrected_id = next(iter(missing))
                            unknown[0]["pair_id"] = corrected_id
                            correction = {"original_id": original_id, "corrected_id": corrected_id,
                                          "method": "single_missing_pair_bijection"}
                            if correction not in detection.setdefault("pair_id_corrections", []):
                                detection["pair_id_corrections"].append(correction)
                                archive.save()
                        seen, errors = set(), []
                        for row in rows:
                            pair_id = row.get("pair_id") if isinstance(row, Mapping) else None
                            if not isinstance(pair_id, str) or pair_id not in requested or pair_id in seen:
                                raise ProbeUnavailable("unknown or duplicate anonymous pair ID")
                            seen.add(pair_id)
                        for row in rows:
                            pair_id = row["pair_id"]
                            try:
                                partial.update(validate_differences({"records": [row]}, [source_by_id[pair_id]]))
                            except (ValueError, KeyError, TypeError) as exc:
                                errors.append(str(exc))
                        missing = [row for row in pairs["records"] if row["pair_id"] not in partial]
                        payload["pairs"] = deepcopy(missing)
                        archive.save()
                        if missing or errors:
                            raise ProbeUnavailable("; ".join(errors) or "missing adjacent differences")
                        return {"records": [dict(pair_id=key, difference=value) for key, value in partial.items()]}
                    payload["pairs"] = [deepcopy(row) for row in pairs["records"] if row["pair_id"] not in partial]
                    if not payload["pairs"]:
                        return {"records": [dict(pair_id=key, difference=value) for key, value in partial.items()]}
                    call_key = f"{number}/differences"
                    if detection.get("pair_id_migration"):
                        call_key += "/short_ids_v1"
                    return await call(call_key, kind, payload, validate_partial)
                try:
                    detection["differences"] = await compare_actions_blindly(
                        plan["target_construct"], pairs["records"], invoke_comparer)
                except RepairStopped as exc:
                    if id_failure not in str(exc) or detection.get("pair_id_migration"):
                        raise
                    used_ids = {pair_id for design in design_records(data)
                                for prior in design.get("rounds", [])
                                for pair_id in (prior.get("adjacent_pairs") or {}).get("identity", {})}
                    detection["adjacent_pairs"] = rekey_legacy_pairs(pairs, used_ids=used_ids)
                    detection["pair_id_migration"] = "short_ids_v1"
                    detection.pop("partial_differences", None)
                    archive.save()
                    continue
                archive.save()
            detection["decision"] = evaluate_differences(detection["roles"], pairs["identity"], detection["differences"])
            data["stage"] = "detected"
            archive.save()
            if detection["decision"]["passes"]:
                break
            if number >= MAX_REWRITES:
                if data["design_stage"] >= 2:
                    raise RepairStopped(DESIGN_LIMIT_REASON)
                if rebuild is None:
                    raise RepairStopped("rewrite_limit: no reconstruction context available")
                data["stage"] = "rebuilding"
                archive.save()
                bundle = await rebuild(archive, data["design_stage"] + 1)
                old = {key: deepcopy(data.get(key)) for key in (
                    "design_stage", "working_item", "active_baseline_item", "source_evidence", "plan", "rounds", "rewrite_count", "staged_design")}
                data["completed_designs"].append(old)
                data.update(design_stage=data["design_stage"] + 1, staged_design=deepcopy(bundle),
                            working_item=deepcopy(bundle["item"]), active_baseline_item=deepcopy(bundle["item"]),
                            rewrite_count=0, rounds=[], stage="planning")
                data["plan"] = deepcopy(plan)
                data["plan"]["fixed_skeleton"] = deepcopy(bundle["skeleton"])
                data["plan"]["constraints"] = deepcopy(bundle["constraints"])
                data["source_evidence"] = {**deepcopy(data["source_evidence"]),
                    "current_item": deepcopy(bundle["item"]), "fixed_skeleton": deepcopy(bundle["skeleton"]),
                    "normal_constraints": deepcopy(bundle["constraints"])}
                plan = data["plan"]
                archive.save()
                continue
            payload = {"scenario": detection["scenario"], "constraints": plan["constraints"],
                       "fixed_skeleton": plan["fixed_skeleton"],
                       "groups": plan["groups"], "roles": detection["roles"],
                       "actions": detection["actions"], "adjacent_differences": detection["decision"]["groups"],
                       "decision": detection["decision"],
                       "failure_memory": failure_memory(data, path)}
            if knowledge_snapshot is not None:
                payload["repair_knowledge"] = deepcopy(dict(knowledge_snapshot))
            detection["rewrite_instruction"] = deepcopy(payload)
            archive.save()
            def validate_scenario(raw: Any) -> Any:
                apply_text_patch(data["working_item"], raw)
                if raw["scenario"].strip() in {r["scenario"].strip() for d in design_records(data) for r in d.get("rounds", [])}:
                    raise ProbeUnavailable("scenario repeats an already tested design")
                validate_knowledge_usage(raw, knowledge_snapshot)
                return raw
            patch = await call(f"{number}/scenario", "rewrite_scenario", payload, validate_scenario)
            detection["rewrite"] = patch
            data["working_item"] = apply_text_patch(data["working_item"], patch)
            data["rewrite_count"] = number + 1
            data["total_rewrite_count"] += 1
            archive.save()
        source = data["source_evidence"]
        gradient = (source.get("option_choice_diagnostics") or {}).get("target_option_gradient") or source.get("target_option_gradient") or {}
        from sjt_system.evaluation.repair_knowledge import non_target_gradients
        findings = non_target_gradients(source)
        option_plan = plan_option_edit(data["working_item"], gradient, scenario_changed=data["total_rewrite_count"] > 0,
                                      non_target_findings=findings)
        data["option_plan"] = option_plan
        payload = {**deepcopy(option_plan), "item": deepcopy(data["working_item"]),
                   "fixed_skeleton": plan["fixed_skeleton"],
                   "constraints": plan["constraints"], "detection": deepcopy(detection),
                   "failure_memory": failure_memory(data, path),
                   "scenario_rewrite_reasons": [r["rewrite"]["reason"] for r in data["rounds"] if r.get("rewrite")],
                   "formal_target_option_gradient": deepcopy(gradient)}
        payload["target_adjacent_differences"] = target_option_differences(data["working_item"], detection["decision"])
        payload["non_target_adjacent_differences"] = {group: deepcopy(result["differences"])
            for group, result in detection["decision"]["groups"].items() if group != "target"}
        payload["non_target_findings"] = findings
        if knowledge_snapshot is not None:
            payload["repair_knowledge"] = deepcopy(dict(knowledge_snapshot))
        def validate_options(raw: Any) -> Any:
            candidate = apply_text_patch(data["working_item"], raw, option_ids=option_plan["authorized_option_ids"])
            validate_candidate(data["active_baseline_item"], candidate)
            validate_knowledge_usage(raw, knowledge_snapshot)
            return raw
        patch = await call("options/" + EDIT_PROTOCOL, "rewrite_options", payload, validate_options)
        data["candidate"] = apply_text_patch(data["working_item"], patch, option_ids=option_plan["authorized_option_ids"])
        data.update(status="ready", stage="staged", option_protocol=EDIT_PROTOCOL)
        archive.save()
    except (Exception, asyncio.CancelledError) as exc:
        data["errors"].append({"stage": data["stage"], "message": str(exc),
                               "type": type(exc).__name__, "recorded_at": _now()})
        data.update(status="paused", pause_reason=str(exc) or type(exc).__name__)
        data["terminal_failure"] = not isinstance(exc, asyncio.CancelledError)
        archive.save()
        if isinstance(exc, asyncio.CancelledError):
            raise
    return archive.result()
