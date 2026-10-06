"""Versioned, repair-only facet experience. Never a source of formal responses."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote
from uuid import uuid4

from typing_extensions import TypedDict

from sjt_system.agent.client import LocalJSONSchemaError
from sjt_system.evaluation.scenario_detection import (
    DETECTION_PROTOCOL, FIELDS, ProbeUnavailable, _number, _text, _now,
    evaluate_differences, validate_differences, validate_actions, design_records,
)
from sjt_system.runtime.io import write_json_atomic
from sjt_system.runtime.output_paths import scoped_output

PROTOCOL = "facet_mechanism_v1"
KINDS = {f"{stage}_{role}" for stage in ("option", "scenario")
         for role in ("target_confusion", "same_domain_differentiation", "cross_domain_differentiation")}
KnowledgeKind = Literal[
    "option_target_confusion", "option_same_domain_differentiation",
    "option_cross_domain_differentiation", "scenario_target_confusion",
    "scenario_same_domain_differentiation", "scenario_cross_domain_differentiation",
]
KnowledgeStatus = Literal["active", "qualified"]
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "repair_knowledge"


class KnowledgeQuote(TypedDict):
    source_id: str
    fragment_id: str


class KnowledgeRule(TypedDict):
    entry_id: str
    kind: KnowledgeKind
    mechanism: str
    conditions: str
    guidance: str
    status: KnowledgeStatus
    source_ids: list[str]
    quotes: list[KnowledgeQuote]


class KnowledgeUpdate(TypedDict):
    updates: list[KnowledgeRule]
    limitations: str


PROMPT = """从本facet的模拟审题/情境检测证据归纳可修订的返修经验，不是证明因果规律。
输入sources区分实际涉及facet、当时的目标facet、目标/同域/跨域角色；相邻差异判断始终针对目标构念。
目标混淆说明具体行为策略或决策节点为何可能不能区分目标水平；非目标分化说明哪些条件可能
让该非目标facet左右目标题的行为选择，不把这个facet本身当成缺陷。
只用usable=true的证据。每项必须给出具体mechanism、conditions适用边界、guidance修改建议，
source_ids以及对应原文quotes；不得仅把均值或差异摘要改写成泛泛建议。最终通过记录可补充反例、
收窄旧知识适用范围，不能据此声称正式统计指标已经改善。
updates中entry_id为空表示新增，否则必须引用existing中的ID进行修订。
kind只能从本次输入allowed_kinds选择，不能概括为non_target_differentiation等自造名称。
六类合法值：option_target_confusion、option_same_domain_differentiation、
option_cross_domain_differentiation、scenario_target_confusion、
scenario_same_domain_differentiation、scenario_cross_domain_differentiation。
status为active或qualified（已有反例/适用范围受限）。不删除条目，不遗漏旧来源，不擅改构念。
quotes每项只返回source_id和fragment_id，从evidence_fragments选择；原文由程序回填，不抄写或转述。
新增只允许creation_kinds；其他类别只有existing中同类别条目可以修订。
若输入有validation_feedback，先修正指出的非法字段，再重新核对所有来源和原文引文。
同一知识尽量修订已有条目、合并证据，不重复新增。证据不足时updates为空并在limitations说明。
只输出updates和limitations；归纳结论均属于模拟证据支持的经验假设。"""


def packed(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def knowledge_payload(facet, sources, existing):
    usable = [source for source in sources if source["usable"]]
    kinds = {source["kind"] for source in usable}
    existing = [rule for rule in existing if rule["kind"] in kinds]
    return {"facet_id": facet, "sources": sources, "existing": existing,
            "allowed_kinds": sorted(kinds),
            "creation_kinds": sorted({source["kind"] for source in usable if source["finding"]}),
            "evidence_fragments": [{"source_id": source["source_id"], "fragment_id": f"text-{index}", "text": text}
                for source in usable for index, text in enumerate(source["texts"])],
            "evidence_status": "simulation_supported_hypothesis"}


def resolve_knowledge_output(raw, payload):
    """Resolve program-owned quotations and reject unsupported creation locally."""
    if not isinstance(raw, Mapping) or not isinstance(raw.get("updates"), list):
        raise ProbeUnavailable("knowledge output requires an updates list")
    raw = deepcopy(dict(raw))
    sources = {s["source_id"]: s for s in payload["sources"]}
    fragments = {(f["source_id"], f["fragment_id"]): f["text"] for f in payload["evidence_fragments"]}
    existing = {
        str(rule.get("entry_id")): rule
        for rule in payload.get("existing") or []
        if isinstance(rule, Mapping) and rule.get("entry_id")
    }
    accepted, rejected = [], []
    for rule in raw["updates"]:
        if not isinstance(rule, Mapping):
            raise ProbeUnavailable("invalid knowledge rule")
        ids = rule.get("source_ids")
        entry_id = str(rule.get("entry_id") or "")
        previous = existing.get(entry_id) or {}
        historical_ids = {
            str(value) for value in previous.get("source_ids") or []
            if isinstance(value, str)
        }
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ProbeUnavailable("knowledge must reference known sources")
        unknown_ids = [i for i in ids if i not in sources and i not in historical_ids]
        if unknown_ids:
            if not entry_id:
                rejected.append({
                    "reason": "unknown_source",
                    "source_ids": unknown_ids,
                    "rule": deepcopy(rule),
                })
                continue
            # A post-repair probe can omit an archive that is already part of
            # an existing entry.  Keep the verifiable current/historical IDs
            # instead of failing the whole knowledge stage on that omission.
            ids = [i for i in ids if i not in unknown_ids]
            if not ids:
                rejected.append({
                    "reason": "unknown_source",
                    "source_ids": unknown_ids,
                    "rule": deepcopy(rule),
                })
                continue
        if not rule.get("entry_id") and ids and not any(sources[i]["usable"] and sources[i]["finding"] for i in ids):
            rejected.append({"reason": "passing_only_creation", "rule": deepcopy(rule)})
            continue
        filtered_quotes = []
        for citation in rule.get("quotes") or []:
            if not isinstance(citation, dict):
                raise ProbeUnavailable("invalid knowledge quotation")
            if "source_id" not in citation:
                raise ProbeUnavailable("invalid knowledge quotation")
            if citation["source_id"] not in sources and citation["source_id"] not in historical_ids:
                continue
            if "fragment_id" in citation:
                if set(citation) != {"source_id", "fragment_id"}:
                    raise ProbeUnavailable("quotation must select source_id and fragment_id")
                pair = (citation["source_id"], citation["fragment_id"])
                if pair not in fragments:
                    if citation["source_id"] in historical_ids:
                        continue
                    raise ProbeUnavailable("unknown knowledge fragment")
                citation.clear()
                citation.update(source_id=pair[0], text=fragments[pair])
            elif citation.get("source_id") in historical_ids and citation.get("source_id") not in sources:
                continue
            filtered_quotes.append(citation)
        if not filtered_quotes:
            rejected.append({
                "reason": "unknown_source",
                "source_ids": unknown_ids,
                "rule": deepcopy(rule),
            })
            continue
        rule = {**dict(rule), "source_ids": ids, "quotes": filtered_quotes}
        accepted.append(rule)
    raw["updates"] = accepted
    if rejected and not accepted and not raw.get("limitations"):
        raw["limitations"] = "Passing-only observations cannot establish a new failure mechanism."
    return raw, rejected


def config_for_mode(state: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"shared", "run_only"}:
        raise ProbeUnavailable("请先选择返修知识模式：shared 或 run_only")
    prior = state.get("repair_knowledge_config") or {}
    root = Path(prior.get("root") or scoped_output("repair_knowledge", DEFAULT_ROOT)).resolve()
    run_id = _text(state.get("run_id"), "knowledge run_id")
    database = root / "shared.sqlite3" if mode == "shared" else root / "runs" / quote(run_id, safe="") / "knowledge.sqlite3"
    return {"mode": mode, "root": str(root), "database_ref": str(database),
            "namespace": prior.get("namespace") or state.get("experiment_condition") or "development"}


def construct_refs(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    item = evidence.get("current_item") or {}
    target = str(item.get("target_dimension_id") or "")
    refs = {}
    for row in evidence.get("normal_constraints") or []:
        key, statement = str(row.get("constraint_id") or ""), str(row.get("statement") or "").strip()
        facet = target if key.startswith("FACET_DEFINITION:") else (
            key.split(":", 2)[1] if key.startswith("NON_TARGET_") and key.endswith(":DEFINITION") else "")
        if facet and statement:
            refs[facet] = {"facet_id": facet, "definition": statement,
                           "construct_version": str(evidence.get("construct_version") or "definition-v1")}
    return refs


def non_target_gradients(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use formal winners and finite, score-sorted means; never absolute rho."""
    observations = {row.get("observation_id"): row for row in evidence.get("observations") or []}
    item = evidence.get("current_item") or {}
    key = item.get("scoring_key") or {}
    if len(key) != 4 or any(_number(value) is None for value in key.values()) or len(set(key.values())) != 4:
        return []
    ids = sorted(key, key=key.get)
    rows = (evidence.get("option_choice_diagnostics") or {}).get("option_score_comparisons") or []
    by_id = {row.get("option_id"): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        return []
    result = []
    for arm in ("same_domain", "cross_domain"):
        gate = observations.get(f"OBS:{arm.upper()}_VTS", {})
        value, threshold = _number(gate.get("value")), _number(gate.get("threshold"))
        if value is None or threshold is None or value >= threshold:
            continue
        winner = observations.get(f"OBS:{arm.upper()}_MAX_NON_TARGET_RHO", {})
        facet = winner.get("dimension_id")
        if not facet or _number(winner.get("signed_rho")) is None:
            continue
        means = [_number(by_id[option].get(f"{arm}_mean_score")) for option in ids]
        if any(mean is None for mean in means):
            continue
        if any(by_id[option].get("score") != key[option] for option in ids):
            raise ProbeUnavailable("非目标选项均值与计分键不一致")
        if all(low < high for low, high in zip(means, means[1:])):
            result.append({"arm": arm, "facet_id": facet, "winner": deepcopy(winner),
                           "option_ids": ids, "means": means})
    return result


def _source(state, evidence, *, stage, role, facet, detail, texts, finding=True,
            usable=True, limitation="", archive_ref=None, probe_round=None, detection_protocol=None):
    item = evidence.get("current_item") or {}
    ref = construct_refs(evidence).get(facet)
    kind = f"{stage}_{'target_confusion' if role == 'target' else role + '_differentiation'}"
    identity = (["probe", archive_ref, probe_round, kind] if archive_ref else
                ["review", state.get("run_id"), state.get("psychometric_analysis_round", 0),
                 item.get("item_id"), item.get("version"), kind])
    if detection_protocol:
        identity.append(detection_protocol)
    return {"source_id": packed(identity), "kind": kind, "facet_id": facet,
            "construct": ref, "target_facet_id": item.get("target_dimension_id"), "role": role,
            "item_id": item.get("item_id"), "item_version": item.get("version"),
            "run_id": state.get("run_id"), "analysis_round": state.get("psychometric_analysis_round", 0),
            "archive_ref": archive_ref, "probe_round": probe_round, "finding": finding,
            "usable": bool(usable and ref), "limitation": limitation if ref else "missing construct definition",
            "texts": [text for text in texts if isinstance(text, str) and text.strip()],
            "evidence": deepcopy(detail)}


def review_sources(state, entries):
    sources = []
    for entry in entries:
        evidence = entry.get("diagnosis_evidence") or {}
        item = evidence.get("current_item") or {}
        texts = [item.get("scenario"), *(row.get("text") for row in item.get("response_options") or [])]
        gradient = (evidence.get("option_choice_diagnostics") or {}).get("target_option_gradient") or evidence.get("target_option_gradient") or {}
        if gradient.get("passes") is False:
            ordered_ids = set(item.get("scoring_key") or {})
            means = {row.get("option_id"): _number(row.get("target_facet_mean")) for row in gradient.get("options") or []}
            usable = len(ordered_ids) == 4 and all(means.get(key) is not None for key in ordered_ids)
            sources.append(_source(state, evidence, stage="option", role="target",
                facet=item.get("target_dimension_id"), detail={"gradient": gradient, "item": item}, texts=texts,
                usable=usable, limitation="" if usable else "missing means: not evidence of a confusing strategy"))
        for finding in non_target_gradients(evidence):
            sources.append(_source(state, evidence, stage="option", role=finding["arm"],
                facet=finding["facet_id"], detail={"gradient": finding, "item": item}, texts=texts))
    return sources


def probe_sources(state, paths):
    sources = []
    for name in sorted(set(str(Path(path).resolve()) for path in paths if path)):
        path = Path(name)
        if not path.exists():
            continue  # Queued but not started probes have no archive yet.
        data = json.loads(path.read_text(encoding="utf-8"))
        origin = data.get("source_context") or {
            "run_id": next((unquote(part[4:]) for part in path.parts if part.startswith("run-")), None),
            "psychometric_analysis_round": next((int(part[9:]) for part in path.parts
                if part.startswith("analysis-") and part[9:].isdigit()), None),
        }
        for design in design_records(data):
            evidence = deepcopy(design.get("source_evidence") or {})
            evidence["current_item"] = design.get("active_baseline_item") or data.get("baseline_item") or evidence.get("current_item") or {}
            groups = {row["group"]: row for row in (design.get("plan") or {}).get("groups") or []}
            for detection in design.get("rounds") or []:
                if (detection.get("detection_protocol") != DETECTION_PROTOCOL
                        or not detection.get("differences") or not detection.get("adjacent_pairs") or not groups):
                    continue
                try:
                    pairs = detection["adjacent_pairs"]
                    differences = validate_differences({"records": [dict(pair_id=key, difference=value)
                        for key, value in detection["differences"].items()]}, pairs["records"])
                    decision = evaluate_differences(detection["roles"], pairs["identity"], differences)
                    if len(detection["actions"]) != len(detection["roles"]):
                        continue
                    for row in detection["actions"]:
                        validate_actions(row["action"])
                except (ValueError, KeyError, TypeError):
                    continue  # Invalid/partial or legacy evidence cannot satisfy the new protocol.
                for role, result in decision["groups"].items():
                    group = groups[role]
                    texts = [detection["scenario"]]
                    actions = []
                    for index, person in enumerate(detection["roles"]):
                        if person["group"] == role:
                            action = detection["actions"][index]["action"]
                            texts.extend(action[field] for field in FIELDS)
                            actions.append({"level": person["level"], "action": action})
                    sources.append(_source(origin, evidence, stage="scenario", role=role,
                        facet=group["facet_id"], detail={"scenario": detection["scenario"], "actions": actions,
                        "adjacent_differences": result, "detection_protocol": DETECTION_PROTOCOL,
                        "compared_target_facet_id": evidence["current_item"].get("target_dimension_id"),
                        "round_passes": decision["passes"]}, texts=texts,
                        finding=not result["passes"], archive_ref=name,
                        detection_protocol=DETECTION_PROTOCOL,
                        probe_round=(detection["number"] if not design.get("design_stage") else
                                     f"design-{design['design_stage']}/round-{detection['number']}")))
    return sources


def archive_refs(state, entries=(), progress=None):
    refs = [entry.get("scenario_archive_ref") for entry in entries]
    for row in state.get("psychometric_repair_history") or []:
        refs.append((row.get("scenario_detection") or {}).get("archive_ref"))
    for rows in (state.get("scenario_repair_progress") or {}, progress or {}):
        refs.extend(row.get("archive_ref") for row in rows.values())
    return [ref for ref in refs if ref]


def needs_review_knowledge(state):
    entries = [*(state.get("items_to_revise") or []), *(state.get("items_to_regenerate") or [])]
    if not entries or not all(isinstance(row, Mapping) and (
            row.get("repair_protocol") == "scenario_detection_first_v1"
            or row.get("diagnosis_status") == "repair_rounds_exhausted") for row in entries):
        return False
    progress = state.get("repair_knowledge_state") or {}
    return not (progress.get("review_complete") and progress.get("analysis_round") == int(state.get("psychometric_analysis_round") or 0))


def make_knowledge_invoker(state):
    from sjt_system.agent.agent_factory import create_agent, PSYCHOMETRIC_REASONING_ROLE_MANIFEST
    from sjt_system.agent.client import get_model_request_timeout_seconds
    role = PSYCHOMETRIC_REASONING_ROLE_MANIFEST["psychometric_item_repair"]
    metadata = {"model_id": role["model_id"], "temperature": role.get("temperature"),
                "thinking_type": role.get("thinking"), "reasoning_effort": role.get("reasoning_effort")}
    async def invoke(payload):
        agent = create_agent(PROMPT, KnowledgeUpdate, **metadata)
        return await asyncio.wait_for(agent.ainvoke({"input_data": packed(payload)}),
                                      timeout=get_model_request_timeout_seconds())
    return invoke, metadata


class KnowledgeStore:
    def __init__(self, config):
        self.config = config
        self.path = Path(config["database_ref"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, scope TEXT, payload TEXT, processed INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, scope TEXT, revision INTEGER);
                CREATE TABLE IF NOT EXISTS revisions (id TEXT, revision INTEGER, payload TEXT, PRIMARY KEY(id, revision));
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, scope TEXT, status TEXT, input TEXT, raw TEXT, archive TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, payload TEXT);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def source_key(self, source_id):
        return packed([self.config["namespace"], source_id])

    def scope(self, ref):
        return packed([self.config["namespace"], ref["facet_id"], ref["construct_version"], ref["definition"]])

    def current(self, scope):
        with self.connect() as db:
            rows = db.execute("SELECT r.payload FROM entries e JOIN revisions r ON e.id=r.id AND e.revision=r.revision WHERE e.scope=? ORDER BY e.id", (scope,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def register(self, sources):
        with self.connect() as db:
            for source in sources:
                scope = self.scope(source["construct"]) if source.get("construct") else "unavailable"
                identity = self.source_key(source["source_id"])
                previous = db.execute("SELECT payload FROM sources WHERE id=?", (identity,)).fetchone()
                if previous and previous[0] != packed(source):
                    raise ProbeUnavailable("知识来源标识已存在但证据不同，需人工检查")
                db.execute("INSERT OR IGNORE INTO sources(id,scope,payload) VALUES(?,?,?)", (identity, scope, packed(source)))

    def pending(self, source_ids):
        result = {}
        with self.connect() as db:
            for identity in sorted(set(source_ids)):
                row = db.execute("SELECT scope,payload FROM sources WHERE id=? AND processed=0", (self.source_key(identity),)).fetchone()
                if row:
                    result.setdefault(row[0], []).append(json.loads(row[1]))
        return result

    @staticmethod
    def _unsupported_quotes(raw, sources):
        unsupported = []
        if not isinstance(raw, Mapping) or not isinstance(raw.get("updates"), list):
            return unsupported
        for rule in raw["updates"]:
            if not isinstance(rule, Mapping) or not isinstance(rule.get("quotes"), list):
                continue
            for citation in rule["quotes"]:
                if not isinstance(citation, Mapping):
                    continue
                source = sources.get(citation.get("source_id"))
                quote = citation.get("text")
                if (isinstance(source, Mapping) and isinstance(quote, str)
                        and not any(quote in text for text in source["texts"])):
                    unsupported.append({"source_id": citation["source_id"], "text": quote})
        return unsupported

    def validate(self, raw, payload):
        if not isinstance(raw, Mapping) or set(raw) != {"updates", "limitations"} or not isinstance(raw["updates"], list) or not isinstance(raw["limitations"], str):
            raise ProbeUnavailable("invalid knowledge update structure")
        existing = {entry["entry_id"]: entry for entry in payload["existing"]}
        sources = {source["source_id"]: source for source in payload["sources"]}
        seen = set()
        normalized = []
        for rule in raw["updates"]:
            if not isinstance(rule, Mapping) or set(rule) != set(KnowledgeRule.__annotations__):
                raise ProbeUnavailable("invalid knowledge rule fields")
            key = rule["entry_id"]
            if not isinstance(key, str) or (key and (key not in existing or key in seen)):
                raise ProbeUnavailable("unknown or duplicate knowledge entry")
            if key:
                seen.add(key)
            allowed_kinds = payload.get("allowed_kinds") or sorted({source["kind"] for source in sources.values()})
            if rule["kind"] not in KINDS or rule["kind"] not in allowed_kinds:
                raise ProbeUnavailable(
                    f"invalid knowledge kind {rule['kind']!r}; allowed: {allowed_kinds}"
                )
            if rule["status"] not in {"active", "qualified"}:
                raise ProbeUnavailable(f"invalid knowledge status {rule['status']!r}")
            ids = rule["source_ids"]
            previous = existing.get(key, {})
            historical_ids = {
                str(value) for value in previous.get("source_ids") or []
                if isinstance(value, str)
            }
            if not isinstance(ids, list) or not ids or any(
                not isinstance(i, str)
                or (i not in sources and i not in historical_ids)
                or (i in sources and not sources[i]["usable"])
                for i in ids
            ):
                raise ProbeUnavailable("knowledge must reference usable evidence")
            current_ids = [i for i in ids if i in sources]
            if any(sources[i]["kind"] != rule["kind"] for i in current_ids):
                raise ProbeUnavailable("knowledge kind differs from evidence")
            if key and existing[key]["kind"] != rule["kind"]:
                raise ProbeUnavailable("knowledge revision cannot change kind")
            if not key and not any(sources[i]["finding"] for i in current_ids):
                raise ProbeUnavailable("passing observations alone cannot create a failure mechanism")
            quotes = rule["quotes"]
            if not isinstance(quotes, list) or not quotes:
                raise ProbeUnavailable("knowledge needs original text evidence")
            grounded_quotes = []
            for citation in quotes:
                if not isinstance(citation, Mapping) or set(citation) != {"source_id", "text"} or citation["source_id"] not in ids:
                    raise ProbeUnavailable("invalid knowledge quotation")
                if citation["source_id"] not in sources:
                    continue
                text = _text(citation["text"], "knowledge quotation")
                if any(text in original for original in sources[citation["source_id"]]["texts"]):
                    grounded_quotes.append({"source_id": citation["source_id"], "text": text})
            if not grounded_quotes and not previous.get("quotes"):
                raise ProbeUnavailable("knowledge quotation not found in evidence")
            normalized.append({**deepcopy(dict(rule)),
                **{name: _text(rule[name], name) for name in ("mechanism", "conditions", "guidance")},
                "entry_id": key or uuid4().hex, "facet_id": payload["facet_id"],
                "revision": int(previous.get("revision", 0)) + 1,
                "evidence_status": "simulation_supported_hypothesis",
                "source_ids": sorted(set([*previous.get("source_ids", []), *ids])),
                "quotes": [*deepcopy(previous.get("quotes", [])), *grounded_quotes],
                "contexts": [*deepcopy(previous.get("contexts", [])), *[
                    {name: sources[i][name] for name in ("source_id", "target_facet_id", "role", "item_id", "item_version", "archive_ref")}
                    for i in current_ids if i not in previous.get("source_ids", [])]], "updated_at": _now()})
        if not normalized and not raw["limitations"].strip():
            raise ProbeUnavailable("no knowledge update requires an evidence limitation")
        return normalized

    async def learn(self, sources, *, stage_id, invoke, model_info, log_root):
        from sjt_system.evaluation.repair_recovery import journaled_call, POLICY
        from sjt_system.runtime.concurrency import gather_all
        self.register(sources)

        async def learn_scope(scope, pending):
            job_id = packed([stage_id, scope, sorted(source["source_id"] for source in pending)])
            with self.connect() as db:
                job = db.execute("SELECT status,input,raw,archive FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job and job[0] == "completed":
                return
            existing = self.current(scope)
            payload = knowledge_payload(pending[0]["facet_id"], pending, existing)
            old = (json.loads(Path(job[3]).read_text(encoding="utf-8"))
                   if job and job[3] and Path(job[3]).exists() else {})
            if old.get("recovery_policy") == POLICY:
                path, record = Path(job[3]), old
                if record["input"] != payload:
                    raise ProbeUnavailable("knowledge changed during recovery; archived input retained")
            else:
                path = Path(log_root) / (uuid4().hex + ".json")
                record = {"job_id": job_id, "input": payload, "prompt": PROMPT,
                          "model": model_info, "started_at": _now(), "status": "running",
                          "recovery_policy": POLICY, "calls": []}
                if old:
                    record["prior_error_archive"] = job[3]
                    if "raw_output" in old:
                        record["calls"].append({"key": "learn", "kind": "knowledge",
                            "status": "error", "raw_output": deepcopy(old["raw_output"]),
                            "error": old.get("error", "legacy output")})
            def save():
                write_json_atomic(path, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            save()
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?,?,?)",
                    (job_id, scope, "running", packed(payload), None, str(path), None))
            def validate(raw):
                resolved, rejected = resolve_knowledge_output(raw, payload)
                updates = self.validate(resolved, payload)
                return {"updates": updates, "rejected": rejected,
                        "limitations": resolved["limitations"],
                        "excluded_ungrounded_quotes": self._unsupported_quotes(
                            resolved, {s["source_id"]: s for s in pending})}
            async def request(kind, data):
                return await invoke(data)
            try:
                if not payload["creation_kinds"] and not payload["existing"]:
                    outcome = {"updates": [], "rejected": [],
                        "limitations": "No usable failure or same-kind existing mechanism to refine."}
                else:
                    outcome = await journaled_call(records=record["calls"], key="learn",
                        kind="knowledge", payload=payload, invoke=request, validate=validate,
                        save=save, prompt=PROMPT, model=model_info)
                updates = outcome["updates"]
                if record.get("prior_error_archive") and record["calls"] and all(
                        "recovery_policy" not in call for call in record["calls"]):
                    record["reused_raw_archive_ref"] = record["prior_error_archive"]
                record.update(status="received", **deepcopy(outcome))
                if record["calls"]:
                    record["raw_output"] = deepcopy(record["calls"][-1].get("raw_output"))
                save()
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    actual = dict(db.execute("SELECT id,revision FROM entries WHERE scope=?", (scope,)).fetchall())
                    if actual != {row["entry_id"]: row["revision"] for row in existing}:
                        raise ProbeUnavailable("shared knowledge changed during publication; original job retained")
                    for rule in updates:
                        db.execute("INSERT INTO revisions VALUES(?,?,?)",
                                   (rule["entry_id"], rule["revision"], packed(rule)))
                        db.execute("INSERT OR REPLACE INTO entries VALUES(?,?,?)",
                                   (rule["entry_id"], scope, rule["revision"]))
                    db.executemany("UPDATE sources SET processed=1 WHERE id=?",
                                   [(self.source_key(s["source_id"]),) for s in pending])
                    db.execute("UPDATE jobs SET status='completed',raw=?,error=NULL WHERE id=?",
                               (packed(record.get("raw_output")), job_id))
                record.update(status="completed", published_entries=[r["entry_id"] for r in updates], completed_at=_now())
                save()
            except (Exception, asyncio.CancelledError) as exc:
                record.update(status="error", error=str(exc), error_type=type(exc).__name__)
                if record["calls"]:
                    record["raw_output"] = deepcopy(record["calls"][-1].get("raw_output"))
                save()
                with self.connect() as db:
                    db.execute("UPDATE jobs SET status='error',raw=?,error=? WHERE id=? AND status!='completed'",
                               (packed(record.get("raw_output")), str(exc), job_id))
                raise

        await gather_all(*(learn_scope(scope, pending) for scope, pending in
                          self.pending([source["source_id"] for source in sources]).items()))

    def snapshot(self, snapshot_id, scopes):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT payload FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if previous:
                return json.loads(previous[0])
            entries = []
            for scope in sorted(set(scopes)):
                rows = db.execute("SELECT r.payload FROM entries e JOIN revisions r ON e.id=r.id AND e.revision=r.revision WHERE e.scope=? ORDER BY e.id", (scope,)).fetchall()
                entries.extend({**json.loads(row[0]), "scope": scope} for row in rows)
            result = {"snapshot_id": snapshot_id, "protocol": PROTOCOL, "entries": entries, "created_at": _now()}
            db.execute("INSERT INTO snapshots VALUES(?,?)", (snapshot_id, packed(result)))
        return result


class KnowledgeSession:
    """A durable, frozen knowledge boundary for one formal analysis batch."""
    def __init__(self, state, entries):
        self.state, self.entries = state, deepcopy(entries)
        profile = (state.get("blueprint") or {}).get("construct_profile_snapshot") or {}
        version = str(profile.get("inventory_version") or profile.get("version") or profile.get("schema_version") or "definition-v1")
        for entry in self.entries:
            if isinstance(entry.get("diagnosis_evidence"), dict):
                entry["diagnosis_evidence"].setdefault("construct_version", version)
        config = state.get("repair_knowledge_config") or {}
        self.config = config_for_mode(state, config.get("mode"))
        self.store = KnowledgeStore(self.config)
        self.batch_id = packed([PROTOCOL, state["run_id"], state.get("psychometric_analysis_round", 0)])
        self.path = Path(self.config["root"]) / "runs" / quote(state["run_id"], safe="") / f"analysis-{int(state.get('psychometric_analysis_round') or 0)}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "protocol": PROTOCOL, "config": self.config, "batch_id": self.batch_id, "status": "pending", "stage": "review", "errors": []}
        if self.data["config"] != self.config:
            raise ProbeUnavailable("不能在恢复时更换本批返修知识库")
        self.save()

    def save(self):
        write_json_atomic(self.path, self.data)

    def summary(self):
        return {"protocol": PROTOCOL, "stage": self.data["stage"], "status": self.data["status"],
                "review_complete": bool(self.data.get("snapshot")),
                "analysis_round": int(self.state.get("psychometric_analysis_round") or 0),
                "journal_ref": str(self.path), "database_ref": str(self.store.path),
                "snapshot_id": (self.data.get("snapshot") or {}).get("snapshot_id"),
                "error": self.data.get("error")}

    def pause(self, stage, exc):
        if self.data.get("status") != "paused":
            self.data.update(stage=stage, status="paused", error=str(exc))
            self.data["errors"].append({"stage": stage, "error": str(exc), "at": _now()})
            self.save()

    async def _learn(self, sources, stage):
        self.data.update(stage=stage, status="running", error=None)
        self.save()
        try:
            invoke, metadata = make_knowledge_invoker(self.state)
            await self.store.learn(sources, stage_id=packed([self.batch_id, stage]), invoke=invoke,
                                   model_info=metadata, log_root=self.path.parent / "calls")
        except (Exception, asyncio.CancelledError) as exc:
            self.pause(stage, exc)
            raise

    async def prepare(self):
        if self.data.get("snapshot"):
            return
        sources = review_sources(self.state, self.entries)
        self.data["review_archive_refs"] = archive_refs(self.state)
        sources.extend(probe_sources(self.state, self.data["review_archive_refs"]))
        await self._learn(sources, "review")
        scopes = [self.store.scope(ref) for entry in self.entries
                  for ref in construct_refs(entry.get("diagnosis_evidence") or {}).values()]
        self.data["snapshot"] = self.store.snapshot(self.batch_id, scopes)
        self.data.update(status="ready", stage="repair")
        self.save()

    def for_item(self, evidence):
        evidence = next((entry["diagnosis_evidence"] for entry in self.entries
                         if (entry.get("diagnosis_evidence") or {}).get("current_item", {}).get("item_id")
                         == (evidence.get("current_item") or {}).get("item_id")), evidence)
        scopes = {self.store.scope(ref) for ref in construct_refs(evidence).values()}
        snapshot = self.data["snapshot"]
        return {"snapshot_id": snapshot["snapshot_id"], "protocol": PROTOCOL,
                "entries": [deepcopy(row) for row in snapshot["entries"] if row["scope"] in scopes]}

    async def finish(self, progress):
        self.data["post_repair_archive_refs"] = archive_refs(self.state, self.entries, progress)
        sources = probe_sources(self.state, self.data["post_repair_archive_refs"])
        await self._learn(sources, "post_repair")
        self.data.update(status="completed", stage="post_repair")
        self.save()
