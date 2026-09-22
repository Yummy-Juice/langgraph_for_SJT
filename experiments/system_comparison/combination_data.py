"""Read-only, version-bound datasets for offline blueprint combination searches."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def content_hash(item):
    # Ignore review annotations, but never merge different wording, keys or versions.
    fields = ("item_id", "version", "blueprint_cell_id", "target_dimension_id",
              "scenario", "response_instruction", "response_options", "scoring_key")
    payload = {key: item.get(key) for key in fields}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def inside(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        # Rebase saved absolute paths when an entire experiment has been relocated.
        parts = str(value).replace("\\", "/").split("/")
        if root.name in parts:
            path = root.joinpath(*parts[parts.index(root.name) + 1:]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"数据引用越出实验目录：{value}")
    return path


@dataclass
class SearchData:
    root: Path
    bank: list
    blueprint: dict
    subjects: list
    matrices: dict
    active: dict
    retest: np.ndarray
    reference: pd.DataFrame
    audit: dict


def source_bundles(root, sample_role):
    if sample_role == "development":
        for folder in sorted((root / "C").glob("round_*/development")):
            yield (folder / "item_bank.json", folder / "responses/manifest.json",
                   folder / "responses/psychometrics/scored_matched_condition_sjt_responses.csv",
                   folder / "responses/psychometrics/scored_target_form_retest_sjt_responses.csv")
    else:
        for method in ("B", "C"):
            for form in sorted((root / method).glob("*/form.json")):
                checkpoint = form.parent / "evaluation/checkpoint.json"
                if not checkpoint.exists():
                    continue
                state = read_json(checkpoint)
                manifest = state.get("virtual_response_data_ref")
                repeat = (state.get("test_statistics") or {}).get("output_files", {}).get(
                    "scored_target_form_retest_sjt_responses")
                if manifest:
                    yield (form, inside(root, manifest), form.parent / "evaluation/scored_responses.csv",
                           inside(root, repeat) if repeat else None)


def load_search_data(root, sample_role="evaluation", bank_path="shared/initial_bank.json"):
    root = Path(root).resolve()
    if sample_role not in ("development", "evaluation"):
        raise ValueError("样本角色必须为development或evaluation")
    bank_file = inside(root, bank_path)
    bank = read_json(bank_file)
    if isinstance(bank, dict):
        bank = bank["items"]
    if not isinstance(bank, list) or not bank:
        raise ValueError("候选题库为空")
    ids = [item["item_id"] for item in bank]
    if len(ids) != len(set(ids)):
        raise ValueError("候选题库含重复题号；每次搜索仅允许每题一个冻结版本")
    for item in bank:
        options = item.get("response_options", [])
        keys = item.get("scoring_key", {})
        if (len(options) != 4 or {o["option_id"] for o in options} != set(keys)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) for v in keys.values())):
            raise ValueError(f"选项/计分键无效：{item['item_id']}")
    blueprint_file = root / "shared/blueprint.json"
    blueprint = read_json(blueprint_file)
    sample_file = root / f"participants/{sample_role}.json"
    sample = read_json(sample_file)
    expected = {}
    for row in sample["respondents"]:
        key = (row["condition_id"], row["matched_subject_id"])
        if key in expected:
            raise ValueError("样本含重复的条件—被试身份")
        expected[key] = float(row["score_values"][row["active_dimension_id"]])
    conditions = sorted({c for c, _ in expected})
    subjects = sorted(s for c, s in expected if c == "target")
    if len(subjects) < 6:
        raise ValueError("至少需要6名target被试")
    if any({s for c, s in expected if c == condition} != set(subjects) for condition in conditions):
        raise ValueError("匹配条件的被试身份不一致")
    active = {c: np.array([expected[c, s] for s in subjects]) for c in conditions}
    matrices = {c: np.full((len(subjects), len(bank)), np.nan) for c in conditions}
    retest = np.full((len(subjects), len(bank)), np.nan)
    used = {}
    source_records = []
    input_files = {bank_file, blueprint_file, sample_file}
    protocol = None

    def block_values(frame, item, *, repeat=False):
        block = frame[frame["item_id"] == item["item_id"]].copy()
        if repeat:
            block = block[(block["condition_id"] == "target") & (block["administration_id"] == 2)]
        if block.empty:
            return None
        if set(block["item_version"]) != {item["version"]}:
            raise ValueError(f"作答版本与冻结题目不符：{item['item_id']}")
        key_cols = ["condition_id", "matched_subject_id"]
        if block.duplicated(key_cols).any():
            raise ValueError(f"重复作答记录：{item['item_id']}")
        expected_keys = {(c, s) for c in (["target"] if repeat else conditions) for s in subjects}
        actual_keys = set(zip(block.condition_id, block.matched_subject_id))
        if not actual_keys <= expected_keys:
            raise ValueError("作答包含其他样本/未知条件，禁止混用开发与评估数据")
        if actual_keys != expected_keys:
            return None
        for row in block.itertuples(index=False):
            key = (row.condition_id, row.matched_subject_id)
            correct = item["scoring_key"].get(row.selected_option_id)
            if correct is None or not np.isfinite(row.score) or abs(row.score - correct) > 1e-10:
                raise ValueError(f"作答计分与当前计分键不符：{item['item_id']}")
            if not np.isfinite(row.active_score) or abs(row.active_score - expected[key]) > 1e-8:
                raise ValueError("人格设定不一致，拒绝跨样本拼接")
        indexed = block.set_index(key_cols)["score"]
        return {c: np.array([indexed.loc[(c, s)] for s in subjects], dtype=float)
                for c in (["target"] if repeat else conditions)}

    for snapshot_path, manifest_path, scored_path, repeat_path in source_bundles(root, sample_role):
        if not all(p.exists() for p in (snapshot_path, manifest_path, scored_path)):
            continue
        snapshot = read_json(snapshot_path)
        source_items = snapshot["items"] if isinstance(snapshot, dict) else snapshot
        source_by_id = {item["item_id"]: item for item in source_items}
        matched = [j for j, item in enumerate(bank) if item["item_id"] in source_by_id
                   and content_hash(item) == content_hash(source_by_id[item["item_id"]])]
        if not matched:
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        if manifest.get("source_sha256") != sample["config"]["source_sha256"]:
            raise ValueError(f"源文件不属于所选样本：{manifest_path}")
        current_protocol = {k: manifest.get(k) for k in (
            "model_id", "prompt_version", "score_prompt_version", "generator_version",
            "conditions", "persona_modes", "virtual_sample_config")}
        if protocol is not None and current_protocol != protocol:
            raise ValueError("候选作答的模型/提示词/施测配置不一致，禁止拼接")
        protocol = current_protocol
        frame = pd.read_csv(scored_path)
        repeats = pd.read_csv(repeat_path) if repeat_path and repeat_path.exists() else None
        source_records.append({"snapshot": str(snapshot_path.relative_to(root)),
                               "matched_versions": len(matched), "model_id": manifest.get("model_id")})
        input_files.update((snapshot_path, manifest_path, scored_path))
        if repeats is not None:
            input_files.add(repeat_path)
        for j in matched:
            item = bank[j]
            # Earliest complete item block, never cherry-pick individual responses.
            if item["item_id"] in used:
                continue
            values = block_values(frame, item)
            if values is None:
                continue
            for c in conditions:
                matrices[c][:, j] = values[c]
            repeat_values = block_values(repeats, item, repeat=True) if repeats is not None else None
            if repeat_values is not None:
                retest[:, j] = repeat_values["target"]
            used[item["item_id"]] = {"snapshot": str(snapshot_path.relative_to(root)),
                                    "scores": str(scored_path.relative_to(root)),
                                    "content_hash": content_hash(item), "version": item["version"],
                                    "retest_available": repeat_values is not None}

    reference = pd.DataFrame(index=subjects)
    ref_file = root / "reference/neo_ffi/scores.csv"
    ref_info = {"available": False, "reason": "没有与所选样本匹配的NEO作答"}
    if ref_file.exists():
        raw_ref = pd.read_csv(ref_file)
        if raw_ref["matched_subject_id"].duplicated().any():
            raise ValueError("NEO包含重复被试身份")
        ref_ids = set(raw_ref["matched_subject_id"])
        if ref_ids == set(subjects):
            reference = raw_ref.set_index("matched_subject_id").loc[subjects]
            if not {"E", "N", "O", "A", "C"} <= set(reference.columns):
                raise ValueError("NEO缺少维度列")
            reference = reference[["E", "N", "O", "A", "C"]].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(reference.to_numpy()).all():
                raise ValueError("NEO包含缺失或非有限分数；不能静默删被试")
            ref_manifest_path = root / "reference/neo_ffi/manifest.json"
            ref_manifest = read_json(ref_manifest_path)
            if ref_manifest.get("status") != "completed":
                raise ValueError("NEO施测尚未完成")
            input_files.update((ref_file, ref_manifest_path))
            ref_info = {"available": True, "model_id": ref_manifest.get("model_id"),
                        "cross_model_exploration": ref_manifest.get("model_id") != (protocol or {}).get("model_id"),
                        "unavailable_domains": {d: "NEO维度分数无变异" for d in reference
                                                if reference[d].nunique() < 2}}
        elif ref_ids & set(subjects):
            raise ValueError("NEO与所选样本只有部分身份匹配，拒绝静默删被试")
    audit = {"sample_role": sample_role, "subject_count": len(subjects), "candidate_count": len(bank),
             "available_count": len(used), "missing_item_ids": [i for i in ids if i not in used],
             "item_sources": used, "source_bundles": source_records, "reference": ref_info,
             "protocol": protocol, "input_hashes": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(input_files)},
             "response_selection": "earliest complete frozen-version item block; no cross-role reuse",
             "interpretation": "探索性重组；若搜索使用原evaluation集，该集不再是本次搜索的独立验证集。"}
    return SearchData(root, bank, blueprint, subjects, matrices, active, retest, reference, audit)
