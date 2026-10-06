"""Versioned synthetic demographic catalogs and reproducible profiles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
from pathlib import Path
import random
from typing import Any


DEMOGRAPHICS_VERSION = "demographics-v1"
DEFAULT_RESPONSE_TEMPERATURE = 1.5
DEFAULT_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "data" / "demographics"
CATALOG_KINDS = ("age", "gender", "nationality", "education", "occupation", "income")
DEMOGRAPHIC_FIELDS = (
    "age", "gender", "nationality", "education", "occupation", "monthly_income_cny",
)


def resolve_response_temperature(value: object = None) -> float:
    if value is None:
        value = os.getenv("VIRTUAL_RESPONDENT_TEMPERATURE", str(DEFAULT_RESPONSE_TEMPERATURE))
    if isinstance(value, bool):
        raise ValueError("response_temperature must be a finite number between 0 and 2")
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("response_temperature must be a finite number between 0 and 2") from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("response_temperature must be a finite number between 0 and 2")
    if temperature != DEFAULT_RESPONSE_TEMPERATURE:
        raise ValueError("The current virtual response protocol requires temperature=1.5")
    return temperature


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _entries(snapshot: Mapping[str, Any], kind: str) -> list[Any]:
    return snapshot["catalogs"][kind]["entries"]


def _by_id(snapshot: Mapping[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in _entries(snapshot, kind)}


def validate_demographics_snapshot(snapshot: object) -> None:
    if not isinstance(snapshot, Mapping) or snapshot.get("schema_version") != 1:
        raise ValueError("Missing or unsupported demographic catalog snapshot")
    if snapshot.get("database_version") != DEMOGRAPHICS_VERSION:
        raise ValueError("Unsupported demographic database version; reconfigure the virtual sample")
    catalogs = snapshot.get("catalogs")
    if not isinstance(catalogs, Mapping) or set(catalogs) != set(CATALOG_KINDS):
        raise ValueError("Demographic snapshot must contain all six catalogs")
    for kind in CATALOG_KINDS:
        catalog = catalogs[kind]
        if (
            not isinstance(catalog, Mapping)
            or catalog.get("schema_version") != 1
            or catalog.get("database_version") != DEMOGRAPHICS_VERSION
            or catalog.get("kind") != kind
            or catalog.get("synthetic") is not True
            or not isinstance(catalog.get("entries"), list)
            or not catalog["entries"]
        ):
            raise ValueError(f"Invalid demographic catalog: {kind}")
        entries = catalog["entries"]
        if kind == "age":
            if any(not _integer(age) or not 12 <= age <= 60 for age in entries) or len(set(entries)) != len(entries):
                raise ValueError("Invalid or duplicate ages in demographic catalog")
            continue
        ids = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str) or not entry["id"]:
                raise ValueError(f"Invalid entry in demographic catalog: {kind}")
            ids.append(entry["id"])
            if kind != "income" and (not isinstance(entry.get("label"), str) or not entry["label"].strip()):
                raise ValueError(f"Missing demographic label: {kind}/{entry['id']}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate demographic IDs: {kind}")
    education = _by_id(snapshot, "education")
    for entry in education.values():
        if not _integer(entry.get("rank")) or not _integer(entry.get("minimum_age")):
            raise ValueError("Education requires integer rank and minimum_age")
    if len({entry["rank"] for entry in education.values()}) != len(education):
        raise ValueError("Education ranks must be unique")
    income = catalogs["income"]
    if income.get("currency") != "CNY" or income.get("period") != "month" or income.get("step") != 100:
        raise ValueError("Income must use the frozen CNY monthly, 100-yuan-step convention")
    for band in income["entries"]:
        if (
            not _integer(band.get("minimum")) or not _integer(band.get("maximum"))
            or not 0 <= band["minimum"] <= band["maximum"]
            or band["minimum"] % 100 or band["maximum"] % 100
        ):
            raise ValueError("Invalid synthetic income band")
    multipliers = income.get("education_multipliers")
    if not isinstance(multipliers, Mapping) or set(multipliers) != set(education):
        raise ValueError("Income multipliers must cover every education level")
    age_multipliers = income.get("age_multipliers")
    if not isinstance(age_multipliers, list) or not age_multipliers:
        raise ValueError("Missing income age multipliers")
    for row in age_multipliers:
        if not isinstance(row, Mapping) or not _integer(row.get("minimum_age")) or not _integer(row.get("maximum_age")):
            raise ValueError("Invalid income age multiplier")
    for value in [*multipliers.values(), *(row.get("multiplier") for row in age_multipliers)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Income multipliers must be positive finite numbers")
    for age in range(18, 61):
        if sum(row["minimum_age"] <= age <= row["maximum_age"] for row in age_multipliers) != 1:
            raise ValueError("Income age multiplier ranges must cover adults without overlap")
    bands = _by_id(snapshot, "income")
    occupations = _by_id(snapshot, "occupation")
    if "student" not in occupations:
        raise ValueError("Occupation catalog must include student")
    for entry in occupations.values():
        if (
            not _integer(entry.get("minimum_age"))
            or (entry.get("maximum_age") is not None and not _integer(entry["maximum_age"]))
            or entry.get("minimum_education_id") not in education
            or entry.get("income_band_id") not in bands
        ):
            raise ValueError(f"Invalid occupation constraints: {entry['id']}")


def load_demographics_snapshot(root: str | Path = DEFAULT_CATALOG_ROOT) -> dict[str, Any]:
    catalogs = {}
    for kind in CATALOG_KINDS:
        with (Path(root) / f"{kind}.json").open(encoding="utf-8") as handle:
            catalogs[kind] = json.load(handle)
    snapshot = {"schema_version": 1, "database_version": DEMOGRAPHICS_VERSION, "catalogs": catalogs}
    validate_demographics_snapshot(snapshot)
    return snapshot


def demographic_settings(
    diagnostics: Mapping[str, Any], *, seed: int, response_temperature: object = None,
) -> dict[str, Any]:
    snapshot = diagnostics.get("demographics_snapshot") or load_demographics_snapshot()
    validate_demographics_snapshot(snapshot)
    return {
        "demographics_version": DEMOGRAPHICS_VERSION,
        "demographics_seed": seed,
        "demographics_snapshot": deepcopy(snapshot),
        "response_temperature": resolve_response_temperature(response_temperature),
    }


def validate_demographics_config(config: Mapping[str, Any]) -> None:
    if config.get("demographics_version") != DEMOGRAPHICS_VERSION or not _integer(config.get("demographics_seed")):
        raise ValueError("Virtual sample lacks frozen demographics; reconfigure without reusing old responses")
    validate_demographics_snapshot(config.get("demographics_snapshot"))
    temperature = config.get("response_temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("Virtual sample lacks a frozen response_temperature; reconfigure")
    resolve_response_temperature(temperature)


def _eligible_occupations(age: int, education_id: str, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    rank = _by_id(snapshot, "education")[education_id]["rank"]
    education = _by_id(snapshot, "education")
    candidates = [entry for entry in _entries(snapshot, "occupation") if (
        entry["minimum_age"] <= age
        and (entry.get("maximum_age") is None or age <= entry["maximum_age"])
        and education[entry["minimum_education_id"]]["rank"] <= rank
    )]
    if age < 18:
        candidates = [entry for entry in candidates if entry["id"] == "student"]
    return candidates


def _income_amount(base: int, age: int, education_id: str, snapshot: Mapping[str, Any]) -> int:
    if base == 0:
        return 0
    income = snapshot["catalogs"]["income"]
    age_factor = next(row["multiplier"] for row in income["age_multipliers"] if row["minimum_age"] <= age <= row["maximum_age"])
    amount = Decimal(base) * Decimal(str(age_factor)) * Decimal(str(income["education_multipliers"][education_id]))
    return int((amount / Decimal(100)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) * 100


def generate_demographics(seed: str | int, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Use streams independent of facet sampling and demographic backgrounds."""

    rng = random.Random(f"{seed}:demographics")
    age = rng.choice(_entries(snapshot, "age"))
    education = rng.choice([entry for entry in _entries(snapshot, "education") if entry["minimum_age"] <= age])
    occupations = _eligible_occupations(age, education["id"], snapshot)
    if not occupations:
        raise ValueError(f"No eligible occupation for age={age}, education={education['id']}")
    occupation = rng.choice(occupations)
    band = _by_id(snapshot, "income")[occupation["income_band_id"]]
    base = rng.randrange(band["minimum"], band["maximum"] + 100, 100)
    return {
        "age": age,
        "gender": random.Random(f"{seed}:gender").choice(_entries(snapshot, "gender"))["id"],
        "nationality": random.Random(f"{seed}:nationality").choice(_entries(snapshot, "nationality"))["id"],
        "education": education["id"],
        "occupation": occupation["id"],
        "monthly_income_cny": _income_amount(base, age, education["id"], snapshot),
    }


def validate_demographics(profile: object, snapshot: Mapping[str, Any]) -> None:
    if not isinstance(profile, Mapping) or set(profile) != set(DEMOGRAPHIC_FIELDS):
        raise ValueError("Each virtual respondent must retain all six demographic fields")
    age = profile["age"]
    if not _integer(age) or age not in _entries(snapshot, "age"):
        raise ValueError("Invalid demographic age")
    for kind in ("gender", "nationality", "education", "occupation"):
        value = profile[kind]
        if not isinstance(value, str) or value not in _by_id(snapshot, kind):
            raise ValueError(f"Unknown demographic {kind}")
    education = _by_id(snapshot, "education")[profile["education"]]
    if age < education["minimum_age"]:
        raise ValueError("Education is incompatible with age")
    occupations = _eligible_occupations(age, profile["education"], snapshot)
    if profile["occupation"] not in {entry["id"] for entry in occupations}:
        raise ValueError("Occupation is incompatible with age or education")
    occupation = _by_id(snapshot, "occupation")[profile["occupation"]]
    band = _by_id(snapshot, "income")[occupation["income_band_id"]]
    amount = profile["monthly_income_cny"]
    if not _integer(amount) or amount < 0 or amount % 100:
        raise ValueError("Monthly income must be a nonnegative multiple of 100 CNY")
    minimum = _income_amount(band["minimum"], age, profile["education"], snapshot)
    maximum = _income_amount(band["maximum"], age, profile["education"], snapshot)
    if not minimum <= amount <= maximum:
        raise ValueError("Income is incompatible with age, education or occupation")


def demographics_to_columns(profile: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    validate_demographics(profile, snapshot)
    return {
        "age": profile["age"],
        **{kind: _by_id(snapshot, kind)[profile[kind]]["label"] for kind in ("gender", "nationality", "education", "occupation")},
        "monthly_income_cny": profile["monthly_income_cny"],
    }
