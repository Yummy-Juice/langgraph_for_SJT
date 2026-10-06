"""CLI for the minimal behavior-evidence cache."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sjt_system.knowledge.behavior_evidence import (
    DEFAULT_OFFLINE_BEHAVIOR_ROOT,
    DEFAULT_CORPUS_PATH,
    DEFAULT_IPIP_SOURCE_PATH,
    NEO_FACET_CODE_TO_ID,
    load_behavior_evidence_bundle,
    load_ipip_corpus,
    parse_ipip_markdown,
    save_behavior_evidence_bundle,
    save_ipip_corpus,
)
from sjt_system.knowledge.behavior_evidence_agents import mine_behavior_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build behavior evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    parse_command = commands.add_parser("parse-ipip")
    parse_command.add_argument("--source", type=Path, default=DEFAULT_IPIP_SOURCE_PATH)
    parse_command.add_argument("--output", type=Path, default=DEFAULT_CORPUS_PATH)
    build = commands.add_parser("build")
    build.add_argument(
        "--facet",
        required=True,
        help="Facet code, comma-separated codes, or ALL.",
    )
    build.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    build.add_argument(
        "--output-root", type=Path, default=DEFAULT_OFFLINE_BEHAVIOR_ROOT
    )
    show = commands.add_parser("show")
    show.add_argument("path", type=Path)
    return parser


async def _build(args: argparse.Namespace) -> int:
    corpus = load_ipip_corpus(args.corpus)
    requested = args.facet.upper()
    requested_codes = (
        list(NEO_FACET_CODE_TO_ID)
        if requested == "ALL"
        else [code.strip() for code in requested.split(",")]
    )
    codes = list(dict.fromkeys(requested_codes))
    unknown = [code for code in codes if code not in NEO_FACET_CODE_TO_ID]
    if unknown:
        raise ValueError("未知 facet code：" + "、".join(unknown))

    async def mine_save_print(code: str) -> None:
        bundle = await mine_behavior_evidence(code, corpus)
        path = save_behavior_evidence_bundle(bundle, args.output_root)
        print(f"{code}: evidence={len(bundle.evidence)} -> {path}")

    results = await asyncio.gather(
        *(mine_save_print(code) for code in codes),
        return_exceptions=True,
    )
    first_error = next(
        (result for result in results if isinstance(result, BaseException)),
        None,
    )
    if first_error is not None:
        raise first_error
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "parse-ipip":
        corpus = parse_ipip_markdown(args.source)
        target = save_ipip_corpus(corpus, args.output)
        count = sum(len(scale.items) for scale in corpus.scales)
        print(f"Parsed {len(corpus.scales)} facets / {count} items -> {target}")
        return 0
    if args.command == "build":
        return asyncio.run(_build(args))
    if args.command == "show":
        bundle = load_behavior_evidence_bundle(args.path)
        print(f"{bundle.facet_code}: evidence={len(bundle.evidence)}")
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
