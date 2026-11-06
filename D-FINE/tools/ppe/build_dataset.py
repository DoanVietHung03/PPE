import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.ppe.dataset import (
    HashReviewRequired,
    build_dataset,
    convert_source,
    fetch_source,
    safe_extract_zip,
)
from src.ppe.registry import load_registry


CANDIDATES = {
    "ppe-13c": {
        "source": "roboflow_ppe_compliance_v2",
        "detector_classes": ("person", "helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves", "boots", "no_boots", "mask", "no_mask", "eye_protection", "no_eye_protection"),
        "state_classes": ("helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves", "boots", "no_boots", "mask", "no_mask", "eye_protection", "no_eye_protection"),
        "context_classes": ("gloves", "boots"),
    },
    "pants-2c": {
        "source": "roboflow_pants_v2_v1",
        "detector_classes": ("long_pants", "short_pants"),
        "state_classes": ("long_pants", "short_pants"),
        "context_classes": (),
    },
}

def _sources(registry, selected):
    sources = [
        source
        for source in registry.sources
        if source.status in {"approved", "pending_hash"}
    ]
    if selected:
        sources = [source for source in sources if source.id == selected]
        if not sources:
            raise ValueError(f"Source is not fetchable: {selected}")
    return sources


def fetch(args) -> int:
    registry = load_registry(args.registry)
    review_required = False
    for source in _sources(registry, args.source):
        try:
            path = fetch_source(source, args.cache)
            print(f"verified {source.id}: {path}")
        except HashReviewRequired as review:
            review_required = True
            print(
                f"review-required {review.source_id}: sha256={review.sha256} path={review.path}",
                file=sys.stderr,
            )
    return 2 if review_required else 0


def _extracted_root(source, archive, cache):
    destination = cache / "extracted" / archive.stem
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        safe_extract_zip(archive, temporary)
        temporary.replace(destination)
    if source.options.get("format") == "yolo":
        configs = sorted(destination.rglob("data.yaml"))
        if len(configs) != 1:
            raise ValueError(f"{source.id}: expected exactly one data.yaml")
        return configs[0].parent
    return destination


def _publish_current(output: Path, build_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    link = output / f".current-{build_id}.tmp"
    pointer = output / ".CURRENT.tmp"
    link.unlink(missing_ok=True)
    link.symlink_to(build_id, target_is_directory=True)
    try:
        os.replace(link, output / "current")
        pointer.write_text(build_id + "\n", encoding="utf-8")
        os.replace(pointer, output / "CURRENT")
    finally:
        link.unlink(missing_ok=True)
        pointer.unlink(missing_ok=True)


def build(args) -> int:
    registry = load_registry(args.registry)
    records = []
    rejections = []
    licenses = []
    for source in registry.approved_sources():
        licenses.append({"source_id": source.id, "license": source.license, "evidence_url": source.evidence_url, "version": source.version, "sha256": source.sha256})
        archive = fetch_source(source, args.cache)
        converted = convert_source(source, _extracted_root(source, archive, args.cache))
        records.extend(converted.records)
        rejections.extend(
            {
                "source_id": source.id,
                "source_image_id": rejection.source_image_id,
                "code": rejection.code,
                "detail": rejection.detail,
            }
            for rejection in converted.rejections
        )
    report = build_dataset(records, args.output, seed=args.seed, rejections=rejections, licenses=licenses)
    _publish_current(args.output, report.build_id)
    print(f"build_id={report.build_id} path={report.path}")
    return 0


def build_candidate(args) -> int:
    candidate = CANDIDATES[args.candidate]
    if args.output.name != args.candidate:
        raise ValueError(f"output directory must end with {args.candidate}")
    registry = load_registry(args.registry)
    source = next((item for item in registry.sources if item.id == candidate["source"]), None)
    if source is None or source.status != "pending_clearance" or not source.sha256:
        raise ValueError(f"Candidate source is not pinned for review: {candidate['source']}")
    archive = fetch_source(source, args.cache)
    converted = convert_source(source, _extracted_root(source, archive, args.cache))
    rejections = [
        {"source_id": source.id, "source_image_id": item.source_image_id, "code": item.code, "detail": item.detail}
        for item in converted.rejections
    ]
    licenses = [{"source_id": source.id, "license": source.license, "evidence_url": source.evidence_url, "version": source.version, "sha256": source.sha256}]
    report = build_dataset(
        converted.records, args.output, seed=args.seed, rejections=rejections, licenses=licenses,
        detector_classes=candidate["detector_classes"], state_classes=candidate["state_classes"],
        context_classes=candidate["context_classes"],
    )
    _publish_current(args.output, report.build_id)
    print(f"build_id={report.build_id} path={report.path}")
    return 0


def verify(args) -> int:
    dataset = Path(args.dataset)
    if dataset.name == "current":
        build_id = (dataset.parent / "CURRENT").read_text(encoding="utf-8").strip()
        dataset = dataset.parent / build_id
    audit = __import__("json").loads((dataset / "audit.json").read_text(encoding="utf-8"))
    if audit["cross_split_duplicate_groups"]:
        raise ValueError("cross-split duplicate leakage detected")
    print(f"verified build_id={audit['build_id']} images={audit['images']}")
    return 0


def parser():
    command = argparse.ArgumentParser()
    subcommands = command.add_subparsers(required=True)
    fetch_command = subcommands.add_parser("fetch")
    fetch_command.add_argument("--registry", type=Path, required=True)
    fetch_command.add_argument("--cache", type=Path, required=True)
    fetch_command.add_argument("--source")
    fetch_command.set_defaults(run=fetch)
    build_command = subcommands.add_parser("build")
    build_command.add_argument("--registry", type=Path, required=True)
    build_command.add_argument("--cache", type=Path, required=True)
    build_command.add_argument("--output", type=Path, required=True)
    build_command.add_argument("--seed", type=int, default=20260901)
    build_command.set_defaults(run=build)
    candidate_command = subcommands.add_parser("build-candidate")
    candidate_command.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    candidate_command.add_argument("--registry", type=Path, required=True)
    candidate_command.add_argument("--cache", type=Path, required=True)
    candidate_command.add_argument("--output", type=Path, required=True)
    candidate_command.add_argument("--seed", type=int, default=20260923)
    candidate_command.set_defaults(run=build_candidate)
    verify_command = subcommands.add_parser("verify")
    verify_command.add_argument("--dataset", type=Path, required=True)
    verify_command.set_defaults(run=verify)
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
