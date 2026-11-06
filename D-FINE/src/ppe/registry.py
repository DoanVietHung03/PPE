from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CANONICAL_CLASSES = frozenset(
    {
        "person",
        "helmet",
        "vest",
        "gloves",
        "boots",
        "no_helmet",
        "no_vest",
        "no_gloves",
        "no_boots",
        "mask",
        "no_mask",
        "eye_protection",
        "no_eye_protection",
        "long_pants",
        "short_pants",
    }
)
COMMERCIAL_LICENSES = frozenset({"Apache-2.0", "BSD-3-Clause", "CC-BY-4.0"})
STATUSES = frozenset({"approved", "pending_hash", "pending_clearance", "quarantined"})


@dataclass(frozen=True)
class SourceSpec:
    id: str
    status: str
    license: str
    evidence_url: str
    resolver: str | None
    version: str | int | None
    sha256: str | None
    mappings: dict[str, str]
    options: dict[str, Any]
    reason: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    status: str
    license: str
    evidence_url: str
    url: str | None
    sha256: str | None
    reason: str | None = None


@dataclass(frozen=True)
class Registry:
    sources: tuple[SourceSpec, ...]
    models: tuple[ModelSpec, ...]

    def approved_sources(self) -> tuple[SourceSpec, ...]:
        return tuple(source for source in self.sources if source.status == "approved")

    def require_approved_model(self, name: str) -> ModelSpec:
        model = next((model for model in self.models if model.id == name), None)
        if model is None:
            raise KeyError(f"Unknown model: {name}")
        if model.status != "approved":
            raise ValueError(f"Model {name} is not approved: {model.status}")
        return model


def _required(entry: dict[str, Any], field: str) -> Any:
    value = entry.get(field)
    if value in (None, ""):
        raise ValueError(f"{entry.get('id', 'entry')}: missing {field}")
    return value


def _validate_common(entry: dict[str, Any]) -> None:
    _required(entry, "id")
    status = _required(entry, "status")
    if status not in STATUSES:
        raise ValueError(f"{entry['id']}: invalid status {status}")
    _required(entry, "license")
    _required(entry, "evidence_url")
    if status == "approved":
        if entry["license"] not in COMMERCIAL_LICENSES:
            raise ValueError(f"{entry['id']}: license is not commercially allowlisted")
        digest = str(_required(entry, "sha256"))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{entry['id']}: invalid sha256")
    if status == "quarantined":
        _required(entry, "reason")


def _source(entry: dict[str, Any]) -> SourceSpec:
    status = entry.get("status")
    resolver = entry.get("resolver")
    version = entry.get("version")
    mappings = dict(entry.get("mappings") or {})
    if status in {"approved", "pending_hash"}:
        _required(entry, "resolver")
        _required(entry, "version")
        if not mappings:
            raise ValueError(f"{entry['id']}: missing mappings")
    _validate_common(entry)
    invalid = sorted(set(mappings.values()) - CANONICAL_CLASSES)
    if invalid:
        raise ValueError(f"{entry['id']}: invalid canonical class {invalid[0]}")
    reserved = {
        "id",
        "status",
        "license",
        "evidence_url",
        "resolver",
        "version",
        "sha256",
        "mappings",
        "reason",
    }
    return SourceSpec(
        id=entry["id"],
        status=status,
        license=entry["license"],
        evidence_url=entry["evidence_url"],
        resolver=resolver,
        version=version,
        sha256=entry.get("sha256"),
        mappings=mappings,
        options={key: value for key, value in entry.items() if key not in reserved},
        reason=entry.get("reason"),
    )


def _model(entry: dict[str, Any]) -> ModelSpec:
    _validate_common(entry)
    if entry["status"] in {"approved", "pending_hash"}:
        _required(entry, "url")
    return ModelSpec(
        id=entry["id"],
        status=entry["status"],
        license=entry["license"],
        evidence_url=entry["evidence_url"],
        url=entry.get("url"),
        sha256=entry.get("sha256"),
        reason=entry.get("reason"),
    )


def load_registry(path: Path) -> Registry:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sources = tuple(_source(dict(entry)) for entry in data.get("sources", []))
    models = tuple(_model(dict(entry)) for entry in data.get("models", []))
    ids = [item.id for item in (*sources, *models)]
    if len(ids) != len(set(ids)):
        raise ValueError("Registry IDs must be unique")
    return Registry(sources=sources, models=models)
