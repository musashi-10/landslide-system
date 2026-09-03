"""
Model Registry
==============

Handles persistence and retrieval of trained model artifacts.

An artifact directory is structured as:
    <model_dir>/
        <version>/
            model.pkl
            preprocessor.pkl
            metadata.json

ModelRegistry scans the artifact directory and loads/saves artifacts.
If no artifacts are present, latest_version() returns None so that
the Predictor can gracefully fall back to MockPredictor.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """
    A loaded model artifact containing everything needed for inference.

    Attributes
    ----------
    model : object
        Fitted sklearn-compatible estimator (has predict_proba method).
    preprocessor : object
        Fitted LandslidePreprocessor (has get_feature_matrix method).
    feature_cols : list[str]
        Ordered list of feature column names used during training.
    version : str
        Version string, e.g. "v1".
    model_type : str
        Human-readable model class name, e.g. "RandomForestClassifier".
    metadata : dict
        Additional metadata stored alongside the artifact (optional).
    """

    model: Any
    preprocessor: Any
    feature_cols: List[str]
    version: str
    model_type: str
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    """
    Filesystem-based model artifact registry.

    Parameters
    ----------
    model_dir : str or Path
        Root directory that contains versioned model subdirectories.
        If the directory does not exist, the registry is treated as empty.

    Directory layout expected
    -------------------------
        <model_dir>/
            v1/
                model.pkl
                preprocessor.pkl
                metadata.json       ← optional
            v2/
                ...

    Usage
    -----
        registry = ModelRegistry(model_dir="ml/models/artifacts")
        version  = registry.latest_version()   # None if no artifacts
        artifact = registry.load(version)       # ModelArtifact
    """

    _MODEL_FILENAME = "model.pkl"
    _PREPROCESSOR_FILENAME = "preprocessor.pkl"
    _METADATA_FILENAME = "metadata.json"

    def __init__(self, model_dir: str = "ml/models/artifacts") -> None:
        self._model_dir = Path(model_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def latest_version(self) -> Optional[str]:
        """
        Return the name of the most recently saved version, or None.

        Versions are sorted lexicographically — this is correct for
        the "v1", "v2", ... naming convention.  If the artifact
        directory does not exist or is empty, returns None.
        """
        versions = self._available_versions()
        if not versions:
            return None
        return sorted(versions)[-1]

    def exists(self, version: str) -> bool:
        """Return True if a complete artifact for *version* exists."""
        artifact_dir = self._model_dir / version
        return (
            artifact_dir.is_dir()
            and (artifact_dir / self._MODEL_FILENAME).is_file()
            and (artifact_dir / self._PREPROCESSOR_FILENAME).is_file()
        )

    def load(self, version: str) -> ModelArtifact:
        """
        Load and return a ModelArtifact for the given version.

        Parameters
        ----------
        version : str
            E.g. "v1".

        Returns
        -------
        ModelArtifact

        Raises
        ------
        FileNotFoundError : if the artifact directory or required files are missing.
        """
        artifact_dir = self._model_dir / version

        if not artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Artifact directory not found: {artifact_dir}"
            )

        model_path = artifact_dir / self._MODEL_FILENAME
        preprocessor_path = artifact_dir / self._PREPROCESSOR_FILENAME

        for path in (model_path, preprocessor_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required artifact file missing: {path}")

        model = _load_pickle(model_path)
        preprocessor = _load_pickle(preprocessor_path)

        # Load optional metadata
        metadata: dict = {}
        metadata_path = artifact_dir / self._METADATA_FILENAME
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)

        feature_cols: List[str] = metadata.get("feature_cols", [])
        model_type: str = metadata.get(
            "model_type", type(model).__name__
        )

        logger.info(
            f"Loaded model artifact: version={version}, "
            f"model_type={model_type}, features={len(feature_cols)}"
        )

        return ModelArtifact(
            model=model,
            preprocessor=preprocessor,
            feature_cols=feature_cols,
            version=version,
            model_type=model_type,
            metadata=metadata,
        )

    def save(
        self,
        model: Any,
        preprocessor: Any,
        feature_cols: List[str],
        version: str,
        extra_metadata: Optional[dict] = None,
    ) -> Path:
        """
        Persist a trained model artifact to the registry.

        Parameters
        ----------
        model : fitted sklearn-compatible estimator
        preprocessor : fitted LandslidePreprocessor
        feature_cols : list[str]
        version : str  — e.g. "v1"
        extra_metadata : dict, optional

        Returns
        -------
        Path to the artifact directory.
        """
        artifact_dir = self._model_dir / version
        artifact_dir.mkdir(parents=True, exist_ok=True)

        _save_pickle(model, artifact_dir / self._MODEL_FILENAME)
        _save_pickle(preprocessor, artifact_dir / self._PREPROCESSOR_FILENAME)

        metadata = {
            "version": version,
            "model_type": type(model).__name__,
            "feature_cols": feature_cols,
            **(extra_metadata or {}),
        }
        with (artifact_dir / self._METADATA_FILENAME).open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(metadata, fh, indent=2)

        logger.info(f"Saved model artifact: {artifact_dir}")
        return artifact_dir

    # ── Internals ─────────────────────────────────────────────────────────────

    def _available_versions(self) -> List[str]:
        """Return a list of version names that have a complete artifact set."""
        if not self._model_dir.is_dir():
            return []
        return [
            d.name
            for d in self._model_dir.iterdir()
            if d.is_dir() and self.exists(d.name)
        ]


# ── Pickle helpers ────────────────────────────────────────────────────────────


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
