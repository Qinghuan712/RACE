from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    camera_id: str
    frame_id: int
    bbox: tuple[int, int, int, int]
    score: float = 1.0
    source_frame: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "proposal_id": self.proposal_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "bbox": list(self.bbox),
            "score": float(self.score),
            "source_frame": self.source_frame if self.source_frame is not None else self.frame_id,
        }
        if self.metadata:
            payload.update(self.metadata)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Proposal":
        known = {"proposal_id", "camera_id", "frame_id", "bbox", "score", "source_frame"}
        metadata = {k: v for k, v in payload.items() if k not in known}
        bbox = tuple(int(v) for v in payload["bbox"])
        return cls(
            proposal_id=str(payload.get("proposal_id") or ""),
            camera_id=str(payload["camera_id"]),
            frame_id=int(payload["frame_id"]),
            bbox=bbox,
            score=float(payload.get("score", 1.0)),
            source_frame=int(payload.get("source_frame", payload["frame_id"])),
            metadata=metadata,
        )


@dataclass(frozen=True)
class PairCalibration:
    src_cam: str
    ref_cam: str
    H: np.ndarray
    hull_src: np.ndarray | None
    hull_ref: np.ndarray | None
    tau: float
    pair_f1: float
    margin: float = 20.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def reversed(self) -> "PairCalibration":
        hull_src = None if self.hull_ref is None else self.hull_ref.copy()
        hull_ref = None if self.hull_src is None else self.hull_src.copy()
        return PairCalibration(
            src_cam=self.ref_cam,
            ref_cam=self.src_cam,
            H=np.linalg.inv(self.H),
            hull_src=hull_src,
            hull_ref=hull_ref,
            tau=self.tau,
            pair_f1=self.pair_f1,
            margin=self.margin,
            metadata=dict(self.metadata),
        )

    def to_json(self) -> dict[str, Any]:
        payload = {
            "src_cam": self.src_cam,
            "ref_cam": self.ref_cam,
            "H": self.H.tolist(),
            "tau": float(self.tau),
            "pair_f1": float(self.pair_f1),
            "margin": float(self.margin),
            "hull_src": None if self.hull_src is None else self.hull_src.reshape(-1, 2).tolist(),
            "hull_ref": None if self.hull_ref is None else self.hull_ref.reshape(-1, 2).tolist(),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PairCalibration":
        return cls(
            src_cam=str(payload["src_cam"]),
            ref_cam=str(payload["ref_cam"]),
            H=np.asarray(payload["H"], dtype=np.float32),
            hull_src=_list_to_hull(payload.get("hull_src")),
            hull_ref=_list_to_hull(payload.get("hull_ref")),
            tau=float(payload.get("tau", 0.0)),
            pair_f1=float(payload.get("pair_f1", 0.0)),
            margin=float(payload.get("margin", 20.0)),
            metadata=dict(payload.get("metadata", {})),
        )


def _list_to_hull(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr.reshape(-1, 1, 2)


def save_proposal_artifact(
    path: str | Path,
    proposals: Iterable[Proposal | dict[str, Any]],
    *,
    cameras: list[str] | None = None,
    frame_size: tuple[int, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    camera_set: set[str] = set(cameras or [])

    for raw in proposals:
        proposal = raw if isinstance(raw, Proposal) else Proposal.from_json(raw)
        camera_set.add(proposal.camera_id)
        cam_frames = grouped.setdefault(proposal.camera_id, {})
        cam_frames.setdefault(str(proposal.frame_id), []).append(proposal.to_json())

    payload = {
        "version": 1,
        "frame_size": list(frame_size) if frame_size is not None else None,
        "cameras": {cam: grouped.get(cam, {}) for cam in sorted(camera_set)},
        "metadata": metadata or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ProposalArtifactLoader:
    def __init__(self, payload: dict[str, Any], source_path: str | None = None):
        self.payload = payload
        self.source_path = source_path
        self.frame_size = tuple(payload.get("frame_size") or []) if payload.get("frame_size") else None
        self.metadata = dict(payload.get("metadata", {}))
        self._index: dict[str, dict[int, list[Proposal]]] = {}
        cameras = payload.get("cameras", {})
        for cam, frames in cameras.items():
            cam_index: dict[int, list[Proposal]] = {}
            for frame_key, records in frames.items():
                frame_id = int(frame_key)
                proposals = []
                for idx, record in enumerate(records):
                    if not record.get("proposal_id"):
                        record = dict(record)
                        record["proposal_id"] = f"{cam}_f{frame_id:06d}_p{idx:06d}"
                    proposals.append(Proposal.from_json(record))
                cam_index[frame_id] = proposals
            self._index[cam] = cam_index
        self.cameras = sorted(self._index.keys())

    @classmethod
    def from_file(cls, path: str | Path) -> "ProposalArtifactLoader":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload, source_path=str(path))

    def get_frame(self, camera_id: str, frame_id: int) -> list[Proposal]:
        return list(self._index.get(camera_id, {}).get(frame_id, []))

    def get_multi_camera_frame(
        self,
        frame_id: int,
        cameras: Iterable[str] | None = None,
    ) -> dict[str, list[Proposal]]:
        selected = cameras or self.cameras
        return {
            cam: self.get_frame(cam, frame_id)
            for cam in selected
            if self.get_frame(cam, frame_id)
        }

    def get_gop(
        self,
        frame_ids: Iterable[int],
        cameras: Iterable[str] | None = None,
    ) -> dict[int, dict[str, list[Proposal]]]:
        return {
            frame_id: self.get_multi_camera_frame(frame_id, cameras=cameras)
            for frame_id in frame_ids
        }


def save_homography_artifact(
    path: str | Path,
    calibrations: Iterable[PairCalibration | dict[str, Any]],
    *,
    cameras: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    pairs = []
    camera_set: set[str] = set(cameras or [])
    for raw in calibrations:
        calib = raw if isinstance(raw, PairCalibration) else PairCalibration.from_json(raw)
        camera_set.add(calib.src_cam)
        camera_set.add(calib.ref_cam)
        pairs.append(calib.to_json())

    payload = {
        "version": 1,
        "cameras": sorted(camera_set),
        "pairs": pairs,
        "metadata": metadata or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class HomographyArtifactLoader:
    def __init__(self, payload: dict[str, Any], source_path: str | None = None):
        self.payload = payload
        self.source_path = source_path
        self.metadata = dict(payload.get("metadata", {}))
        self.cameras = list(payload.get("cameras", []))
        self._pairs: dict[tuple[str, str], PairCalibration] = {}
        for record in payload.get("pairs", []):
            calib = PairCalibration.from_json(record)
            self._pairs[(calib.src_cam, calib.ref_cam)] = calib

    @classmethod
    def from_file(cls, path: str | Path) -> "HomographyArtifactLoader":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload, source_path=str(path))

    def get_pair(self, src_cam: str, ref_cam: str) -> PairCalibration | None:
        direct = self._pairs.get((src_cam, ref_cam))
        if direct is not None:
            return direct
        reverse = self._pairs.get((ref_cam, src_cam))
        if reverse is not None:
            return reverse.reversed()
        return None

    def iter_pairs(self) -> list[PairCalibration]:
        return list(self._pairs.values())
