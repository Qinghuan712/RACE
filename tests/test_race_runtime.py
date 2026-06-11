from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from RACE.artifacts import HomographyArtifactLoader, PairCalibration, Proposal, ProposalArtifactLoader, save_homography_artifact, save_proposal_artifact
from RACE.core import BIN_H, BIN_W, CandidatePatch, PlacementEntry, blend_back_frame, blend_back_frame_torch, blend_back_frames_torch, compute_importance, normalize_candidate_importance, optimize_launch_plan, pack_objects, run_sr_batch, select_candidates_under_bin_budget
from RACE.runtime import GOPOrchestrator, OnlineProposalGenerator, ProposalMatcher


class RaceRuntimeTests(unittest.TestCase):
    def test_proposal_loader_indexes_by_camera_and_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "proposals.json"
            proposals = [
                Proposal("p1", "c001", 10, (1, 2, 3, 4), score=0.7),
                Proposal("p2", "c002", 10, (5, 6, 7, 8), score=0.8),
            ]
            save_proposal_artifact(path, proposals, cameras=["c001", "c002"])
            loader = ProposalArtifactLoader.from_file(path)
            self.assertEqual(len(loader.get_frame("c001", 10)), 1)
            self.assertEqual(loader.get_frame("c001", 10)[0].bbox, (1, 2, 3, 4))
            self.assertEqual(set(loader.get_multi_camera_frame(10).keys()), {"c001", "c002"})

    def test_proposal_loader_generates_missing_ids(self) -> None:
        payload = {
            "version": 1,
            "cameras": {
                "c001": {
                    "1": [
                        {"camera_id": "c001", "frame_id": 1, "bbox": [1, 2, 3, 4], "score": 1.0},
                    ]
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "proposals.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loader = ProposalArtifactLoader.from_file(path)
            self.assertTrue(loader.get_frame("c001", 1)[0].proposal_id.startswith("c001_f000001_p"))

    def test_homography_loader_recovers_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "homography.json"
            calibration = PairCalibration(
                src_cam="c001",
                ref_cam="c002",
                H=np.eye(3, dtype=np.float32),
                hull_src=np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.float32),
                hull_ref=np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.float32),
                tau=15.0,
                pair_f1=0.9,
            )
            save_homography_artifact(path, [calibration], cameras=["c001", "c002"])
            loader = HomographyArtifactLoader.from_file(path)
            restored = loader.get_pair("c001", "c002")
            self.assertIsNotNone(restored)
            self.assertTrue(np.allclose(restored.H, np.eye(3)))
            self.assertAlmostEqual(restored.tau, 15.0)

    def test_matcher_clusters_same_object_across_cameras(self) -> None:
        loader = HomographyArtifactLoader(
            {
                "version": 1,
                "cameras": ["c001", "c002"],
                "pairs": [
                    {
                        "src_cam": "c001",
                        "ref_cam": "c002",
                        "H": np.eye(3, dtype=np.float32).tolist(),
                        "tau": 20.0,
                        "pair_f1": 0.8,
                        "margin": 0.0,
                        "hull_src": None,
                        "hull_ref": None,
                    }
                ],
            }
        )
        matcher = ProposalMatcher(loader, min_pair_f1=0.5, score_floor=0.0)
        proposals_by_camera = {
            "c001": [Proposal("a", "c001", 1, (10, 10, 10, 10))],
            "c002": [
                Proposal("b", "c002", 1, (12, 11, 10, 10)),
                Proposal("c", "c002", 1, (200, 200, 10, 10)),
            ],
        }
        clusters = matcher.match_frame(1, proposals_by_camera)
        sizes = sorted(len(cluster.proposals) for cluster in clusters)
        self.assertEqual(sizes, [1, 2])

    def test_single_view_clusters_keep_each_proposal_independent(self) -> None:
        proposals_by_camera = {
            "c001": [Proposal("a", "c001", 1, (10, 10, 10, 10))],
            "c002": [Proposal("b", "c002", 1, (12, 11, 10, 10))],
        }

        clusters = GOPOrchestrator._single_view_clusters(1, proposals_by_camera)

        self.assertEqual(len(clusters), 2)
        self.assertEqual([len(cluster.proposals) for cluster in clusters], [1, 1])
        self.assertEqual({cluster.proposals[0].proposal_id for cluster in clusters}, {"a", "b"})

    def test_packer_is_deterministic(self) -> None:
        patch = np.zeros((32, 32, 3), dtype=np.uint8)
        candidates = [
            CandidatePatch("c2", "c001", 2, patch, (0, 0, 32, 32), 0.5),
            CandidatePatch("c1", "c001", 1, patch, (0, 0, 32, 32), 1.0),
        ]
        _, placements, _ = pack_objects(candidates, num_bins=1)
        self.assertEqual([p.cluster_id for p in placements], ["c1", "c2"])

    def test_packer_opens_new_bins_on_demand(self) -> None:
        full_bin_patch = np.zeros((BIN_H, BIN_W, 3), dtype=np.uint8)
        candidates = [
            CandidatePatch("c1", "c001", 1, full_bin_patch, (0, 0, BIN_W, BIN_H), 1.0),
            CandidatePatch("c2", "c001", 1, full_bin_patch, (0, 0, BIN_W, BIN_H), 0.9),
        ]

        bins, placements, deferred = pack_objects(candidates, num_bins=1)

        self.assertEqual(len(bins), 2)
        self.assertEqual([placement.bin_idx for placement in placements], [0, 1])
        self.assertFalse(deferred)

    def test_budget_admission_prioritizes_importance(self) -> None:
        full_bin_patch = np.zeros((BIN_H, BIN_W, 3), dtype=np.uint8)
        candidates = [
            CandidatePatch("low", "c001", 1, full_bin_patch, (0, 0, BIN_W, BIN_H), 0.1),
            CandidatePatch("high", "c001", 1, full_bin_patch, (0, 0, BIN_W, BIN_H), 1.0),
        ]

        selected, deferred = select_candidates_under_bin_budget(candidates, max_bins=1)

        self.assertEqual([candidate.cluster_id for candidate in selected], ["high"])
        self.assertEqual([candidate.cluster_id for candidate in deferred], ["low"])

    def test_maxrects_split_keeps_multiple_free_regions(self) -> None:
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        candidates = [
            CandidatePatch("a", "c001", 1, patch, (0, 0, 64, 64), 1.0),
            CandidatePatch("b", "c001", 1, patch, (0, 0, 64, 64), 0.9),
        ]

        bins, placements, deferred = pack_objects(candidates)

        self.assertEqual(len(bins), 1)
        self.assertEqual(len(placements), 2)
        self.assertFalse(deferred)
        self.assertGreaterEqual(len(bins[0].free), 2)

    def test_optimize_launch_plan_prefers_lower_total_latency(self) -> None:
        plan = optimize_launch_plan(6, {1: 1.0, 2: 1.8, 4: 3.3, 8: 7.5})
        self.assertEqual(plan, [4, 2])

    def test_importance_raw_score_is_non_negative(self) -> None:
        importance = compute_importance(
            {
                "c001": (0, 0, 1, 1),
                "c002": (0, 0, 1, 1),
            },
            {
                "c001": -10.0,
                "c002": 10.0,
            },
            "c001",
        )

        self.assertGreaterEqual(importance, 0.0)

    def test_candidate_importance_is_normalized(self) -> None:
        patch = np.zeros((16, 16, 3), dtype=np.uint8)
        candidates = [
            CandidatePatch("c1", "c001", 1, patch, (0, 0, 16, 16), 10.0),
            CandidatePatch("c2", "c001", 1, patch, (0, 0, 16, 16), 20.0),
            CandidatePatch("c3", "c001", 1, patch, (0, 0, 16, 16), 30.0),
        ]

        normalize_candidate_importance(candidates)

        self.assertEqual([candidate.importance for candidate in candidates], [0.0, 0.5, 1.0])

    def test_run_sr_batch_splits_batches_by_launch_plan(self) -> None:
        class TrackingSR:
            def __init__(self) -> None:
                self.seen_batch_sizes: list[int] = []

            def inference(self, batch: torch.Tensor):
                self.seen_batch_sizes.append(int(batch.shape[0]))
                return torch.nn.functional.interpolate(
                    batch,
                    scale_factor=3,
                    mode="bilinear",
                    align_corners=False,
                )

        sr_model = TrackingSR()
        bin_images = [np.zeros((BIN_H, BIN_W, 3), dtype=np.uint8) for _ in range(3)]
        sr_bins = run_sr_batch(sr_model, bin_images, device="cpu", launch_plan=[2, 1])

        self.assertEqual(sr_model.seen_batch_sizes, [2, 1])
        self.assertEqual(len(sr_bins), 3)
        self.assertEqual(sr_bins[0].shape[:2], (BIN_H * 3, BIN_W * 3))

        sr_bins_tensor = run_sr_batch(sr_model, bin_images, device="cpu", launch_plan=[2, 1], return_tensors=True)
        self.assertTrue(torch.is_tensor(sr_bins_tensor[0]))

    def test_blend_back_only_updates_roi(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        sr_bin = np.full((BIN_H * 3, BIN_W * 3, 3), 255, dtype=np.uint8)
        placements = [
            PlacementEntry(
                cluster_id="x",
                cam_id="c001",
                frame_id=1,
                bin_idx=0,
                bx=0,
                by=0,
                bw=10,
                bh=10,
                orig_bbox=(5, 5, 10, 10),
                orig_w=10,
                orig_h=10,
                importance=1.0,
            )
        ]
        blended = blend_back_frame(frame, [sr_bin], placements)
        self.assertEqual(int(blended[0, 0].sum()), 0)
        self.assertGreater(int(blended[20, 20].sum()), 0)

    def test_blend_back_torch_only_updates_roi(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        sr_bin = torch.full((3, BIN_H * 3, BIN_W * 3), 255.0, dtype=torch.float32)
        placements = [
            PlacementEntry(
                cluster_id="x",
                cam_id="c001",
                frame_id=1,
                bin_idx=0,
                bx=0,
                by=0,
                bw=10,
                bh=10,
                orig_bbox=(5, 5, 10, 10),
                orig_w=10,
                orig_h=10,
                importance=1.0,
            )
        ]
        blended = blend_back_frame_torch(frame, [sr_bin], placements, device="cpu")
        self.assertEqual(int(blended[:, 0, 0].sum().item()), 0)
        self.assertGreater(int(blended[:, 20, 20].sum().item()), 0)

    def test_batch_blend_matches_single_frame_blend(self) -> None:
        frame_a = np.zeros((360, 640, 3), dtype=np.uint8)
        frame_b = np.full((360, 640, 3), 16, dtype=np.uint8)
        sr_bin = torch.full((3, BIN_H * 3, BIN_W * 3), 255.0, dtype=torch.float32)
        placements = [
            PlacementEntry(
                cluster_id="x",
                cam_id="c001",
                frame_id=1,
                bin_idx=0,
                bx=0,
                by=0,
                bw=10,
                bh=10,
                orig_bbox=(5, 5, 10, 10),
                orig_w=10,
                orig_h=10,
                importance=1.0,
            )
        ]

        singles = [
            blend_back_frame_torch(frame_a, [sr_bin], placements, device="cpu"),
            blend_back_frame_torch(frame_b, [sr_bin], [], device="cpu"),
        ]
        batched = blend_back_frames_torch(
            [frame_a, frame_b],
            [sr_bin],
            [placements, []],
            device="cpu",
        )
        for single, batch_item in zip(singles, batched):
            self.assertTrue(torch.equal(single, batch_item))

    def test_orchestrator_smoke(self) -> None:
        class FakeSR:
            def inference(self, batch: torch.Tensor):
                outputs = []
                for img in batch:
                    up = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        scale_factor=3,
                        mode="bilinear",
                        align_corners=False,
                    )
                    outputs.append(up)
                return outputs

        class FakeDetector:
            def inference_raw(self, img_tensor: torch.Tensor):
                _ = img_tensor
                boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
                scores = np.array([0.9], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                return np.array([1], dtype=np.int32), boxes, scores, labels

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_dir = tmp / "videos"
            video_dir.mkdir()
            cameras = ["c001", "c002", "c003", "c004"]
            for cam_idx, cam in enumerate(cameras):
                writer = cv2.VideoWriter(
                    str(video_dir / f"{cam}_aligned.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    5.0,
                    (640, 360),
                )
                for frame_idx in range(2):
                    frame = np.full((360, 640, 3), 10 * (cam_idx + 1) + frame_idx, dtype=np.uint8)
                    cv2.rectangle(frame, (20 + frame_idx * 4, 40), (100 + frame_idx * 4, 120), (255, 255, 255), -1)
                    writer.write(frame)
                writer.release()

            pair_records = []
            for idx in range(len(cameras) - 1):
                pair_records.append(
                    PairCalibration(
                        src_cam=cameras[idx],
                        ref_cam=cameras[idx + 1],
                        H=np.eye(3, dtype=np.float32),
                        hull_src=None,
                        hull_ref=None,
                        tau=100.0,
                        pair_f1=0.9,
                    )
                )
            homography_path = tmp / "homography.json"
            save_homography_artifact(homography_path, pair_records, cameras=cameras)

            homography_loader = HomographyArtifactLoader.from_file(homography_path)
            matcher = ProposalMatcher(homography_loader)
            output_dir = tmp / "output"
            proposal_cache = tmp / "proposal_cache.json"
            proposal_generator = OnlineProposalGenerator(
                cameras,
                cfg={
                    "warmup_frames": 0,
                    "detect_shadows": False,
                    "min_area": 100,
                },
            )

            orchestrator = GOPOrchestrator(
                video_dir=str(video_dir),
                matcher=matcher,
                sr_model=FakeSR(),
                detector=FakeDetector(),
                output_dir=str(output_dir),
                proposal_generator=proposal_generator,
                proposal_cache_path=str(proposal_cache),
                gop=2,
                num_bins=2,
                detection_device="cpu",
                sr_device="cpu",
            )
            try:
                summaries = orchestrator.run(start_frame=1, num_frames=2)
            finally:
                orchestrator.close()

            self.assertEqual(len(summaries), 1)
            self.assertTrue((output_dir / "runtime_stats.json").exists())
            self.assertTrue((output_dir / "detections.json").exists())
            self.assertTrue(any((output_dir / "visualizations").iterdir()))
            self.assertTrue(proposal_cache.exists())


if __name__ == "__main__":
    unittest.main()
