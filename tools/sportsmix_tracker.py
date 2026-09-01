#!/usr/bin/env python3
"""Lightweight SportsMOT/MixSort-inspired multi-player association.

The original MixSort tracker couples a motion tracker with a learned MixFormer
appearance-association network.  Pulling that research stack into this project
would require its detector, checkpoints, CUDA-only inference code, and an older
PyTorch environment.  This module keeps the part that can be evaluated with the
models already present here:

* predict every active track with a velocity model;
* associate detections with a weighted motion/IoU and appearance cost;
* do not update an appearance template from an overlapped crop;
* keep unmatched tracks alive as predictions through short occlusions.

It is deliberately named ``sportsmix`` rather than ``mixsort`` so generated
results do not claim to reproduce the paper's learned MixFormer association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class SportsMixOutput:
    track_id: int
    bbox: tuple[float, float, float, float]
    source_index: int | None
    observed: bool
    confidence: float
    prediction_age: int = 0


@dataclass
class _SportsMixState:
    track_id: int
    bbox: np.ndarray
    embedding: np.ndarray | None
    velocity: np.ndarray
    confidence: float
    last_frame: int
    missed: int = 0
    hits: int = 1


class SportsMixTracker:
    """Fuse motion/IoU and OSNet appearance for sports-player association."""

    def __init__(
        self,
        *,
        max_age: int = 30,
        match_threshold: float = 0.92,
        motion_weight: float = 0.55,
        appearance_weight: float = 0.35,
        center_weight: float = 0.10,
        appearance_scale: float = 1.0,
        max_center_distance: float = 3.0,
        template_momentum: float = 0.85,
        overlap_iou: float = 0.25,
    ) -> None:
        weights = motion_weight + appearance_weight + center_weight
        if weights <= 0:
            raise ValueError("SportsMix association weights must sum to a positive value")
        self.max_age = max(0, int(max_age))
        self.match_threshold = float(match_threshold)
        self.motion_weight = float(motion_weight) / weights
        self.appearance_weight = float(appearance_weight) / weights
        self.center_weight = float(center_weight) / weights
        self.appearance_scale = max(1e-6, float(appearance_scale))
        self.max_center_distance = max(0.0, float(max_center_distance))
        self.template_momentum = min(1.0, max(0.0, float(template_momentum)))
        self.overlap_iou = max(0.0, float(overlap_iou))
        self._states: list[_SportsMixState] = []
        self._next_id = 1

    def update(self, detections: list[Any], frame_idx: int) -> list[SportsMixOutput]:
        active = [state for state in self._states if state.missed <= self.max_age]
        predicted = [self._predict_bbox(state, frame_idx) for state in active]
        cost = self._cost_matrix(active, predicted, detections)

        matches: list[tuple[int, int]] = []
        unmatched_states = set(range(len(active)))
        unmatched_detections = set(range(len(detections)))
        if cost.size:
            rows, cols = _linear_sum_assignment_with_forbidden_edges(cost)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if not np.isfinite(cost[row, col]) or cost[row, col] > self.match_threshold:
                    continue
                matches.append((row, col))
                unmatched_states.discard(row)
                unmatched_detections.discard(col)

        outputs: list[SportsMixOutput] = []
        for state_idx, detection_idx in matches:
            state = active[state_idx]
            detection = detections[detection_idx]
            observed_bbox = np.asarray(detection.bbox, dtype=np.float64)
            self._update_velocity(state, observed_bbox, frame_idx)
            state.bbox = observed_bbox
            state.last_frame = frame_idx
            state.missed = 0
            state.hits += 1
            state.confidence = float(detection.confidence)
            if float(getattr(detection, "overlap_iou", 0.0) or 0.0) < self.overlap_iou:
                state.embedding = self._update_embedding(state.embedding, getattr(detection, "embedding", []))
            outputs.append(
                SportsMixOutput(
                    track_id=state.track_id,
                    bbox=_bbox_tuple(state.bbox),
                    source_index=detection_idx,
                    observed=True,
                    confidence=state.confidence,
                    prediction_age=0,
                )
            )

        for state_idx in sorted(unmatched_states):
            state = active[state_idx]
            state.bbox = predicted[state_idx]
            state.last_frame = frame_idx
            state.missed += 1
            state.confidence = max(0.05, state.confidence * 0.90)
            if state.missed <= self.max_age:
                outputs.append(
                    SportsMixOutput(
                        track_id=state.track_id,
                        bbox=_bbox_tuple(state.bbox),
                        source_index=None,
                        observed=False,
                        confidence=state.confidence,
                        prediction_age=state.missed,
                    )
                )

        for detection_idx in sorted(unmatched_detections):
            detection = detections[detection_idx]
            embedding = _normalized_embedding(getattr(detection, "embedding", []))
            state = _SportsMixState(
                track_id=self._next_id,
                bbox=np.asarray(detection.bbox, dtype=np.float64),
                embedding=embedding,
                velocity=np.zeros(2, dtype=np.float64),
                confidence=float(detection.confidence),
                last_frame=frame_idx,
            )
            self._next_id += 1
            self._states.append(state)
            outputs.append(
                SportsMixOutput(
                    track_id=state.track_id,
                    bbox=_bbox_tuple(state.bbox),
                    source_index=detection_idx,
                    observed=True,
                    confidence=state.confidence,
                    prediction_age=0,
                )
            )

        self._states = [state for state in self._states if state.missed <= self.max_age]
        outputs.sort(key=lambda item: item.track_id)
        return outputs

    def _cost_matrix(
        self,
        states: list[_SportsMixState],
        predicted: list[np.ndarray],
        detections: list[Any],
    ) -> np.ndarray:
        cost = np.empty((len(states), len(detections)), dtype=np.float64)
        if not states or not detections:
            return cost
        for state_idx, state in enumerate(states):
            for detection_idx, detection in enumerate(detections):
                detection_bbox = np.asarray(detection.bbox, dtype=np.float64)
                iou_cost = 1.0 - bbox_iou(predicted[state_idx], detection_bbox)
                center_cost = normalized_center_distance(predicted[state_idx], detection_bbox)
                appearance_cost = embedding_distance(state.embedding, getattr(detection, "embedding", []))

                if self.max_center_distance > 0 and center_cost > self.max_center_distance:
                    cost[state_idx, detection_idx] = np.inf
                    continue

                overlap = float(getattr(detection, "overlap_iou", 0.0) or 0.0)
                appearance_reliability = max(0.0, 1.0 - overlap / max(self.overlap_iou, 1e-6))
                appearance_weight = self.appearance_weight * appearance_reliability
                motion_weight = self.motion_weight + self.appearance_weight - appearance_weight
                scaled_appearance = min(1.5, appearance_cost / self.appearance_scale)
                value = (
                    motion_weight * iou_cost
                    + appearance_weight * scaled_appearance
                    + self.center_weight * min(1.5, center_cost / max(self.max_center_distance, 1e-6))
                )

                # When boxes no longer overlap, require either plausible local motion
                # or useful appearance evidence before allowing a long jump.
                if iou_cost >= 0.999 and center_cost > 1.5 and scaled_appearance > 0.8:
                    value = np.inf
                cost[state_idx, detection_idx] = value
        return cost

    @staticmethod
    def _predict_bbox(state: _SportsMixState, frame_idx: int) -> np.ndarray:
        dt = max(1, int(frame_idx - state.last_frame))
        dx, dy = state.velocity * dt
        predicted = state.bbox.copy()
        predicted[[0, 2]] += dx
        predicted[[1, 3]] += dy
        return predicted

    @staticmethod
    def _update_velocity(state: _SportsMixState, bbox: np.ndarray, frame_idx: int) -> None:
        dt = max(1, int(frame_idx - state.last_frame))
        previous_center = bbox_center(state.bbox)
        observed_center = bbox_center(bbox)
        measured = (observed_center - previous_center) / dt
        state.velocity = 0.60 * state.velocity + 0.40 * measured

    def _update_embedding(self, current: np.ndarray | None, observed: Any) -> np.ndarray | None:
        vector = _normalized_embedding(observed)
        if vector is None:
            return current
        if current is None or current.shape != vector.shape:
            return vector
        mixed = self.template_momentum * current + (1.0 - self.template_momentum) * vector
        norm = float(np.linalg.norm(mixed))
        return mixed / norm if norm > 0 else vector


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(1e-9, area_a + area_b - intersection)


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.asarray(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0), dtype=np.float64)


def normalized_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    distance = float(np.linalg.norm(bbox_center(a) - bbox_center(b)))
    height = max(1.0, float((a[3] - a[1] + b[3] - b[1]) / 2.0))
    return distance / height


def embedding_distance(current: np.ndarray | None, observed: Any) -> float:
    vector = _normalized_embedding(observed)
    if current is None or vector is None or current.shape != vector.shape:
        return 1.0
    return float(np.linalg.norm(current - vector))


def _linear_sum_assignment_with_forbidden_edges(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run Hungarian assignment when motion gates have forbidden whole rows/columns.

    SciPy raises ``ValueError: cost matrix is infeasible`` when a rectangular
    assignment contains no finite edge for a required row or column.  SportsMix
    deliberately represents motion-gated pairs as infinity, and such isolated
    tracks/detections are normal after cuts, detector flicker, or rapid camera
    motion.  Give forbidden edges a finite sentinel for the optimization, then
    let the caller reject those pairs using the original matrix.
    """

    finite = cost[np.isfinite(cost)]
    largest_finite = float(np.max(finite)) if finite.size else 0.0
    forbidden_cost = max(1_000_000.0, largest_finite + 1_000.0)
    assignment_cost = np.where(np.isfinite(cost), cost, forbidden_cost)
    return linear_sum_assignment(assignment_cost)


def _normalized_embedding(value: Any) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def _bbox_tuple(value: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(item) for item in value.tolist())  # type: ignore[return-value]
