#!/usr/bin/env python
"""Run a tiny matched PSPO objective comparison and plot actual LID slices.

The two runs use the same base policy, shield, PPO seed, rollout budget, and
LID settings.  A small subclass records the phase-end PPO proposal before the
LID objective acts, together with the selected certified region afterwards.
This makes it possible to check that any geometric difference is attributable
to the LID objective rather than a different proposed policy update.
"""

from __future__ import annotations

# Support direct execution as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from provably_safe_policy_optimisation.regions import OrthotopeRegion, ZonotopeRegion, flatten_tensors
from projects.safe_policy_optimisation.utils.pspo_modes import add_pspo_mode_args

from projects.safe_policy_optimisation.stages import train_pspo

REPO = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/inputs/colour_bomb"
)
OBJECTIVES = ("weighted_width", "projection_distance")
COLORS = {"weighted_width": "#d55e00", "projection_distance": "#0072b2"}


def _cpu_clones(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().cpu().clone() for tensor in tensors]


class GeometryCaptureMixin:
    """Capture certified regions and actual deployed endpoints without changing updates."""

    def _record_geometry(self, anchor, proposal, region=None) -> None:
        named = {id(param): name for name, param in self.policy.named_parameters()}
        deployed = _cpu_clones(self._live_actor_params)
        capture = {
            "anchor": anchor, "proposal": proposal, "deployed": deployed,
            "actor_param_names": [named.get(id(param), f"actor_tensor_{i}")
                                  for i, param in enumerate(self._live_actor_params)],
            "objective": self._rashomon_objective,
            "lid_iterations": int(self._last_rashomon_iterations_run),
            "target_contained_and_certified": bool(self._last_rashomon_target_contained_and_certified),
            "selected_checkpoint_index": self._last_selected_checkpoint_index,
            "safe_region_shape": self._safe_region_shape,
            "region_computed": region is not None,
        }
        if isinstance(region, OrthotopeRegion):
            capture.update(lower=_cpu_clones(region.lower), upper=_cpu_clones(region.upper))
        elif isinstance(region, ZonotopeRegion):
            if region.generators.shape[0] != 1:
                raise ValueError("Geometry capture requires a rank-one segment")
            center = flatten_tensors(region.center_params).cpu()
            generator = region.generators[0].detach().cpu()
            ends = [center + float(z) * generator
                    for z in (region.coefficient_l[0], region.coefficient_u[0])]
            capture.update(segment_endpoints=[ends[0], ends[1]],
                           lower=[torch.minimum(*ends)], upper=[torch.maximum(*ends)])
        else:
            # A verify-first acceptance certifies only the endpoint, not the path.
            capture.update(lower=deployed, upper=deployed)
        captures = getattr(self, "_lid_geometry_captures", [])
        captures.append(capture)
        self._lid_geometry_captures = captures

    def save(self, path: str | Path, *args: Any, **kwargs: Any) -> None:
        torch.save({"objective": self._rashomon_objective,
                    "captures": getattr(self, "_lid_geometry_captures", []),
                    "adaptive_diagnostics": self.adaptive_diagnostics()},
                   Path(path).parent / "lid_geometry.pt")
        super().save(path, *args, **kwargs)


class GeometryCapturePSPO(GeometryCaptureMixin, train_pspo.AdaptiveSafePPOV2):
    def _on_train_phase_end(self) -> None:
        self._ensure_adaptive_state()
        anchor = _cpu_clones(self._last_safe_params)
        proposal = _cpu_clones(self._phase_candidate_params())
        super()._on_train_phase_end()
        region = self._active_regions[-1] if self._active_regions else None
        self._record_geometry(anchor, proposal, region)


class GeometryCaptureVerifyFirstPSPO(GeometryCaptureMixin, train_pspo.AdaptiveSafePPO):
    def _compute_rashomon_around_last_safe(self, **kwargs):
        result = super()._compute_rashomon_around_last_safe(**kwargs)
        if result is not None:
            self._geometry_region = result[0]
        return result

    def _accept_or_project_candidate(self) -> None:
        anchor = _cpu_clones(self._last_safe_params)
        proposal = _cpu_clones(self._live_actor_params)
        self._geometry_region = None
        super()._accept_or_project_candidate()
        self._record_geometry(anchor, proposal, self._geometry_region)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-policy-path",
        type=Path,
        default=DEFAULT_INPUT / "rashomon_fullcov/base_policy.pt",
    )
    parser.add_argument(
        "--shield-path",
        type=Path,
        default=DEFAULT_INPUT / "shield_q.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--lid-iters", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=5e-2,
        help="Small entropy term used to ensure the solved tiny task produces a nonzero actor proposal.",
    )
    add_pspo_mode_args(parser)
    args = parser.parse_args()
    args.output_dir = args.output_dir or REPO / f"outputs/pspo_lid_geometry_{args.safe_region_shape}_{args.verify_first}"
    return args


def _matched_pspo_argv(args: argparse.Namespace, objective: str) -> list[str]:
    return [
        "--base-policy-path",
        str(args.base_policy_path),
        "--shield-path",
        str(args.shield_path),
        "--env-id",
        "CustomColourBombGridWorld-v0",
        "--state-representation",
        "one_hot",
        "--output-dir",
        str(args.output_dir / "runs"),
        "--run-id",
        objective,
        "--seed",
        str(args.seed),
        "--total-timesteps",
        str(args.timesteps),
        "--max-episode-steps",
        str(args.timesteps),
        "--eval-episodes",
        "1",
        "--curve-eval-freq",
        "0",
        "--early-stop-eval-freq",
        "0",
        "--freq",
        "rollout",
        "--verify-first",
        str(args.verify_first).lower(),
        "--safe-region-shape", args.safe_region_shape,
        "--segment-tolerance", str(args.segment_tolerance),
        "--segment-splits", str(args.segment_splits),
        "--segment-max-splits", str(args.segment_max_splits),
        "--directional",
        "true",
        "--region-mode",
        "replace",
        "--n-iters",
        str(args.lid_iters),
        "--rashomon-checkpoint",
        "1",
        "--rashomon-batch-size",
        "auto",
        "--certificate-samples",
        "74",
        "--rashomon-inverse-temp",
        "1",
        "--rashomon-multi-label-mode",
        "any",
        "--surrogate",
        "auto",
        "--rashomon-objective",
        objective,
        "--n-steps",
        str(args.timesteps),
        "--batch-size",
        str(args.timesteps),
        "--n-epochs",
        "1",
        "--learning-rate",
        str(args.learning_rate),
        "--ent-coef",
        str(args.entropy_coefficient),
        "--device",
        "cpu",
    ]


def run_experiments(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_v2, original_v1 = train_pspo.AdaptiveSafePPOV2, train_pspo.AdaptiveSafePPO
    train_pspo.AdaptiveSafePPOV2 = GeometryCapturePSPO
    train_pspo.AdaptiveSafePPO = GeometryCaptureVerifyFirstPSPO
    try:
        for objective in OBJECTIVES:
            print(f"Running {objective}: {args.safe_region_shape}, verify_first={args.verify_first}")
            run_args = train_pspo.parse_args(_matched_pspo_argv(args, objective))
            train_pspo.run(run_args)
    finally:
        train_pspo.AdaptiveSafePPOV2, train_pspo.AdaptiveSafePPO = original_v2, original_v1


def _flatten(tensors: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [tensor.detach().cpu().numpy().reshape(-1) for tensor in tensors]
    )


def _load_capture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    captures = payload.get("captures", [])
    complete = [
        capture for capture in captures if "lower" in capture and "upper" in capture
    ]
    if not complete:
        raise RuntimeError(f"No complete geometry capture found in {path}.")
    return payload, complete[-1]


def _coordinate_labels(capture: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for name, tensor in zip(capture["actor_param_names"], capture["anchor"]):
        labels.extend(f"{name}[{index}]" for index in range(tensor.numel()))
    return labels


def _box_faces(
    lower: np.ndarray, upper: np.ndarray
) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _draw_slice(
    ax: plt.Axes,
    captures: dict[str, dict[str, Any]],
    indices: np.ndarray,
    title: str,
) -> None:
    reference = captures[OBJECTIVES[0]]
    anchor = _flatten(reference["anchor"])
    proposal = _flatten(reference["proposal"])
    for objective in OBJECTIVES:
        capture = captures[objective]
        lower = _flatten(capture["lower"])[indices] - anchor[indices]
        upper = _flatten(capture["upper"])[indices] - anchor[indices]
        if "segment_endpoints" in capture:
            endpoints = np.vstack([point.numpy()[indices] - anchor[indices]
                                   for point in capture["segment_endpoints"]])
            ax.plot(*endpoints.T, color=COLORS[objective], linewidth=2.0,
                    label=objective.replace("_", " "))
        elif capture.get("region_computed", True):
            poly = Poly3DCollection(_box_faces(lower, upper), facecolor=COLORS[objective],
                                    edgecolor=COLORS[objective], linewidths=1.6, alpha=0.18,
                                    label=objective.replace("_", " "))
            ax.add_collection3d(poly)
        else:
            endpoint = _flatten(capture["deployed"])[indices] - anchor[indices]
            ax.scatter(*endpoint, color=COLORS[objective], marker="s", s=35,
                       label="verified endpoint")


    proposed_delta = proposal[indices] - anchor[indices]
    ax.scatter(
        0.0, 0.0, 0.0, color="black", marker="o", s=45, label="shared safe anchor"
    )
    ax.scatter(
        *proposed_delta,
        color="#009e73",
        marker="*",
        edgecolor="black",
        linewidth=0.5,
        s=130,
        label="shared PPO proposal",
    )
    values = [np.zeros(3), proposed_delta]
    for objective in OBJECTIVES:
        values.extend(
            [
                _flatten(captures[objective]["lower"])[indices] - anchor[indices],
                _flatten(captures[objective]["upper"])[indices] - anchor[indices],
            ]
        )
    points = np.vstack(values)
    span = np.maximum(np.ptp(points, axis=0), 1e-8)
    lo = points.min(axis=0) - 0.15 * span
    hi = points.max(axis=0) + 0.15 * span
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel(rf"$\Delta\theta_{{{indices[0]}}}$")
    ax.set_ylabel(rf"$\Delta\theta_{{{indices[1]}}}$")
    ax.set_zlabel(rf"$\Delta\theta_{{{indices[2]}}}$")
    ax.set_title(title)
    ax.view_init(elev=23, azim=-54)
    ax.grid(True, alpha=0.25)


def analyse_and_plot(args: argparse.Namespace) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    captures: dict[str, dict[str, Any]] = {}
    for objective in OBJECTIVES:
        payload, capture = _load_capture(
            args.output_dir / "runs" / objective / "lid_geometry.pt"
        )
        payloads[objective] = payload
        captures[objective] = capture

    anchor = _flatten(captures[OBJECTIVES[0]]["anchor"])
    proposal = _flatten(captures[OBJECTIVES[0]]["proposal"])
    proposal_delta = proposal - anchor
    labels = _coordinate_labels(captures[OBJECTIVES[0]])
    anchor_difference = float(
        np.max(np.abs(anchor - _flatten(captures[OBJECTIVES[1]]["anchor"])))
    )
    proposal_difference = float(
        np.max(np.abs(proposal - _flatten(captures[OBJECTIVES[1]]["proposal"])))
    )
    if anchor_difference > 1e-7 or proposal_difference > 1e-7:
        raise RuntimeError(
            "The comparison is not matched: max anchor/proposal differences are "
            f"{anchor_difference:.3e}/{proposal_difference:.3e}."
        )

    widths = {
        objective: _flatten(capture["upper"]) - _flatten(capture["lower"])
        for objective, capture in captures.items()
    }
    top_proposal = np.argsort(np.abs(proposal_delta))[-3:][::-1]
    width_difference = np.abs(widths[OBJECTIVES[0]] - widths[OBJECTIVES[1]])
    nonzero_update = np.abs(proposal_delta) > 0.0
    difference_scores = np.where(nonzero_update, width_difference, -1.0)
    top_difference = np.argsort(difference_scores)[-3:][::-1]

    metrics: dict[str, Any] = {}
    coordinate_rows: list[dict[str, Any]] = []
    for objective, capture in captures.items():
        lower = _flatten(capture["lower"])
        upper = _flatten(capture["upper"])
        below = np.maximum(lower - proposal, 0.0)
        above = np.maximum(proposal - upper, 0.0)
        distance = (float(np.linalg.norm(proposal - _flatten(capture["deployed"])))
                    if capture.get("safe_region_shape") == "segment" or not capture.get("region_computed", True)
                    else float(np.sqrt(np.sum(below**2 + above**2))))
        direction_extent = np.where(
            proposal_delta > 0.0,
            upper - anchor,
            np.where(proposal_delta < 0.0, anchor - lower, 0.0),
        )
        desired_extent = np.abs(proposal_delta)
        active = desired_extent > 0.0
        overgrowth = np.maximum(direction_extent - desired_extent, 0.0)
        shortfall = np.maximum(desired_extent - direction_extent, 0.0)
        metrics[objective] = {
            "lid_iterations": int(capture["lid_iterations"]),
            "selected_checkpoint_index": capture["selected_checkpoint_index"],
            "target_contained_and_certified": bool(
                capture["target_contained_and_certified"]
            ),
            "proposal_contained_by_selected_lid": bool(
                np.all(lower <= proposal) and np.all(proposal <= upper)
            ),
            "total_width_l1": float(widths[objective].sum()),
            "mean_width": float(widths[objective].mean()),
            "max_width": float(widths[objective].max()),
            "proposal_to_lid_l2": distance,
            "total_directional_overgrowth": float(overgrowth.sum()),
            "max_directional_overgrowth": float(overgrowth.max()),
            "total_directional_shortfall": float(shortfall.sum()),
            "covered_update_coordinate_fraction": float(
                np.mean(direction_extent[active] >= desired_extent[active])
            )
            if bool(active.any())
            else None,
        }
        for index in sorted(set(top_proposal.tolist() + top_difference.tolist())):
            coordinate_rows.append(
                {
                    "objective": objective,
                    "global_index": int(index),
                    "parameter": labels[index],
                    "anchor": float(anchor[index]),
                    "proposal": float(proposal[index]),
                    "proposal_delta": float(proposal_delta[index]),
                    "lower": float(lower[index]),
                    "upper": float(upper[index]),
                    "width": float(widths[objective][index]),
                }
            )

    summary = {
        "intent": "Tiny matched geometry comparison; not a performance experiment.",
        "safe_region_shape": args.safe_region_shape,
        "verify_first": args.verify_first,
        "objective_note": "Orthotope growth objectives do not affect the segment alpha search." if args.safe_region_shape == "segment" else "Matched orthotope growth objectives.",
        "environment": "CustomColourBombGridWorld-v0",
        "seed": args.seed,
        "training_timesteps": args.timesteps,
        "ppo_n_steps": args.timesteps,
        "ppo_n_epochs": 1,
        "ppo_learning_rate": args.learning_rate,
        "ppo_entropy_coefficient": args.entropy_coefficient,
        "lid_iteration_budget": args.lid_iters,
        "certificate_states": 74,
        "actor_parameter_count": int(anchor.size),
        "max_shared_anchor_difference": anchor_difference,
        "max_shared_proposal_difference": proposal_difference,
        "shared_proposal_l2": float(np.linalg.norm(proposal_delta)),
        "top_proposal_coordinates": [
            {"global_index": int(index), "parameter": labels[index]}
            for index in top_proposal
        ],
        "top_width_difference_coordinates": [
            {"global_index": int(index), "parameter": labels[index]}
            for index in top_difference
        ],
        "objectives": metrics,
    }

    fig = plt.figure(figsize=(13.2, 5.8))
    _draw_slice(
        fig.add_subplot(121, projection="3d"),
        captures,
        top_proposal,
        "Largest shared PPO-update coordinates",
    )
    _draw_slice(
        fig.add_subplot(122, projection="3d"),
        captures,
        top_difference,
        "Coordinates with largest LID-width difference",
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color=COLORS[o],
            linestyle="",
            label=o.replace("_", " "),
        )
        for o in OBJECTIVES
    ]
    handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="",
                label="shared safe anchor",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                markeredgecolor="black",
                color="#009e73",
                linestyle="",
                markersize=12,
                label="shared PPO proposal",
            ),
        ]
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Tiny matched PSPO experiment: certified LID projections",
        y=0.98,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        "Coordinates are displacements from the shared safe base policy. Lines show certified segments, boxes show orthotope projections, and isolated points show verify-first accepted endpoints. Segment coordinate bounds are not a certified box.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"lid_geometry_comparison.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)

    with (args.output_dir / "geometry_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (args.output_dir / "selected_coordinate_bounds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coordinate_rows[0]))
        writer.writeheader()
        writer.writerows(coordinate_rows)
    return summary


def main() -> int:
    torch.set_num_threads(1)
    args = parse_args()
    run_experiments(args)
    summary = analyse_and_plot(args)
    print(json.dumps(summary, indent=2))
    print(f"Wrote comparison artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
