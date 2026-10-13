Script audit, 2026-10-12. Scope: the original seven experiments, plus PSPO ablations and the requested test-time, initialisation, environment-frame, learning-curve and gain-retained figures. Updated after the expanded user request. The active scripts are organised by main use: 28 in `scripts/experiments/` (including `lab_cluster/`) and 25 in `scripts/plots/`. Recommendation: retain 53 of 82 scripts; 29 scripts have been moved to `scripts/archive/`. All filenames below are relative to `../scripts/`. This is a static source/dependency audit, not a successful execution of the experiments. The 29 listed archive candidates were moved into `scripts/archive/` on 2026-10-12 at the user’s request. No results were moved and no training jobs were launched.

**Experiment map**

See the [launch command tables](launch_commands.md) for commands covering all
51 standalone experiment, plotting, analysis and cluster entry points.

| Experiment | Training/evaluation entry points | Analysis and important options |
| --- | --- | --- |
| PSPO and baselines on MASA | `experiments/run_pspo_one_env.sh` (or `experiments/launch_pspo_multi_env.py`); `experiments/run_seed_experiments.py`; `experiments/run_baselines_pspo_init_ablation.py --methods rl_sgf` for the safe-initialised RL-SGF cohort | Set `SAFE_REGION_SHAPE=segment` for PSPO-LS; line segment is the default. Explicitly set `METHOD_GROUPS=ppo,baselines_lag,cpo,shielded` for the standard baseline sweep. Use `plots/plot_masa_learning_curve_comparisons.py` and `plots/generate_masa_all_methods_table.py`. `experiments/run_extended_rl_baseline_sweeps.sh` handles Bridge v2/MiniPacman only; set `BRIDGE_CROSSING_V2_TIMESTEPS=1600000` for the paper cohort (wrapper default is 1000000). |
| PSPO and PPO on FrozenLake | `experiments/run_frozenlake_shaping_sweep.py --methods ppo pspo`; for the segment verify-first/frequency-1 PSPO cohort, use `experiments/run_frozenlake_segment_frequency_sweep.py --safety-frequency 1` and run matched PPO through the shaping sweep with `--methods ppo` | Select the same `--size` and `--total-timesteps`. Use `plots/plot_frozenlake_scalability.py`, `plots/plot_frozenlake_freq1_evaluation_curves.py`, and `plots/plot_frozenlake_ppo_pspo_reward_efficiency.py`. The joint shaping sweep's PSPO is not automatically the segment variant. These are shaped-training/raw-reward-evaluation workflows; unshaped PPO can also be run through `experiments/run_frozenlake_ppo_shaping.py --shaping-scale 0`. |
| Projected baselines | `experiments/run_safe_initialised_baseline_projection.py` | Safe initialisation, ordinary baseline training, then conditional final projection into a fixed certified orthotope. Include `--methods rl_sgf` explicitly when required. Requires the existing base-policy/LID inputs selected by `--lid-root`. Use `plots/generate_safe_projection_comparison_assets.py` and the MASA comparison plot. |
| PSPO versus PPO-Shield inference | `experiments/run_inference_time_comparison.py --pspo-variant segment --output-dir <new-directory> --launch` | Use `--pspo-variant orthotope` for the older PSPO cohort. Requires saved checkpoints. Writes aggregate reports and, for a complete segment cohort, latency tables. `plots/plot_pspo_ls_test_time_bars.py` plots reward/safety at deployment; it is not a latency benchmark. |
| Record training time for each method | `experiments/run_timed_main_comparison.py --output-dir <new-directory> --launch` | Fresh timings for PPO, PPO-Lagrangian, PPO-PID-Lagrangian, CPO, PPO-Shield, segment PSPO, and lookup PSPO; select RL-SGF with `--methods`. Each job stores timing JSON; the completed sweep exports `training_times.csv`. Select `--safe-region-shape orthotope` or `--verify-first` explicitly. `plots/generate_training_time_comparison.py` recovers historical timings; `plots/generate_pspo_variant_tables.py` reports PSPO variant results/timing. The dedicated runner does not cover projected-baseline variants or FrozenLake automatically. |
| PSPO with verify-first strategy | `experiments/run_pspo_verify_first_segment.py --run-name <new-name>` for preparation, followed by its `--supervise <manifest>` mode; alternatively `VERIFY_FIRST=true SAFE_REGION_SHAPE=segment bash .../run_pspo_one_env.sh` with environment/seed/CPU settings | The dedicated runner matches and reuses existing `segment_lid` configurations and initial actors. Use `SAFE_REGION_SHAPE=orthotope` for orthotope verify-first. FrozenLake uses `experiments/run_frozenlake_segment_frequency_sweep.py` or `experiments/run_frozenlake_segment_shaping.py --verify-first`. Analyse with `plots/plot_pspo_variant_reward_time_aamas.py` / `plots/generate_pspo_variant_tables.py`. |
| Empirical safety under stochastic action sampling | `experiments/run_temperature_sweep.py` | Includes `--variant pspo`, `--variant pspo_ls`, and baseline/shield-on/shield-off variants. Uses softmax(logits/T), with T=0 giving greedy execution. Use `plots/plot_temperature_sweep.py` and `plots/tabulate_temperature_sweep_safety.py`. RL-SGF is not currently in this evaluator's variant registry. |
| PSPO ablations: initialisation and safe updates | `experiments/run_pspo_four_ablations.py`; analyse with `plots/analyse_pspo_four_ablations.py` | Defaults to segment controls with `no_entropy`, `no_margin`, `region_first_instrumented` and `verify_first`. Select `--safe-region-shape orthotope` for `fixed_lid` and `no_gradient`; `ce_only` remains available. Select via `--variants`. Keep `experiments/run_policy_initialisation_ablations.sh` and `experiments/lab_cluster/pspo_initialisation.py` for cluster initialisation ablations. `plots/generate_policy_initialisation_tables.py` and `plots/generate_policy_initialisation_paper_assets.py` produce additional ablation assets. |
| PSPO-LS initialisation ablations | `experiments/run_pspo_segment_init_ablations.py --cpu-ids <pool>` | Supports `--variants no_entropy no_margin`. Use `--dry-run` to inspect the jobs. This dedicated launcher reproduces the retained segment initialisation cohort. |
| LID-objective/geometry diagnostic | `experiments/run_pspo_lid_geometry_comparison.py` | Retained conservatively under the expanded ablation scope: a small matched objective comparison with actual LID-slice plots, not the full multi-seed ablation suite. |
| Test-time empirical reward and safety: baselines, PSPO and projected baselines | `plots/plot_pspo_ls_test_time_bars.py` | Produces separate main and projected-baseline reward/safety figures, plus a projected reward-only figure. Uses region-first PSPO-LS labelled PSPO. First generate its input CSVs with `plots/plot_rl_sgf_final_policy_reward_safety_bars.py` and `plots/generate_safe_projection_comparison_assets.py`. |
| Initialisation loss terms' effects on action logits | `plots/plot_safe_policy_initialisation_logits.py` | Illustrates CE, minimum margin and safe-action entropy using fixed illustrative logits; this is a conceptual figure, not measured trained-policy logits. |
| Initial frames of MASA environments | `plots/render_masa_initial_frames.py` | Renders all six environments at reset seed 0 using saved canonical configurations; produces individual PNG/PDF frames and a combined row. |
| Evaluation-time and exploration-time learning curves | `plots/plot_masa_learning_curve_comparisons.py` | Covers main baselines/PSPO-LS, projected baselines and PSPO variants; emits combined and separate evaluation/exploration figures. Projected baselines have pre-projection training curves and terminal post-projection evaluation markers, not post-projection exploration curves. `plots/plot_extended_budget_learning_curves.py` remains the older orthotope comparison and an imported dependency. |
| Fraction of PSPO reward gain retained by ablations | `plots/plot_pspo_ablation_gain_retained.py --family orthotope` or `--family segment` | Plots reward gain relative to the safe initial policy, with paired stratified-bootstrap intervals. `plots/compute_pspo_ablation_reward_ratios.py` is retained for the separate shifted, seed-paired reward-ratio analysis; it does not produce the same statistic. |

The timing row has a coverage gap if “each method” means every method/variant in the expanded experiment list: there is no single existing script that implements that full timing matrix. Training-loop, stage and process times have different boundaries. The dedicated timing sweep suppresses final checkpoints, so it cannot supply models for the inference/stochastic-sampling studies.

Several launchers and plots read fixed historical cohort locations. New output names do not automatically retarget every report/evaluator. Preserve input shields, demonstrations, initial actors, fixed LIDs, checkpoints and configurations, and check the selected cohort paths before comparing new runs. This audit does not certify that all required ignored data are available in a fresh clone.

**Figure and ablation dependencies**

For the requested PSPO-LS test-time figures, run these generators in order from the repository root (existing training/evaluation results are required):

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/plots/plot_rl_sgf_final_policy_reward_safety_bars.py
.venv/bin/python projects/safe_policy_optimisation/scripts/plots/generate_safe_projection_comparison_assets.py
.venv/bin/python projects/safe_policy_optimisation/scripts/plots/plot_pspo_ls_test_time_bars.py
```

The first two write the default `--main-csv` and `--projected-csv` inputs consumed by the third. Its output stems are `main_test_time_reward_safety_pspo_ls` and `projected_baselines_test_time_reward_safety_pspo_ls` under `figures/pspo_ls_test_time/`. If using custom output directories, pass the matching CSV paths explicitly. The projected-figure code checks that all final policies have 100% measured safety; new cohorts that fail this check require adapting the figure/caption rather than silently reusing the historical claim.

The gain-retained plot reads saved metrics directly and has historical initial rewards in `INIT_REWARDS`. Its orthotope fixed-LID arm reads `runs/static_lid_ablation_masa_matched`, whereas the general ablation launcher writes its `fixed_lid` arm under `artifacts/ablation_studies/pspo_four_ablations`. A fresh suite therefore needs explicit cohort-path/initial-reward alignment before plotting; running the suite does not automatically retarget the historical figure. Its segment family reads `segment_lid` controls and `pspo_segment_init_ablations` results.

The general ablation runner now defaults to `artifacts/ablation_studies/pspo_segment_ablations`; pass `--safe-region-shape orthotope` to launch or analyse the historical orthotope protocol.

The initialisation paper-asset generators default to `artifacts/ablation_studies/pspo_four_ablations/analysis/policy_initialisation/results.json`, produced by the initialisation wrapper's `analyse` mode or by `plots/analyse_pspo_four_ablations.py` with matching variants/output directory. The cluster wrapper has a historical `SOURCE_ROOT`; set it to the intended source checkout when using it for new runs.

**Retain: experiment/figure entry points and reports (32)**

- [analyse_pspo_four_ablations.py](../scripts/plots/analyse_pspo_four_ablations.py)
- [compute_pspo_ablation_reward_ratios.py](../scripts/plots/compute_pspo_ablation_reward_ratios.py)
- [generate_masa_all_methods_table.py](../scripts/plots/generate_masa_all_methods_table.py)
- [generate_policy_initialisation_paper_assets.py](../scripts/plots/generate_policy_initialisation_paper_assets.py)
- [generate_policy_initialisation_tables.py](../scripts/plots/generate_policy_initialisation_tables.py)
- [generate_pspo_variant_tables.py](../scripts/plots/generate_pspo_variant_tables.py)
- [generate_safe_projection_comparison_assets.py](../scripts/plots/generate_safe_projection_comparison_assets.py)
- [generate_training_time_comparison.py](../scripts/plots/generate_training_time_comparison.py)
- [launch_pspo_multi_env.py](../scripts/experiments/launch_pspo_multi_env.py)
- [plot_frozenlake_scalability.py](../scripts/plots/plot_frozenlake_scalability.py)
- [plot_masa_learning_curve_comparisons.py](../scripts/plots/plot_masa_learning_curve_comparisons.py)
- [plot_pspo_ablation_gain_retained.py](../scripts/plots/plot_pspo_ablation_gain_retained.py)
- [plot_pspo_ls_test_time_bars.py](../scripts/plots/plot_pspo_ls_test_time_bars.py)
- [plot_safe_policy_initialisation_logits.py](../scripts/plots/plot_safe_policy_initialisation_logits.py)
- [plot_temperature_sweep.py](../scripts/plots/plot_temperature_sweep.py)
- [render_masa_initial_frames.py](../scripts/plots/render_masa_initial_frames.py)
- [run_baselines_pspo_init_ablation.py](../scripts/experiments/run_baselines_pspo_init_ablation.py)
- [run_extended_rl_baseline_sweeps.sh](../scripts/experiments/run_extended_rl_baseline_sweeps.sh)
- [run_frozenlake_segment_frequency_sweep.py](../scripts/experiments/run_frozenlake_segment_frequency_sweep.py)
- [run_frozenlake_shaping_sweep.py](../scripts/experiments/run_frozenlake_shaping_sweep.py)
- [run_inference_time_comparison.py](../scripts/experiments/run_inference_time_comparison.py)
- [run_policy_initialisation_ablations.sh](../scripts/experiments/run_policy_initialisation_ablations.sh)
- [run_pspo_four_ablations.py](../scripts/experiments/run_pspo_four_ablations.py)
- [run_pspo_lid_geometry_comparison.py](../scripts/experiments/run_pspo_lid_geometry_comparison.py)
- [run_pspo_one_env.sh](../scripts/experiments/run_pspo_one_env.sh)
- [run_pspo_segment_init_ablations.py](../scripts/experiments/run_pspo_segment_init_ablations.py)
- [run_pspo_verify_first_segment.py](../scripts/experiments/run_pspo_verify_first_segment.py)
- [run_safe_initialised_baseline_projection.py](../scripts/experiments/run_safe_initialised_baseline_projection.py)
- [run_seed_experiments.py](../scripts/experiments/run_seed_experiments.py)
- [run_temperature_sweep.py](../scripts/experiments/run_temperature_sweep.py)
- [run_timed_main_comparison.py](../scripts/experiments/run_timed_main_comparison.py)
- [tabulate_temperature_sweep_safety.py](../scripts/plots/tabulate_temperature_sweep_safety.py)

**Retain: imported support and prerequisite generators (15)**

These files supply imported helpers, environment/method registries, or input CSVs used by the mapped workflows. A plot/report filename does not make a file safe to remove.

- [_aamas_reward_safety.py](../scripts/plots/_aamas_reward_safety.py)
- [_paper_style.py](../scripts/plots/_paper_style.py)
- [compare_pspo_initial_final_rewards.py](../scripts/experiments/compare_pspo_initial_final_rewards.py)
- [generate_final_policy_table.py](../scripts/plots/generate_final_policy_table.py)
- [plot_extended_budget_learning_curves.py](../scripts/plots/plot_extended_budget_learning_curves.py)
- [plot_frozenlake_freq1_evaluation_curves.py](../scripts/plots/plot_frozenlake_freq1_evaluation_curves.py)
- [plot_frozenlake_ppo_pspo_reward_efficiency.py](../scripts/plots/plot_frozenlake_ppo_pspo_reward_efficiency.py)
- [plot_pspo_variant_reward_time_aamas.py](../scripts/plots/plot_pspo_variant_reward_time_aamas.py)
- [plot_rl_sgf_final_policy_reward_safety_bars.py](../scripts/plots/plot_rl_sgf_final_policy_reward_safety_bars.py)
- [run_frozenlake_ppo_shaping.py](../scripts/experiments/run_frozenlake_ppo_shaping.py)
- [run_frozenlake_pspo_shaping.py](../scripts/experiments/run_frozenlake_pspo_shaping.py)
- [run_frozenlake_safe_baseline_shaping.py](../scripts/experiments/run_frozenlake_safe_baseline_shaping.py)
- [run_frozenlake_segment_shaping.py](../scripts/experiments/run_frozenlake_segment_shaping.py)
- [run_stochastic_frozenlake_baselines.py](../scripts/experiments/run_stochastic_frozenlake_baselines.py)
- [run_stochastic_frozenlake_pspo.py](../scripts/experiments/run_stochastic_frozenlake_pspo.py)

**Retain: cluster support and existing tests (6)**

The cluster tools support stochastic sampling and initialisation ablations. Existing tests also import `plots/plot_final_policy_reward_safety_bars.py`.

- [lab_cluster/dispatch.py](../scripts/experiments/lab_cluster/dispatch.py)
- [lab_cluster/dispatch_temperature_sweep.py](../scripts/experiments/lab_cluster/dispatch_temperature_sweep.py)
- [lab_cluster/probe_hosts.sh](../scripts/experiments/lab_cluster/probe_hosts.sh)
- [lab_cluster/pspo_initialisation.py](../scripts/experiments/lab_cluster/pspo_initialisation.py)
- [lab_cluster/status.py](../scripts/experiments/lab_cluster/status.py)
- [plot_final_policy_reward_safety_bars.py](../scripts/plots/plot_final_policy_reward_safety_bars.py)

**Archived scripts for the expanded scope (29)**

These 29 scripts are now in `scripts/archive/`, outside the mapped experiment, ablation, figure and dependency set. Their contents are unchanged. Historical documentation may still name their original locations. Scripts that derive repository paths from their location or import active sibling modules may need path repairs or restoration to `scripts/` before execution. Experiment data and checkpoints remain in place.

- [analyse_unsafe_proposed_actions.py](../scripts/archive/analyse_unsafe_proposed_actions.py)
- [collect_masa_results.py](../scripts/archive/collect_masa_results.py)
- [collect_two_hidden_safe_rl_baselines.py](../scripts/archive/collect_two_hidden_safe_rl_baselines.py)
- [compare_extended_budget_pspo_vs_baselines.py](../scripts/archive/compare_extended_budget_pspo_vs_baselines.py)
- [finalise_frozenlake64_short_comparison.py](../scripts/archive/finalise_frozenlake64_short_comparison.py)
- [generate_environment_description_table.py](../scripts/archive/generate_environment_description_table.py)
- [launch_two_hidden_ppo_reference.py](../scripts/archive/launch_two_hidden_ppo_reference.py)
- [plot_extended_budget_comparison.py](../scripts/archive/plot_extended_budget_comparison.py)
- [plot_final_metrics_with_learning_curves.py](../scripts/archive/plot_final_metrics_with_learning_curves.py)
- [plot_frozenlake_freq1_reward_safety_curves.py](../scripts/archive/plot_frozenlake_freq1_reward_safety_curves.py)
- [plot_frozenlake_shaping_learning_curves.py](../scripts/archive/plot_frozenlake_shaping_learning_curves.py)
- [plot_interval_propagation_pspo.py](../scripts/archive/plot_interval_propagation_pspo.py)
- [plot_minipacman_pspo_reward_safety_tradeoff.py](../scripts/archive/plot_minipacman_pspo_reward_safety_tradeoff.py)
- [plot_pspo_lid_update_mechanism.py](../scripts/archive/plot_pspo_lid_update_mechanism.py)
- [plot_rl_sgf_learning_curves.py](../scripts/archive/plot_rl_sgf_learning_curves.py)
- [plot_rl_sgf_reward_safety_bars.py](../scripts/archive/plot_rl_sgf_reward_safety_bars.py)
- [plot_safe_parameter_region_3d.py](../scripts/archive/plot_safe_parameter_region_3d.py)
- [plot_shield_action_map.py](../scripts/archive/plot_shield_action_map.py)
- [plot_shield_removal_comparison.py](../scripts/archive/plot_shield_removal_comparison.py)
- [plot_shield_removal_comparison_aamas.py](../scripts/archive/plot_shield_removal_comparison_aamas.py)
- [plot_shield_removal_learning_curves.py](../scripts/archive/plot_shield_removal_learning_curves.py)
- [render_frozenlake_initial_frames.py](../scripts/archive/render_frozenlake_initial_frames.py)
- [render_stochastic_frozenlake_initial_frame.py](../scripts/archive/render_stochastic_frozenlake_initial_frame.py)
- [report_frozenlake_shaping_comparison.py](../scripts/archive/report_frozenlake_shaping_comparison.py)
- [run_frozenlake_initialisation_gated_sweep.py](../scripts/archive/run_frozenlake_initialisation_gated_sweep.py)
- [run_frozenlake_ppo_sweep.py](../scripts/archive/run_frozenlake_ppo_sweep.py)
- [run_parallel_pipelines.py](../scripts/archive/run_parallel_pipelines.py)
- [run_pspo_seed_experiments.py](../scripts/archive/run_pspo_seed_experiments.py)
- [run_seed_sweep.py](../scripts/archive/run_seed_sweep.py)
