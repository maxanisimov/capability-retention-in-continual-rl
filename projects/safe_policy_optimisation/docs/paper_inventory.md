# PSPO paper inventory

This inventory records the retained paper workflow after the 2026-10-09 cleanup.
Paths below are relative to `projects/safe_policy_optimisation/` unless labelled
repository-root. `RUNS` means `artifacts/paper_2503_07671/runs/`.
[`paper_inventory.json`](paper_inventory.json) lists every retained run directory
and why it remains. [`archive/manifest.json`](../archive/manifest.json) records
relocations; historical cohorts were moved intact, not renamed as newer studies.

## Studies and reproduction entry points

| Study | Training / evaluation entry points in `scripts/` | Analysis / assets in `scripts/` | Retained outputs |
| --- | --- | --- | --- |
| MASA main comparison: PPO, Lagrangian/PID, CPO, RL-SGF, PPO-Shield, PSPO-LS | `run_extended_rl_baseline_sweeps.sh`, `run_pspo_one_env.sh`, `run_baselines_pspo_init_ablation.py` | `plot_masa_learning_curve_comparisons.py`, `generate_masa_all_methods_table.py` | `figures/masa_learning_curve_comparisons/`, `figures/masa_all_methods_table/` |
| Final projection from safe initial policies | `run_safe_initialised_baseline_projection.py` | `generate_safe_projection_comparison_assets.py`, main comparison (b) | `figures/pspo_vs_safe_projection/` |
| Four PSPO variants | `run_pspo_one_env.sh`, `run_pspo_lid_geometry_comparison.py`, `run_pspo_verify_first_segment.py` | main comparison (c), `plot_pspo_variant_reward_time_aamas.py`, `generate_pspo_variant_tables.py` | `results/pspo/variant_reward_time_aamas/` and MASA comparison figures |
| Initialisation, update and fixed-LID ablations | `run_pspo_four_ablations.py`, `run_pspo_segment_init_ablations.py`, `run_policy_initialisation_ablations.sh` | `analyse_pspo_four_ablations.py`, `compute_pspo_ablation_reward_ratios.py`, `plot_pspo_ablation_gain_retained.py`, `generate_policy_initialisation_tables.py`, `generate_policy_initialisation_paper_assets.py` | `figures/aamas/pspo*_ablation_gain_retained.*`, `figures/pspo_policy_initialisation/`, `results/pspo/ablation_reward_ratios/` |
| Training and inference timing | `run_timed_main_comparison.py`, `run_inference_time_comparison.py` | `generate_training_time_comparison.py`, `generate_pspo_variant_tables.py`, `plot_pspo_ls_test_time_bars.py` | `results/training_time_main_baselines/`, `figures/pspo_ls_test_time/`, `figures/pspo_ls_inference_latency/` |
| Temperature / stochastic evaluation | `run_temperature_sweep.py` | `plot_temperature_sweep.py`, `tabulate_temperature_sweep_safety.py` | `artifacts/paper_2503_07671/analysis/temperature_sweep/`, `results/pspo/temperature_sweep/` |
| Safety-only FrozenLake and scalability | `run_stochastic_frozenlake_pspo.py`, `run_frozenlake_shaping_sweep.py`, `run_frozenlake_segment_frequency_sweep.py` | `plot_frozenlake_scalability.py`, `plot_frozenlake_freq1_evaluation_curves.py`, `plot_frozenlake_ppo_pspo_reward_efficiency.py` | `figures/frozenlake_scalability_20261008/` and retained `frozenlake*` comparison folders |
| Orthotope controls, extended budgets and shield removal | `run_extended_rl_baseline_sweeps.sh`, `run_pspo_one_env.sh` | `compare_extended_budget_pspo_vs_baselines.py`, `generate_final_policy_table.py`, `plot_rl_sgf_final_policy_reward_safety_bars.py`, `plot_shield_removal_comparison_aamas.py` and corresponding learning-curve scripts | `docs/extended_budget_comparison/`, `docs/two_hidden_safe_rl_baselines/`, `figures/aamas/`, `figures/pspo_vs_rl_baselines_learning_curves_all_envs/` |
| Method / environment illustrations | `notebooks/pspo_safe_update_stages.ipynb`, `render_masa_initial_frames.py`, `render_frozenlake_initial_frames.py` | `plot_interval_propagation_pspo.py`, `plot_pspo_lid_update_mechanism.py`, `plot_safe_parameter_region_3d.py`, `plot_safe_policy_initialisation_logits.py`, `plot_shield_action_map.py`, `generate_environment_description_table.py` | Retained mechanism/shield/initial-frame figures and `results/environments/` |

The main MASA comparison uses `segment_lid/two_hidden` for PSPO. Older final
bar/extended-budget generators use the orthotope controls from
`plot_extended_budget_learning_curves.py`; they remain supporting results and
must not be interpreted as the same cohort. Projected baselines have ordinary
training curves plus a separate terminal post-projection evaluation. RL-SGF's
irregular updates use past-observation alignment in the main comparison.

## Exact MASA input roots

All six tasks use the retained shields and demonstrations under
`artifacts/paper_2503_07671/inputs/{media_streaming,colour_bomb,colour_bomb_v2,bridge_crossing,bridge_crossing_v2,mini_pacman}`.

| Inputs | Location |
| --- | --- |
| PSPO-LS | `RUNS/segment_lid/two_hidden/<environment>/seed<k>` |
| Orthotope PSPO, media/Colour Bomb | `RUNS/pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default/two_hidden/<environment>` |
| Orthotope PSPO, Bridge v1 | `RUNS/pspo_adaptive_bridge_v1_safe_entropy_w1_min0p95_freq1/two_hidden/bridge_crossing` |
| Orthotope PSPO, Bridge v2 | `RUNS/pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base/two_hidden/bridge_crossing_v2` |
| Orthotope PSPO, MiniPacman | `RUNS/pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base/two_hidden/mini_pacman` |
| Verify-first orthotope / segment | `RUNS/pspo_verifyfirst_masa/two_hidden/<environment>` / `RUNS/pspo_verifyfirst_segment_masa_20261007T131200Z/two_hidden/<environment>` |
| RL-SGF safe initialisation / projection | `RUNS/rl_sgf_safe_initialised/<environment>/rl_sgf/seed<k>` / `RUNS/rl_sgf_safe_initialised_projection/<environment>/rl_sgf/seed<k>` |
| Other final-projected baselines | `RUNS/safe_initialised_baseline_projection/<environment>/<method>/seed<k>` |
| Shared safe actors and certified LIDs | `RUNS/static_lid_ablation_masa_matched/<environment>/` |
| Bridge v2 / MiniPacman baselines | `RUNS/rl_baselines_extended_budgets_1600k/bridge_crossing_v2` / `RUNS/rl_baselines_extended_budgets/mini_pacman` |
| Other four baseline tasks | **Repository-root** `outputs/_sweeps_2hidden_<environment>_baselines_only/<environment>/` |
| Initialisation/update ablations | **Repository-root** `artifacts/ablation_studies/pspo_four_ablations/` and `artifacts/ablation_studies/pspo_segment_init_ablations/` |

`masa_all_envs` is a result-file consolidation. It is not a replacement for
checkpoint-bearing source cohorts when re-evaluating policies. Several older
input runs remain because current configurations refer to their initial actors;
the JSON inventory labels these as metadata dependencies. Repository-root inputs
are outside this project's archive operation and remain in place.

Training timing uses `timing_main_lookup_20261006T224805Z`; inference uses
`inference_main_20261007T103000Z` and `inference_pspo_ls_20261008T225928Z`.
Temperature analysis retains its pre-slip-fix Colour Bomb v2 shield under
`inputs/colour_bomb_v2/_pre_slipfix_20261007/` to reproduce the trained dynamics.

## FrozenLake cohorts

The final scalability plot compares safety-only PSPO with PPO on 16, 32, 64 and
128 square layouts. Its exact run names are in `plot_frozenlake_scalability.py`
and `plot_frozenlake_freq1_evaluation_curves.py`, and listed below. The smaller
tasks use 204,800 steps; the larger two use 200,704 realised steps. Training
reward shaping is excluded from reported raw-reward evaluations.

Earlier shaped comparisons remain as supporting inputs to the reward-efficiency
and evaluation-curve scripts. Safety-only initialisation cohorts preserve the
actor/shield/layout inputs. Four additional output roots are protected because
FrozenLake workers/supervisors were using them during cleanup; their presence
does not assert that incomplete results appear in a final figure. Historical
goal-aware actor results and their evaluator are in the archive.

## Retained run directories

Each line gives a directory under `RUNS` and its retention evidence.

- `frozenlake128_shaping_ppo_pspo_t400k_20261008T085100Z`: scripts/plot_frozenlake_shaping_learning_curves.py; scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py.
- `frozenlake128_shaping_ppo_t200000_20261008T210643Z`: scripts/plot_frozenlake_scalability.py.
- `frozenlake128_shaping_pspo_segment_verify_first_freq1_t200000_20261008T173822Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `frozenlake128_shaping_pspo_segment_verify_first_t1024000_20261008T124400Z`: scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py.
- `frozenlake16_shaping_ppo_t204800_20261008T150953Z`: scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py.
- `frozenlake16_shaping_pspo_segment_verify_first_freq1_t204800_20261008T155100Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `frozenlake16_shaping_pspo_segment_verify_first_t204800_20261008T123000Z`: scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py.
- `frozenlake256_shaping_pspo_segment_verify_first_freq1_t200000_20261008T173822Z`: Running FrozenLake job (protected).
- `frozenlake256_shaping_pspo_segment_verify_first_t4096000_20261008T124400Z`: Running FrozenLake job (protected).
- `frozenlake32_shaping_ppo_t204800_20261008T151817Z`: scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py.
- `frozenlake32_shaping_pspo_segment_verify_first_freq1_t204800_20261008T160100Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `frozenlake32_shaping_pspo_segment_verify_first_t204800_20261008T123000Z`: scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py.
- `frozenlake64_shaping_ppo_t200000_20261008T172454Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `frozenlake64_shaping_pspo_segment_verify_first_freq1_t200000_20261008T172800Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `frozenlake64_shaping_pspo_segment_verify_first_freq1_t409600_20261008T160100Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `frozenlake64_shaping_pspo_segment_verify_first_t409600_20261008T124400Z`: Running FrozenLake job (protected).
- `frozenlake_pspo_segment_verify_first_scaling_20261008T124400Z`: Running FrozenLake job (protected).
- `inference_main_20261007T103000Z`: Retained inference-time comparison.
- `inference_pspo_ls_20261008T225928Z`: Retained inference-time comparison.
- `masa_all_envs`: scripts/collect_masa_results.py; Metadata dependency of static_lid_ablation_masa_matched.
- `ppo_shaping_frozenlake64_20261008T070700Z`: scripts/plot_frozenlake_freq1_evaluation_curves.py.
- `pspo_adaptive_bridge_v1_safe_entropy_w1_min0p95_freq1`: scripts/plot_pspo_variant_reward_time_aamas.py; scripts/plot_extended_budget_learning_curves.py; scripts/collect_masa_results.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of masa_all_envs; Metadata dependency of static_lid_ablation_masa_matched; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1`: Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of masa_all_envs; Metadata dependency of pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base; Metadata dependency of static_lid_ablation_masa_matched; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base`: scripts/plot_pspo_variant_reward_time_aamas.py; scripts/plot_extended_budget_learning_curves.py; scripts/compare_extended_budget_pspo_vs_baselines.py; scripts/collect_masa_results.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of masa_all_envs; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default`: scripts/plot_pspo_variant_reward_time_aamas.py; scripts/plot_extended_budget_learning_curves.py; scripts/collect_masa_results.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of masa_all_envs; Metadata dependency of static_lid_ablation_masa_matched; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100`: scripts/compare_extended_budget_pspo_vs_baselines.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of masa_all_envs; Metadata dependency of pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base; Metadata dependency of static_lid_ablation_masa_matched; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base`: scripts/plot_pspo_variant_reward_time_aamas.py; scripts/plot_extended_budget_learning_curves.py; scripts/compare_extended_budget_pspo_vs_baselines.py; scripts/collect_masa_results.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of masa_all_envs; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `pspo_adaptive_two_hidden_directional_replace_all_margin2_200iters`: Metadata dependency of pspo_reward_pilot_bcv2_m2_reuse_i200_f1_t400k.
- `pspo_reward_pilot_bcv2_m2_reuse_i200_f1_t400k`: scripts/compare_extended_budget_pspo_vs_baselines.py.
- `pspo_stochastic_frozenlake128_safety_only_state_id_lookup`: scripts/generate_environment_description_table.py; Safety-only FrozenLake actor, shield and layout inputs.
- `pspo_stochastic_frozenlake16_safety_only_state_id_lookup`: Safety-only FrozenLake actor, shield and layout inputs.
- `pspo_stochastic_frozenlake256_safety_only_state_id_lookup`: Safety-only FrozenLake actor, shield and layout inputs.
- `pspo_verifyfirst_masa`: scripts/plot_pspo_variant_reward_time_aamas.py.
- `pspo_verifyfirst_segment_masa_20261007T131200Z`: scripts/plot_pspo_variant_reward_time_aamas.py.
- `rl_baselines_extended_budgets`: scripts/plot_extended_budget_learning_curves.py; scripts/compare_extended_budget_pspo_vs_baselines.py; scripts/collect_masa_results.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of inference_pspo_ls_20261008T225928Z; Metadata dependency of masa_all_envs; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `rl_baselines_extended_budgets_1600k`: scripts/plot_extended_budget_learning_curves.py; scripts/collect_masa_results.py; Metadata dependency of inference_main_20261007T103000Z; Metadata dependency of inference_pspo_ls_20261008T225928Z; Metadata dependency of masa_all_envs; Metadata dependency of timing_main_lookup_20261006T224805Z.
- `rl_sgf_safe_initialised`: scripts/plot_rl_sgf_final_policy_reward_safety_bars.py; scripts/plot_masa_learning_curve_comparisons.py; scripts/plot_rl_sgf_learning_curves.py; scripts/plot_rl_sgf_reward_safety_bars.py.
- `rl_sgf_safe_initialised_projection`: scripts/plot_masa_learning_curve_comparisons.py; scripts/generate_safe_projection_comparison_assets.py.
- `safe_initialised_baseline_projection`: scripts/plot_masa_learning_curve_comparisons.py; scripts/generate_safe_projection_comparison_assets.py.
- `segment_lid`: scripts/plot_pspo_variant_reward_time_aamas.py; scripts/plot_pspo_ablation_gain_retained.py; scripts/plot_pspo_ls_test_time_bars.py; Metadata dependency of inference_pspo_ls_20261008T225928Z; Metadata dependency of pspo_verifyfirst_segment_masa_20261007T131200Z.
- `static_lid_ablation_masa_matched`: scripts/generate_safe_projection_comparison_assets.py; scripts/plot_pspo_ablation_gain_retained.py; Metadata dependency of rl_sgf_safe_initialised; Metadata dependency of rl_sgf_safe_initialised_projection; Metadata dependency of safe_initialised_baseline_projection.
- `timing_main_lookup_20261006T224805Z`: scripts/generate_pspo_variant_tables.py.

## Local data and archive policy

The two RL-SGF cohorts were imported from the existing `rl-sgf` worktree:
700 files (173,791,473 bytes) and 900 files (205,212,151 bytes), respectively.
Every copied file matched its source SHA-256. Source-worktree files and its
random-initialisation cohort were left in place; active analysis defaults use
this project's copies. The JSON inventory records copy-manifest digests.

Checkpoint/log/output files are ignored by Git in both active and archived trees.
A fresh clone includes source, settings and documentation, not the local runs.
Restore datasets from storage or rerun the documented studies before generating
paper results. The archive manifest records 2026-10-09 relocations; preserved
historical paths in saved metadata are not silently rewritten.
