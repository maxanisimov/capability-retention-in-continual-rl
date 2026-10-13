Run the following setup once in Bash. CPU IDs are examples: replace them with CPUs available to your job. Each experiment row is an independent launch; do not run the entire table as one batch.

```bash
cd /vol/bitbucket/ma5923/_projects/CertifiedContinualLearning
PY="$PWD/.venv/bin/python"
EXP="$PWD/projects/safe_policy_optimisation/scripts/experiments"
PLOT="$PWD/projects/safe_policy_optimisation/scripts/plots"
RUNS="$PWD/projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
CPUS="8-17"  # Example: replace with CPUs available to your job.
JOBS=10
SIZE=16
STEPS=204800
ABLA="$PWD/artifacts/ablation_studies/pspo_four_ablations_$TAG"
INIT_JSON="$ABLA/analysis/policy_initialisation/results.json"
TEMP="$RUNS/temperature_$TAG"
export PYTHONPATH="$PWD/core:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/pspo-revamp-matplotlib
```

**Main experiments**

| Experiment / output | Command | Scope / dependency |
| --- | --- | --- |
| MASA: PSPO-LS, six environments × ten seeds | `"$PY" "$EXP/launch_pspo_multi_env.py" --safe-region-shape segment --run-name "pspo_ls_$TAG"` | Requires 60 distinct idle CPU IDs; selects them automatically with mpstat. Use --cpu-ids only with exactly 60 IDs. |
| MASA: orthotope PSPO | `"$PY" "$EXP/launch_pspo_multi_env.py" --safe-region-shape orthotope --run-name "pspo_orthotope_$TAG"` | Six environments × ten seeds; 60 CPUs. |
| MASA: standard baselines, four default-budget environments | `ENVS=media_streaming,colour_bomb,colour_bomb_v2,bridge_crossing METHOD_GROUPS=ppo,baselines_lag,cpo,shielded SWEEP_PARALLEL="$JOBS" PAPER_OUT_BASE="$RUNS/masa_baselines_$TAG" "$PY" "$EXP/run_seed_experiments.py"` | PPO, Lagrangian/PID, CPO, PPO-Shield; ten seeds. Pair with the following extended-budget row to cover all six tasks. |
| MASA: baselines on Bridge v2 and MiniPacman | `BRIDGE_CROSSING_V2_TIMESTEPS=1600000 MINIPACMAN_TIMESTEPS=2000000 CPU_OFFSET=0 SWEEP_PARALLEL="$JOBS" PAPER_OUT_BASE="$RUNS/masa_baselines_$TAG" bash "$EXP/run_extended_rl_baseline_sweeps.sh"` | All standard baseline methods, ten seeds; the two environments run sequentially. Explicit budgets match the retained paper comparison. |
| MASA: safe-initialised RL-SGF | `"$PY" "$EXP/run_baselines_pspo_init_ablation.py" --methods rl_sgf --no-cold-start --cpu-ids "$CPUS" --max-parallel "$JOBS" --output-root "$RUNS/rl_sgf_init_$TAG"` | Six environments × ten seeds. Defaults to RL-SGF-PPO; use --rl-sgf-update-mode reinforce for the original variant. Does not automatically replace the historical plotting cohort. |
| MASA: all final-projected baselines | `"$PY" "$EXP/run_safe_initialised_baseline_projection.py" --methods ppo ppo_lagrangian ppo_pid_lagrangian cpo ppo_shield rl_sgf --output-root "$RUNS/projected_$TAG"` | Methods × ten seeds concurrently, one environment at a time; all six methods require 60 admitted CPUs. Requires existing static-LID/base-policy inputs. |
| FrozenLake: joint PPO/PSPO-LS sweep | `"$PY" "$EXP/run_frozenlake_shaping_sweep.py" --methods ppo pspo --size "$SIZE" --total-timesteps "$STEPS" --output-dir "$RUNS/fl_joint_${SIZE}_$TAG" && "$PY" "$EXP/run_frozenlake_shaping_sweep.py" --supervise "$RUNS/fl_joint_${SIZE}_$TAG/_orchestrator/launch_manifest.json"` | Prepare, then execute 20 jobs: two methods × ten seeds. One map size per invocation; matched shaped training and raw evaluation. |
| FrozenLake: PPO control for PSPO-LS | `"$PY" "$EXP/run_frozenlake_shaping_sweep.py" --methods ppo --size "$SIZE" --total-timesteps "$STEPS" --output-dir "$RUNS/fl_ppo_${SIZE}_$TAG" && "$PY" "$EXP/run_frozenlake_shaping_sweep.py" --supervise "$RUNS/fl_ppo_${SIZE}_$TAG/_orchestrator/launch_manifest.json"` | Ten seeds; use the same size/budget as the next row. |
| FrozenLake: verify-first PSPO-LS, enforcement every rollout | `"$PY" "$EXP/run_frozenlake_segment_frequency_sweep.py" --size "$SIZE" --total-timesteps "$STEPS" --safety-frequency 1 --output-dir "$RUNS/fl_ls_vf_${SIZE}_$TAG" && "$PY" "$EXP/run_frozenlake_segment_frequency_sweep.py" --supervise "$RUNS/fl_ls_vf_${SIZE}_$TAG/_orchestrator/launch_manifest.json"` | Prepare, then execute ten seeds. Repeat the PPO-control and this row for each requested map size. |
| Inference: PSPO-LS vs PPO-Shield | `"$PY" "$EXP/run_inference_time_comparison.py" --pspo-variant segment --max-parallel "$JOBS" --output-dir "$RUNS/inference_ls_$TAG" --launch` | Six environments × ten seeds × two methods; uses existing saved checkpoints. Replace segment with orthotope for the older PSPO variant. |
| Fresh training-time comparison | `"$PY" "$EXP/run_timed_main_comparison.py" --max-parallel "$JOBS" --output-dir "$RUNS/timing_$TAG" --launch` | Default methods: PPO, Lagrangian/PID, CPO, PPO-Shield, segment PSPO and lookup PSPO. Select --safe-region-shape orthotope or --verify-first as needed. Does not cover projected variants or FrozenLake. Saves timing JSON and training_times.csv; does not save final models. |
| Training times: all MASA baselines | `"$PY" "$EXP/run_timed_main_comparison.py" --methods ppo ppo_lagrangian ppo_pid_lagrangian cpo rl_sgf ppo_shield --max-parallel "$JOBS" --output-dir "$RUNS/masa_baseline_times_$TAG" --launch` | Six environments × ten seeds = 360 jobs. RL-SGF-PPO is the new default and reuses its saved safe initial actor. Select --rl-sgf-update-mode reinforce for native RL-SGF. Saves timing JSON and training_times.csv; does not save final models. |
| MASA: matched verify-first PSPO-LS cohort | `"$PY" "$EXP/run_pspo_verify_first_segment.py" --run-name "pspo_verifyfirst_$TAG" && "$PY" "$EXP/run_pspo_verify_first_segment.py" --supervise "$RUNS/pspo_verifyfirst_$TAG/_orchestrator/launch_manifest.json"` | Reuses exact initial actors/configurations from segment_lid. Six environments × ten seeds; 60 CPUs. |
| MASA: verify-first orthotope PSPO | `"$PY" "$EXP/launch_pspo_multi_env.py" --safe-region-shape orthotope --verify-first true --run-name "pspo_orthotope_vf_$TAG"` | Six environments × ten seeds; 60 CPUs. For a newly initialised verify-first segment run, change orthotope to segment. |
| Stochastic action-sampling safety | `"$PY" "$EXP/run_temperature_sweep.py" --workers "$JOBS" --output-dir "$TEMP"` | Baseline variants plus default PSPO-LS, six environments, ten seeds, all default temperatures. Select --variant pspo for historical orthotope or --variant pspo_ls_verify_first for verify-first; RL-SGF is not registered. |
| Default segment ablation suite | `"$PY" "$EXP/run_pspo_four_ablations.py" --cpu-ids "$CPUS" --max-parallel "$JOBS" --output-root "$PWD/artifacts/ablation_studies/pspo_segment_ablations_$TAG" --allow-dirty` | Defaults to no_entropy, no_margin, instrumented region-first and verify-first against segment controls. |
| Orthotope initialisation/update ablations | `"$PY" "$EXP/run_pspo_four_ablations.py" --safe-region-shape orthotope --variants no_entropy ce_only fixed_lid no_gradient region_first_instrumented verify_first --cpu-ids "$CPUS" --max-parallel "$JOBS" --output-root "$ABLA" --allow-dirty` | Six environments × ten seeds. --allow-dirty records the current uncommitted source state; omit it after committing the revamp. Canonical controls/inputs must exist. |
| PSPO-LS initialisation ablations | `"$PY" "$EXP/run_pspo_segment_init_ablations.py" --variants no_entropy no_margin --cpu-ids "$CPUS" --output-root "$PWD/artifacts/ablation_studies/pspo_segment_init_ablations_$TAG"` | Each environment/variant group requires one CPU per selected seed: use at least ten CPUs for the default ten seeds, or select fewer with --seeds. |
| LID-objective geometry diagnostic | `"$PY" "$EXP/run_pspo_lid_geometry_comparison.py" --output-dir "$RUNS/lid_geometry_$TAG"` | Small matched training experiment that also plots LID slices; requires its default base actor and shield inputs. |
| Re-evaluate initial/final PSPO-LS policies | `"$PY" "$EXP/compare_pspo_initial_final_rewards.py" --workers "$JOBS" --output-dir "$RUNS/initial_final_$TAG"` | Defaults to historical segment_lid/two_hidden; use --run-root for another cohort. |

**Plots, tables and result analyses**

| Experiment / output | Command | Scope / dependency |
| --- | --- | --- |
| Main MASA evaluation/exploration curves | `"$PY" "$PLOT/plot_masa_learning_curve_comparisons.py"` | Produces combined and separate evaluation/exploration figures, including projected baselines and PSPO variants. |
| MASA final-method table | `"$PY" "$PLOT/generate_masa_all_methods_table.py"` | Reads retained historical cohorts by default. |
| Test-time prerequisite 1: baseline reward/safety CSV | `"$PY" "$PLOT/plot_rl_sgf_final_policy_reward_safety_bars.py"` | Run before the PSPO-LS test-time chart; supplies its default --main-csv input. |
| Test-time prerequisite 2: projected-baseline CSV/assets | `"$PY" "$PLOT/generate_safe_projection_comparison_assets.py"` | Run before the PSPO-LS test-time chart; supplies its default --projected-csv input. |
| Test-time reward/safety: baselines and projected baselines vs PSPO-LS | `"$PY" "$PLOT/plot_pspo_ls_test_time_bars.py"` | Run after both prerequisites. Produces main and projected reward/safety figures, plus projected reward-only. |
| Initialisation losses vs action logits | `"$PY" "$PLOT/plot_safe_policy_initialisation_logits.py"` | Illustrative constants; no trained-policy results required. |
| MASA initial frames | `"$PY" "$PLOT/render_masa_initial_frames.py"` | Six environment frames at reset seed 0; reads canonical configurations. |
| Ablation reward gain retained: orthotope | `"$PY" "$PLOT/plot_pspo_ablation_gain_retained.py" --family orthotope` | Historical initial-reward anchors and fixed-LID/control roots; align them before using fresh runs. |
| Ablation reward gain retained: segment | `"$PY" "$PLOT/plot_pspo_ablation_gain_retained.py" --family segment` | Reads historical segment_lid and segment-ablation roots; align paths/anchors for fresh runs. |
| PSPO variant reward/time figure | `"$PY" "$PLOT/plot_pspo_variant_reward_time_aamas.py" --all-variants` | Includes both geometries and both enforcement strategies. |
| PSPO variant reward/time tables | `"$PY" "$PLOT/generate_pspo_variant_tables.py"` | Reads retained historical cohorts by default. |
| FrozenLake scalability figure | `"$PY" "$PLOT/plot_frozenlake_scalability.py"` | Uses the exact historical 16/32/64/128 cohorts configured in the script. |
| FrozenLake frequency-1 evaluation curves | `"$PY" "$PLOT/plot_frozenlake_freq1_evaluation_curves.py"` | Reads retained historical cohorts by default. |
| FrozenLake PPO/PSPO reward-efficiency curves | `"$PY" "$PLOT/plot_frozenlake_ppo_pspo_reward_efficiency.py"` | Reads retained historical cohorts by default. |
| Temperature-sampling reward/safety figures | `"$PY" "$PLOT/plot_temperature_sweep.py" --sweep-dir "$TEMP"` | Reads the new sampling sweep from the experiment table. |
| Temperature-sampling safety table | `"$PY" "$PLOT/tabulate_temperature_sweep_safety.py" --variant pspo_ls --sweep-dir "$TEMP"` | Change --variant to pspo or another registered policy variant as needed. |
| Recover historical training times | `"$PY" "$PLOT/generate_training_time_comparison.py"` | Extracts existing historical timings; does not automatically summarise the new timing_$TAG run. |
| Analyse default segment ablations | `"$PY" "$PLOT/analyse_pspo_four_ablations.py" --root "$PWD/artifacts/ablation_studies/pspo_segment_ablations_$TAG"` | Uses segment controls and the new segment suite from the experiment table. |
| Analyse newly launched orthotope ablations | `"$PY" "$PLOT/analyse_pspo_four_ablations.py" --safe-region-shape orthotope --root "$ABLA"` | Writes summary data, reports, curves and paired comparisons for the new suite. |
| Initialisation-only ablation analysis | `"$PY" "$PLOT/analyse_pspo_four_ablations.py" --safe-region-shape orthotope --root "$ABLA" --variants control no_entropy ce_only --output-dir "$ABLA/analysis/policy_initialisation"` | Run before the following two initialisation asset generators. |
| Initialisation-ablation tables | `"$PY" "$PLOT/generate_policy_initialisation_tables.py" --results "$INIT_JSON"` | Uses the explicit new initialisation analysis JSON. |
| Initialisation-ablation paper figures/tables | `"$PY" "$PLOT/generate_policy_initialisation_paper_assets.py" --results "$INIT_JSON"` | Uses the explicit new initialisation analysis JSON. |
| Shifted, seed-paired ablation reward ratios | `"$PY" "$PLOT/compute_pspo_ablation_reward_ratios.py"` | Historical cohorts; separate statistic from the fraction of reward gain retained. |
| Historical orthotope evaluation/exploration curves | `"$PY" "$PLOT/plot_extended_budget_learning_curves.py"` | Older orthotope comparison; also provides shared environment definitions. |
| Historical orthotope final-policy table | `"$PY" "$PLOT/generate_final_policy_table.py"` | Reads retained historical cohorts by default. |
| Historical orthotope final reward/safety bars | `"$PY" "$PLOT/plot_final_policy_reward_safety_bars.py" --include-shield-off` | Optional older presentation; PSPO uses orthotope controls, not PSPO-LS. |

**Additional experiment entry points and cluster launches**

| Experiment / output | Command | Scope / dependency |
| --- | --- | --- |
| One MASA environment, ten PSPO seeds | `ENV_NAME=colour_bomb SEEDS="0 1 2 3 4 5 6 7 8 9" CPU_IDS="8-17" RUN_NAME="pspo_one_$TAG" SAFE_REGION_SHAPE=segment bash "$EXP/run_pspo_one_env.sh"` | Replace the example CPU range with ten available CPUs. Add DRY_RUN=1 to preview. |
| Safe-initialisation vs cold-start baseline ablation | `"$PY" "$EXP/run_baselines_pspo_init_ablation.py" --methods ppo ppo_lagrangian ppo_pid_lagrangian cpo ppo_shield rl_sgf --cpu-ids "$CPUS" --max-parallel "$JOBS" --output-root "$RUNS/baseline_init_$TAG"` | All six environments × ten seeds × six methods × two initialisation arms. |
| Single-seed shaped FrozenLake PPO | `"$PY" "$EXP/run_frozenlake_ppo_shaping.py" --size "$SIZE" --seed 0 --total-timesteps "$STEPS" --output-dir "$RUNS/fl_ppo_single_$TAG"` | Add --shaping-scale 0 for the unshaped control. |
| Single-seed shaped FrozenLake orthotope PSPO | `"$PY" "$EXP/run_frozenlake_pspo_shaping.py" --safe-region-shape orthotope --size "$SIZE" --seed 0 --total-timesteps "$STEPS" --output-dir "$RUNS/fl_pspo_single_$TAG"` | One-seed worker; the sweep launcher is the preferred multi-seed entry point. |
| Single-seed shaped FrozenLake cost-based baseline | `"$PY" "$EXP/run_frozenlake_safe_baseline_shaping.py" --algorithm cpo --size "$SIZE" --seed 0 --total-timesteps "$STEPS" --output-dir "$RUNS/fl_cpo_single_$TAG"` | Choose the supported algorithm; the shaping sweep supports the multi-seed safe-RL matrix. |
| FrozenLake segment-first PSPO, default cadence | `"$PY" "$EXP/run_frozenlake_segment_shaping.py" --size "$SIZE" --total-timesteps "$STEPS" --output-dir "$RUNS/fl_segment_${SIZE}_$TAG" && "$PY" "$EXP/run_frozenlake_segment_shaping.py" --supervise "$RUNS/fl_segment_${SIZE}_$TAG/_orchestrator/launch_manifest.json"` | Ten seeds; add --verify-first to the preparation command for verify-first at this cadence. |
| Safety-only slippery FrozenLake PSPO | `"$PY" "$EXP/run_stochastic_frozenlake_pspo.py" --size "$SIZE" --state-representation state_id_lookup --output-root "$RUNS/fl_safety_${SIZE}_$TAG" --launch` | Prepares safety inputs and launches a detached ten-seed controller. Stochastic environment transitions; distinct from stochastic action sampling. |
| Matched slippery FrozenLake baselines | `"$PY" "$EXP/run_stochastic_frozenlake_baselines.py" --size "$SIZE" --pspo-root "$RUNS/fl_safety_${SIZE}_$TAG" --methods ppo ppo_lagrangian ppo_pid_lagrangian cpo ppo_shield --output-root "$RUNS/fl_safety_baselines_${SIZE}_$TAG" --screen-name "fl-baselines-$TAG" --launch` | Run after the preceding PSPO experiment has prepared experiment.json and inputs. |
| Probe cluster hosts | `bash "$EXP/lab_cluster/probe_hosts.sh" --out /tmp/pspo_hosts.tsv` | Read-only SSH capacity probe; requires your host/SSH setup. |
| Cluster baseline launch | `"$PY" "$EXP/lab_cluster/dispatch.py" --hosts /tmp/pspo_hosts.tsv --env mini_pacman --out-base "$RUNS/cluster_baselines_$TAG" --launch` | One environment per invocation; defaults to the dispatcher's registered method groups. Inspect --help / plan before selecting a different protocol. |
| Cluster baseline status | `"$PY" "$EXP/lab_cluster/status.py" --out-base "$RUNS/cluster_baselines_$TAG" --env mini_pacman --check-hosts` | Monitor the preceding distributed baseline launch. |
| Cluster temperature-sweep launch | `"$PY" "$EXP/lab_cluster/dispatch_temperature_sweep.py" --hosts /tmp/pspo_hosts.tsv --sweep-dir "$TEMP" --workers-per-host 4 --launch` | Distributed alternative to the local temperature sweep; avoid launching both against the same pending cells. |
| Cluster initialisation ablations: probe | `SOURCE_ROOT="$PWD" OUTPUT_ROOT="$ABLA" HOSTS_TSV=/tmp/pspo_init_hosts.tsv bash "$EXP/run_policy_initialisation_ablations.sh" probe` | Wrapper for lab_cluster/pspo_initialisation.py. Use a source checkout containing the new paths; requires SSH access. |
| Cluster orthotope initialisation ablations: prepare base policies | `"$PY" "$EXP/lab_cluster/pspo_initialisation.py" dispatch --safe-region-shape orthotope --phase base --hosts-tsv /tmp/pspo_init_hosts.tsv --source-root "$PWD" --output-root "$ABLA" --allow-dirty --launch` | Wait for the base phase to finish before launching training. |
| Cluster orthotope initialisation ablations: train | `"$PY" "$EXP/lab_cluster/pspo_initialisation.py" dispatch --safe-region-shape orthotope --phase train --hosts-tsv /tmp/pspo_init_hosts.tsv --source-root "$PWD" --output-root "$ABLA" --allow-dirty --launch` | Alternative to running the local initialisation-ablation jobs; requires ready shared base policies. |
| Cluster initialisation ablations: status | `"$PY" "$EXP/lab_cluster/pspo_initialisation.py" status --output-root "$ABLA" --check-hosts` | Reports base-policy and training completion. |

Commands were checked against the current source interfaces; no experiments were launched while preparing this catalogue.

The experiment commands default to all six MASA environments and seeds 0–9 unless a row narrows the selection. Automatic full MASA PSPO launchers need 60 CPUs. For the segment-initialisation row, the example `CPUS` value supplies ten CPU IDs; retain at least ten IDs for the default seed list or pass fewer seeds. The standard baseline and orthotope-ablation schedulers preassign CPU IDs, so a concurrency limit alone does not guarantee disjoint cores for all active jobs. Run independent sweeps on separate allocated CPU pools.

FrozenLake sweep preparation does not execute training: each table command explicitly chains preparation and supervision with `&&`. The supervisors run in the foreground; use a persistent host terminal/session for long jobs. Run one map size at a time, or allocate distinct CPU pools. For the retained scalability protocol, use SIZE=16 or 32 with STEPS=204800, and SIZE=64 or 128 with STEPS=200000 (the rollout-aligned realised count may differ).

Training commands above generally create new output cohorts. Most plotters, the inference benchmark and the temperature evaluator still select fixed historical cohort/checkpoint roots. The plotting defaults reproduce retained results; they do not automatically use newly named runs. Where available, supply the matching root/CSV options. The gain-retained plot additionally has historical initial-reward anchors and a fixed-LID source root that differs from the general ablation launcher output. Align these inputs before interpreting fresh results.

Run the test-time plotting prerequisites in table order. The final plotter emits both main and projected-baseline reward/safety charts. Its projected-data validation requires all final safety rates to be 1; that historical condition must also hold for any new cohort used without editing the caption/validation.

The fresh timing sweep covers its registered baseline/PSPO matrix (segment by default, orthotope available explicitly) and suppresses final checkpoints. New RL-SGF training defaults to PPO proposals filtered by SGF; use `--rl-sgf-update-mode reinforce` for the original variant. The baseline, projected-baseline and timing launchers accept the option; see [RL-SGF update modes](rl_sgf_update_modes.md). There is no single retained launcher that times every method and PSPO variant across the expanded study list. The historical timing extraction command is a separate workflow.

`--allow-dirty` is shown for the currently uncommitted revamp; omit it after committing a source snapshot. Remote initialisation work needs both base and training phases and the same source/output paths. The two plotting modules `_aamas_reward_safety.py` and `_paper_style.py` are importable helpers and have no standalone launch command.

PSPO line segments are now the default for active training, timing and inference
launchers. Orthotope mode must be requested explicitly; verify-first remains
independent of geometry. Saved historical run configurations and plotting
cohorts retain their original meanings.
