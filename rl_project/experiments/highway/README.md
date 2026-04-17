# Highway Parking Setup

This directory provides a clean setup for the `highway-env` Parking task with:

- discrete actions (via action-grid discretization),
- flat vector observations,
- configurable parked-vehicle placement from YAML task settings.

## Files

- `parking_setup.py`: environment class + wrappers + config loaders.
- `create_env.py`: CLI smoke-test script that instantiates an env from YAML.
- `settings/task_settings.yaml`: source/downstream task definitions.

## Configure Parked Vehicle Locations

In `settings/task_settings.yaml`, set `parked_vehicles_spots` under a task:

```yaml
source:
  parked_vehicles_spots:
    - ["a", "b", 0]
    - ["a", "b", 1]
    - ["b", "c", 0]
    - ["b", "c", 2]
  vehicles_count: 4
```

Notes:

- Each spot is a lane index tuple `[start_node, end_node, lane_id]`.
- If `parked_vehicles_spots` is set, `vehicles_count` must match its length.
- Set `parked_vehicles_spots: null` to let placements be random.

## Quick Usage

From repository root:

```bash
python -m rl_project.experiments.highway.create_env \
  --task-setting default \
  --task-role source \
  --rollout-steps 20
```
