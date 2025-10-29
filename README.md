# Faster Automatic Waypoint Extraction (FastAWE)

This repo contains the re-implementation of [Automatic Waypoint Extraction (AWE)](https://github.com/lucys0/awe)

If you encountered any issue, feel free to contact lucyshi (at) stanford (dot) edu

## Overview

FastAWE provides efficient waypoint extraction from robot trajectories using dynamic programming-based geometric methods. It can process position-only trajectories (2D/3D) or full pose trajectories (position + orientation).

## Installation

1. Clone this repository
```bash
git clone git@github.com:2BH/FastAWE.git
cd FastAWE
```

2. Create a virtual environment
```bash 
conda create -n awe_venv python=3.10
conda activate awe_venv
```


## Dataset Format

The waypoint extraction expects the following files in your data directory:

- `states.npy`: NumPy array of shape `[Num_Episodes, Max_Episode_Len, Num_Dims]`
  - First `n` dimensions are positions (2 for x,y plane; 3 for x,y,z space)
  - Can include additional state dimensions beyond positions
  
- `seq_lengths.pkl`: Python dictionary where:
  - Key: episode ID (int)
  - Value: actual length of that episode (int)
  
- `actions.npy` (optional): NumPy array of shape `[Num_Episodes, Max_Episode_Len, Num_Actions]`
  - Robot actions corresponding to each state
  - Last dimension typically represents gripper openness

## Usage

### Command Line Interface

Basic usage:
```bash
python waypoints.py --data-dir /path/to/your/data --save-dir /path/to/output
```

With custom parameters:
```bash
python waypoints.py \
    --data-dir /path/to/your/data \
    --save-dir /path/to/output \
    --thresholds 2.0 4.0 8.0 \
    --num-episodes 100 \
    --num-dims 2 \
    --max-workers 8
```

### Command Line Arguments

- `--data-dir`: Directory containing `states.npy` and `seq_lengths.pkl` (default: pusht val set)
- `--save-dir`: Directory to save waypoint pickle files (default: current directory)
- `--thresholds`: One or more error thresholds for waypoint selection (default: [4.0])
  - Lower values = more waypoints (higher fidelity)
  - Higher values = fewer waypoints (more compression)
- `--num-episodes`: Number of episodes to process (default: None = all episodes)
- `--num-dims`: Number of position dimensions to consider (default: 2 for x,y plane; use 3 for x,y,z)
- `--max-workers`: Number of parallel workers (default: None = auto-detect CPU count)
- `--no-normalize`: Disable normalization of positions (enabled by default)
- `--verbose`: Print detailed information during processing

### Python API

```python
from waypoint_extraction import dp_waypoint_selection
import numpy as np

# Example: Extract waypoints from a single trajectory
positions = np.random.rand(100, 2)  # 100 timesteps, 2D positions (x, y)

waypoints = dp_waypoint_selection(
    actions=positions,
    states=positions,
    err_threshold=4.0,
    pos_only=True
)
# Returns: list of waypoint indices, e.g., [0, 15, 32, 67, 99]
```

### Function Reference

#### `dp_waypoint_selection()`

Extracts waypoints using dynamic programming to minimize waypoints while keeping trajectory error below threshold.

**Parameters:**
- `actions` (np.ndarray, optional): Action sequence of shape `[T, D]` where T is timesteps, D is action dimensions. If None, uses `states`.
- `states` (np.ndarray, optional): State sequence of shape `[T, D]`. If None, uses `actions`.
- `err_threshold` (float): Maximum allowed trajectory reconstruction error. Lower = more waypoints.
- `pos_only` (bool): If True, uses position-only geometric computation. If False, includes orientation (requires specific state format).

**Returns:**
- `list[int]`: Sorted list of waypoint indices

**Example:**
```python
waypoints = dp_waypoint_selection(
    actions=trajectory_positions,  # shape: [T, 2] or [T, 3]
    states=trajectory_positions,
    err_threshold=4.0,
    pos_only=True
)
```

#### `load_data(data_dir)`

Loads dataset from directory containing `states.npy` and `seq_lengths.pkl`.

**Parameters:**
- `data_dir` (str or Path): Path to data directory

**Returns:**
- `tuple`: (states array, seq_lengths dict)

#### `compute_statistics(states, seq_lengths, num_episodes, num_dims)`

Computes mean and standard deviation for normalization.

**Parameters:**
- `states` (np.ndarray): State tensor of shape `[Num_Episodes, Max_Episode_Len, Num_Dims]`
- `seq_lengths` (list): Actual sequence lengths for each episode
- `num_episodes` (int, optional): Number of episodes for computing stats (None = all)
- `num_dims` (int): Number of position dimensions (2 or 3)

**Returns:**
- `dict`: Dictionary with 'mean' and 'std' numpy arrays

#### `preload_episodes(states, seq_lengths, num_episodes, normalize, stats, num_dims)`

Prepares episode data for parallel processing.

**Parameters:**
- `states` (np.ndarray): State tensor
- `seq_lengths` (list): Sequence lengths
- `num_episodes` (int, optional): Number to process (None = all)
- `normalize` (bool): Whether to normalize positions
- `stats` (dict, optional): Statistics dict (required if normalize=True)
- `num_dims` (int): Number of position dimensions

**Returns:**
- `list[tuple]`: List of (episode_id, episode_positions) tuples

#### `compute_waypoints_parallel(preload_tasks, err_threshold, max_workers)`

Computes waypoints for all episodes in parallel.

**Parameters:**
- `preload_tasks` (list): List of (ep_id, episode_positions) tuples from `preload_episodes()`
- `err_threshold` (float): Error threshold for waypoint selection
- `max_workers` (int, optional): Number of parallel workers (None = auto)

**Returns:**
- `dict`: Dictionary mapping episode_id to list of waypoint indices

## Output Format

The script saves waypoint results as pickle files with the naming convention:
```
{save_dir}/pusht_waypoints_{threshold}_{num_episodes}_{norm|unorm}.pkl
```

Each pickle file contains a dictionary:
```python
{
    0: [0, 15, 32, 67, 99],      # waypoint indices for episode 0
    1: [0, 8, 21, 45, 78, 120],  # waypoint indices for episode 1
    ...
}
```

## Examples

### Example 1: Extract waypoints with default settings
```bash
python waypoints.py --data-dir ./data/val
```

### Example 2: Multiple thresholds for comparison
```bash
python waypoints.py \
    --data-dir ./data/val \
    --thresholds 1.0 2.0 4.0 8.0 \
    --save-dir ./waypoints
```

### Example 3: Process subset without normalization
```bash
python waypoints.py \
    --data-dir ./data/val \
    --num-episodes 50 \
    --no-normalize
```

### Example 4: 3D trajectory waypoints
```bash
python waypoints.py \
    --data-dir ./data/3d_trajectories \
    --num-dims 3 \
    --thresholds 5.0
```

## Performance

- Parallel processing scales with CPU cores
- Typical processing time: ~0.1-1s per episode (depends on length and threshold)
- Memory usage: O(episodes × max_length × dims)


## License

MIT License
