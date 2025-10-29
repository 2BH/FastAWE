"""
Batch waypoint extraction for robot trajectory datasets.

This script provides command-line and programmatic interfaces for extracting waypoints
from robot trajectory datasets using position-only geometric waypoint extraction with
dynamic programming optimization.

Author: Baohe Zhang

Dataset Structure:
    The script expects the following files in the data directory:
    
    - states.npy: NumPy array of shape [Num_Episodes, Max_Episode_Len, Num_Dims]
        First n dimensions are positions (2 for x,y plane, 3 for x,y,z space).
        Additional dimensions can contain other state information (ignored).
    
    - seq_lengths.pkl: Python pickle file containing a dictionary where:
        Key: episode_id (int)
        Value: actual length of that episode (int)
    
    - actions.npy (optional): NumPy array of shape [Num_Episodes, Max_Episode_Len, Num_Actions]
        Robot actions corresponding to each state.
        For full pose extraction (pos_only=False), the last action dimension is
        assumed to be gripper openness.

Usage:
    Command line:
        python waypoints.py --data-dir /path/to/data --thresholds 2.0 4.0 --num-dims 2
    
    As module:
        from waypoints import load_data, compute_statistics, preload_episodes, compute_waypoints_parallel
        
        states, seq_lengths = load_data(data_dir)
        stats = compute_statistics(states, seq_lengths, num_dims=2)
        tasks = preload_episodes(states, seq_lengths, normalize=True, stats=stats, num_dims=2)
        waypoints = compute_waypoints_parallel(tasks, err_threshold=4.0)

Features:
    - Parallel processing using multiprocessing for efficiency
    - Optional position normalization for consistent thresholding
    - Support for 2D and 3D trajectories
    - Multiple error thresholds in a single run
    - Comprehensive progress tracking and summary statistics
"""

import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from waypoint_extraction import dp_waypoint_selection


def _compute_waypoints_worker(ep_id: int, episode_positions: np.ndarray, err_threshold: float):
    """
    Worker function for parallel waypoint computation.
    
    This function is called by worker processes in the parallel processing pool.
    It computes waypoints for a single episode using the DP waypoint selection algorithm.
    
    Args:
        ep_id (int): Episode ID for tracking which episode is being processed.
        episode_positions (np.ndarray): Position sequence of shape [T, D] where T is
            trajectory length and D is dimensionality (2 for x,y; 3 for x,y,z).
            Should already be normalized if normalization is being used.
        err_threshold (float): Maximum allowed trajectory reconstruction error.
            Lower values produce more waypoints.
        
    Returns:
        tuple: (ep_id, waypoints) where:
            - ep_id (int): The episode ID (passed through for identification)
            - waypoints (list[int]): List of waypoint indices for this episode
    """
    waypoints = dp_waypoint_selection(
        actions=episode_positions,
        states=episode_positions,
        err_threshold=err_threshold,
        pos_only=True,
    )
    return ep_id, waypoints


def load_data(data_dir):
    """
    Load dataset from directory containing states and sequence lengths.
    
    Expects the following files in the data directory:
    - states.npy: NumPy array of shape [Num_Episodes, Max_Episode_Len, Num_Dims]
    - seq_lengths.pkl: Pickle file containing dict mapping episode_id -> actual_length
    
    Args:
        data_dir (str or Path): Path to directory containing the data files.
    
    Returns:
        tuple: (states, seq_lengths) where:
            - states (np.ndarray): State array of shape [Num_Episodes, Max_Episode_Len, Num_Dims]
            - seq_lengths (dict or list): Actual sequence length for each episode
    
    Raises:
        FileNotFoundError: If required data files are not found.
    """
    data_dir = Path(data_dir)
    
    # Load states and sequence lengths
    states = np.load(data_dir / "states.npy")
    with open(data_dir / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)
    
    # Print dataset information
    print(f"Loaded states with shape: {states.shape}")
    print(f"Number of episodes: {len(seq_lengths)}")
    print(f"Sequence lengths range: [{min(seq_lengths)}, {max(seq_lengths)}]")
    
    return states, seq_lengths


def compute_statistics(states, seq_lengths, num_episodes=None, num_dims=2):
    """
    Compute mean and standard deviation statistics for position normalization.
    
    Collects all position data from the specified episodes and computes dataset-wide
    statistics. These statistics are used to normalize positions to zero mean and
    unit variance, which can improve waypoint extraction consistency.
    
    Args:
        states (np.ndarray): State tensor of shape [Num_Episodes, Max_Episode_Len, Num_Dims].
            First num_dims dimensions should be positions.
        seq_lengths (dict or list): Actual sequence length for each episode. Indexable by
            episode ID.
        num_episodes (int, optional): Number of episodes to use for computing statistics.
            If None, uses all episodes. Default: None.
        num_dims (int, optional): Number of position dimensions to consider. Use 2 for
            x,y planar motion, 3 for x,y,z spatial motion. Default: 2.
        
    Returns:
        dict: Statistics dictionary containing:
            - 'mean' (np.ndarray): Mean position of shape [num_dims]
            - 'std' (np.ndarray): Standard deviation of shape [num_dims]
    
    Notes:
        - Only uses actual trajectory data (up to seq_lengths[i] for each episode),
          ignoring padded values.
        - If std is too small (< 1e-6), it's set to 1.0 to avoid division by zero.
    """
    
    # Determine number of episodes to use
    if num_episodes is None:
        num_episodes = len(seq_lengths)
    else:
        num_episodes = min(num_episodes, len(seq_lengths))
    
    # Collect all valid position data from episodes
    all_positions = []
    for ep_id in range(num_episodes):
        seq_len = seq_lengths[ep_id]
        # Extract only position dimensions for this episode (up to actual length)
        positions = states[ep_id, :seq_len, :num_dims]
        all_positions.append(positions)
    
    # Concatenate all positions and compute statistics
    all_positions = np.concatenate(all_positions, axis=0)
    mean = np.mean(all_positions, axis=0)
    std = np.std(all_positions, axis=0)
    
    # Avoid division by zero for dimensions with no variance
    std = np.where(std < 1e-6, 1.0, std)
    
    # Print computed statistics
    print(f"\nComputed statistics from {num_episodes} episodes:")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    
    return {"mean": mean, "std": std}


def preload_episodes(states, seq_lengths, num_episodes=None, normalize=True, stats=None, num_dims=2):
    """
    Preload and prepare episode data for parallel waypoint computation.
    
    Extracts position data for each episode, optionally normalizes it, and prepares
    a list of tasks for parallel processing. This separates data loading from
    computation to enable efficient parallelization.
    
    Args:
        states (np.ndarray): State tensor of shape [Num_Episodes, Max_Episode_Len, Num_Dims].
        seq_lengths (dict or list): Actual sequence length for each episode.
        num_episodes (int, optional): Number of episodes to process. If None, processes
            all episodes. Default: None.
        normalize (bool, optional): Whether to normalize positions using provided stats.
            Normalization applies (x - mean) / std. Default: True.
        stats (dict, optional): Statistics dictionary with 'mean' and 'std' keys.
            Required if normalize=True. Should be computed using compute_statistics().
        num_dims (int, optional): Number of position dimensions to extract. Use 2 for
            x,y planar motion, 3 for x,y,z spatial motion. Default: 2.

    Returns:
        list[tuple]: List of (ep_id, episode_positions) tuples where:
            - ep_id (int): Episode ID
            - episode_positions (np.ndarray): Position array of shape [T, num_dims],
              normalized if normalize=True
    
    Raises:
        ValueError: If normalize=True but stats is None.
    """
    # Determine number of episodes to process
    if num_episodes is None:
        num_episodes = len(seq_lengths)
    else:
        num_episodes = min(num_episodes, len(seq_lengths))
    
    # Print status message
    normalize_str = "with normalization" if normalize else "without normalization"
    print(f"\nPreloading {num_episodes} episodes {normalize_str}...")
    preload_tasks = []
    
    # Prepare each episode for processing
    for ep_id in tqdm(range(num_episodes), desc="Preloading episodes"):
        seq_len = seq_lengths[ep_id]
        # Extract position dimensions for this episode (up to actual length)
        episode_positions = states[ep_id, :seq_len, :num_dims]
        
        # Apply normalization if requested
        if normalize:
            if stats is None:
                raise ValueError("Stats must be provided when normalize=True")
            episode_positions = (episode_positions - stats["mean"]) / stats["std"]
        
        preload_tasks.append((ep_id, episode_positions))
    
    return preload_tasks


def compute_waypoints_parallel(preload_tasks, err_threshold, max_workers=None):
    """
    Compute waypoints for all episodes in parallel using multiprocessing.
    
    Distributes waypoint computation across multiple worker processes for efficiency.
    Each worker processes one episode at a time using the DP waypoint selection algorithm.
    
    Args:
        preload_tasks (list[tuple]): List of (ep_id, episode_positions) tuples from
            preload_episodes(). Each tuple contains an episode ID and its position array.
        err_threshold (float): Maximum allowed trajectory reconstruction error for
            waypoint selection. Lower values produce more waypoints.
        max_workers (int, optional): Number of parallel worker processes. If None,
            uses os.cpu_count() to auto-detect available CPUs. Default: None.
        
    Returns:
        dict: Waypoints dictionary mapping episode_id (int) to waypoint indices (list[int]).
            Example: {0: [0, 15, 32, 99], 1: [0, 8, 21, 120], ...}
    
    Notes:
        - Uses ProcessPoolExecutor for true parallelism (bypasses Python GIL).
        - Progress is displayed using tqdm progress bar.
        - Episodes are processed in any order (results collected as they complete).
    """
    waypoints_dict = {}
    
    # Auto-detect number of workers if not specified
    if max_workers is None:
        max_workers = os.cpu_count()
    
    # Print processing information
    print(f"\nComputing waypoints with {max_workers} workers...")
    print(f"Error threshold: {err_threshold}")
    print("=" * 80)
    
    # Submit all tasks to the process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_waypoints_worker, ep_id, episode_positions, err_threshold): ep_id
            for (ep_id, episode_positions) in preload_tasks
        }
        
        # Collect results as they complete
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Computing waypoints@{err_threshold}"):
            ep_id, waypoints = fut.result()
            waypoints_dict[ep_id] = waypoints
    
    return waypoints_dict


def print_summary_statistics(waypoints_dict, seq_lengths):
    """
    Print summary statistics about waypoint extraction results.
    
    Displays aggregate statistics about number of waypoints per episode,
    compression ratios, and shows examples from the first few episodes.
    
    Args:
        waypoints_dict (dict): Dictionary mapping episode_id to list of waypoint indices.
        seq_lengths (dict or list): Actual sequence length for each episode.
    
    Notes:
        - Compression ratio is (num_waypoints / trajectory_length) * 100.
        - Lower compression ratios indicate more aggressive compression.
    """
    print("\n" + "=" * 80)
    print("Waypoint Extraction Summary")
    print("=" * 80)
    
    # Compute statistics
    num_waypoints = [len(wp) for wp in waypoints_dict.values()]
    compression_ratios = [
        len(waypoints_dict[ep_id]) / seq_lengths[ep_id] * 100
        for ep_id in waypoints_dict.keys()
    ]
    
    # Print aggregate statistics
    print(f"Total episodes processed: {len(waypoints_dict)}")
    print(f"Average waypoints per episode: {np.mean(num_waypoints):.2f}")
    print(f"Median waypoints per episode: {np.median(num_waypoints):.2f}")
    print(f"Min waypoints: {min(num_waypoints)}")
    print(f"Max waypoints: {max(num_waypoints)}")
    print(f"Average compression ratio: {np.mean(compression_ratios):.2f}%")
    
    # Show examples from first few episodes
    print("\n" + "=" * 80)
    print("Example waypoints for first 5 episodes:")
    print("=" * 80)
    for i in range(min(5, len(waypoints_dict))):
        if i in waypoints_dict:
            wp_count = len(waypoints_dict[i])
            ep_len = seq_lengths[i]
            ratio = wp_count / ep_len * 100
            print(f"Episode {i} (length {ep_len}): {wp_count} waypoints ({ratio:.1f}%) -> {waypoints_dict[i]}")


def save_waypoints(waypoints_dict, save_path):
    """
    Save waypoints dictionary to a pickle file.
    
    Args:
        waypoints_dict (dict): Dictionary mapping episode_id to list of waypoint indices.
        save_path (str or Path): Path where the pickle file will be saved.
            Parent directories will be created if they don't exist.
    
    Notes:
        - Creates parent directories automatically if needed.
        - Uses pickle protocol (binary format).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(waypoints_dict, f)
    print(f"\nWaypoints saved to: {save_path}")


def main():
    """
    Main function for command-line waypoint extraction.
    
    Parses command-line arguments, loads data, computes normalization statistics,
    and extracts waypoints for one or more error thresholds. Results are saved
    as pickle files.
    
    See argparse configuration below for available command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Compute waypoints for dataset using DP-based geometric waypoint extraction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/work/dlclarge2/zhangb-WM/dino_wm/data/pusht_noise/val",
        help="Directory containing states.pth and seq_lengths.pkl"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/work/dlclarge2/zhangb-WM/FastAWE",
        help="Directory to save waypoint files"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[4.0],
        help="Error thresholds for waypoint selection (lower = more waypoints)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to process (None = all episodes)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers (None = auto, 1 = sequential)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization of positions"
    )
    parser.add_argument(
        "--num-dims",
        type=int,
        default=2,
        help="Number of dimensions to consider (2 for x,y plane, 3 for x,y,z)"
    )
    args = parser.parse_args()
    normalize = not args.no_normalize
    
    # Load data
    print("Loading dataset...")
    states, seq_lengths = load_data(args.data_dir)
    
    # Compute statistics for normalization if needed
    stats = None
    if normalize:
        stats = compute_statistics(states, seq_lengths, num_episodes=args.num_episodes, num_dims=args.num_dims)
    
    # Preload episodes
    preload_tasks = preload_episodes(
        states, 
        seq_lengths, 
        num_episodes=args.num_episodes,
        normalize=normalize,
        stats=stats,
        num_dims=args.num_dims
    )
    
    # Process each error threshold
    for err_threshold in args.thresholds:
        print("\n" + "=" * 80)
        print(f"Processing with error threshold: {err_threshold}")
        print("=" * 80)
        
        # Compute waypoints
        waypoints_dict = compute_waypoints_parallel(
            preload_tasks,
            err_threshold,
            max_workers=args.max_workers
        )
        
        # Print statistics
        print_summary_statistics(waypoints_dict, seq_lengths)
        
        # Save results
        num_episodes = len(preload_tasks)
        normalize_flag = "norm" if normalize else "unorm"
        save_path = f"{args.save_dir}/pusht_waypoints_{err_threshold}_{num_episodes}_{normalize_flag}.pkl"
        save_waypoints(waypoints_dict, save_path)
    
    print("\n" + "=" * 80)
    print("All waypoint extraction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
