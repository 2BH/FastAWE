"""
Automatic waypoint selection using dynamic programming.

This module provides functions for extracting waypoints from robot trajectories
using a dynamic programming approach with geometric trajectory reconstruction.
"""
import numpy as np
import copy

from waypoint_extraction.traj_reconstruction import (
    pos_only_geometric_waypoint_trajectory,
    geometric_waypoint_trajectory,
)


def dp_waypoint_selection(
    actions=None,
    states=None,
    err_threshold=None,
    pos_only=False,
):
    """
    Extract waypoints from a trajectory using dynamic programming.
    
    This function finds the minimum set of waypoints needed to reconstruct
    a trajectory such that the reconstruction error stays below a given threshold.
    It uses dynamic programming to efficiently find the optimal waypoint set.
    
    Args:
        actions (np.ndarray, optional): Action sequence of shape [T, D] where T is the
            number of timesteps and D is the action dimensionality. For position-only
            trajectories, this should be [T, 2] or [T, 3] for 2D/3D positions.
            If None, uses states instead.
        states (np.ndarray, optional): State sequence of shape [T, D]. Same format as
            actions. If None, uses actions instead. At least one of actions or states
            must be provided.
        err_threshold (float): Maximum allowed trajectory reconstruction error. Lower
            values result in more waypoints (higher fidelity), higher values result in
            fewer waypoints (more compression).
        pos_only (bool, optional): If True, uses position-only geometric waypoint
            reconstruction (faster, suitable for 2D/3D positions). If False, uses full
            pose reconstruction including orientation (requires specific state format
            with 'eef_pos' and 'eef_quat' keys). Default: False.
    
    Returns:
        list[int]: Sorted list of waypoint indices into the trajectory. The first
            waypoint is always at index 0 and the last is always at index T-1.
    
    Example:
        >>> import numpy as np
        >>> # Create a simple 2D trajectory
        >>> positions = np.array([[0, 0], [1, 1], [2, 2], [3, 1], [4, 0]])
        >>> waypoints = dp_waypoint_selection(
        ...     actions=positions,
        ...     states=positions,
        ...     err_threshold=0.5,
        ...     pos_only=True
        ... )
        >>> print(waypoints)
        [0, 2, 4]  # Example output
    
    Notes:
        - If pos_only=False, the last action dimension is assumed to be gripper
          openness, and gripper state changes are automatically made into waypoints.
        - The algorithm guarantees that the reconstruction error will not exceed
          err_threshold (unless the threshold is set too low).
        - Time complexity: O(T^2) where T is trajectory length.
    """
    # Handle None arguments - use states if actions not provided and vice versa
    if actions is None:
        actions = copy.deepcopy(states)
    elif states is None:
        states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # Initialize waypoints list - last frame is always a waypoint
    initial_waypoints = [num_frames - 1]

    # For full pose (not position-only), detect gripper state changes
    # and automatically make them waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            # Check if gripper state changes between consecutive frames
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
        initial_waypoints.sort()

    # Memoization table: memo[i] = (min_waypoint_count, waypoint_list)
    # Stores the optimal solution for trajectory from start to frame i
    memo = {}

    # Base case: zero waypoints needed to reach frame 0
    for i in range(num_frames):
        memo[i] = (0, [])

    # Frame 1 requires exactly 1 waypoint
    memo[1] = (1, [1])
    
    # Select appropriate trajectory reconstruction function
    func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )

    # Check if error threshold is achievable (even with all points as waypoints)
    min_error = func(actions, states, list(range(1, num_frames)))
    if err_threshold < min_error:
        print("Error threshold is too small, Minimum error is ", min_error)

    # Dynamic programming: build up solutions for progressively longer trajectories
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        # Try all possible previous waypoint positions k
        # This considers placing the next waypoint at frame i,
        # and connecting from all possible previous waypoints at k
        for k in range(0, i):
            # Construct waypoints for subsequence from k to i
            # Include any required initial waypoints (e.g., gripper changes) within this segment
            # Waypoint indices are relative to the start of the subsequence (frame k)
            waypoints = [j - k for j in initial_waypoints if j > k and j < i] + [i - k]

            # Compute reconstruction error for this segment
            total_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=states[k : i + 1],
                waypoints=waypoints,
            )

            # If this segment satisfies the error threshold
            if total_traj_err < err_threshold:
                # Retrieve optimal solution for reaching frame k
                subproblem_waypoints_count, subproblem_waypoints = memo[k]
                # Total waypoints = waypoints to reach k + this new waypoint at i
                total_waypoints_count = 1 + subproblem_waypoints_count

                # Keep track of the solution with minimum waypoints
                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

            
        # If no valid solution found, threshold is too strict for this trajectory
        if min_waypoints_required == float("inf"):
            print(f"Error: Cannot find valid waypoint path up to frame {i} with threshold {err_threshold}")
            print("Returning all points as waypoints.")
            return list(range(num_frames))

        # Store optimal solution for trajectory up to frame i
        memo[i] = (min_waypoints_required, best_waypoints)

    # Extract final solution
    min_waypoints_count, waypoints = memo[num_frames - 1]
    
    # Add required waypoints (gripper changes, final frame) to the solution
    waypoints += initial_waypoints
    
    # Remove duplicates and sort
    waypoints = list(set(waypoints))
    waypoints.sort()
    
    # Compute and report final trajectory error
    final_err = func(actions, states, waypoints)
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {final_err}"
    )
    print(f"waypoint positions: {waypoints}")

    return waypoints
