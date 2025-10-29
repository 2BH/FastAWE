"""
Trajectory reconstruction utilities for waypoint-based path evaluation.

This module provides functions for computing trajectory reconstruction errors
when interpolating between waypoints using geometric methods (linear interpolation
for positions, SLERP for quaternions).
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .utils import quat_slerp
import warnings


def linear_interpolation(p1, p2, t):
    """
    Compute linear interpolation between two points.
    
    Args:
        p1 (np.ndarray): Start point of shape [D] where D is dimensionality (2D or 3D).
        p2 (np.ndarray): End point of shape [D].
        t (float): Interpolation parameter in [0, 1]. t=0 returns p1, t=1 returns p2.
    
    Returns:
        np.ndarray: Interpolated point of shape [D].
    """
    return p1 + t * (p2 - p1)


def point_line_distance(point, line_start, line_end):
    """
    Compute shortest distance from a point to a line segment.
    
    This function projects the point onto the line defined by line_start and line_end,
    clips the projection to the segment, and computes the Euclidean distance.
    
    Args:
        point (np.ndarray): Query point of shape [D] where D is dimensionality.
        line_start (np.ndarray): Start of line segment, shape [D].
        line_end (np.ndarray): End of line segment, shape [D].
    
    Returns:
        float: Euclidean distance from point to closest point on line segment.
    
    Notes:
        - If line_start == line_end (degenerate segment), returns distance to line_start.
        - Uses orthogonal projection with clamping to handle segment endpoints.
    """
    line_vector = line_end - line_start
    if np.allclose(line_vector, 0, atol=1e-5):
        warnings.warn("line_vector is zero")
        return np.linalg.norm(point - line_start)
    point_vector = point - line_start
    
    # Compute parameter t for orthogonal projection of point onto the line
    # t=0 projects to line_start, t=1 projects to line_end
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    
    # Clamp to [0, 1] to constrain projection to the segment
    t = np.clip(t, 0, 1)
    
    # Compute projection point and distance
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def point_quat_distance(point, quat_start, quat_end, t, total):
    """
    Compute orientation error between actual and interpolated quaternion.
    
    Uses spherical linear interpolation (SLERP) to interpolate between
    quat_start and quat_end, then computes the rotation magnitude between
    the interpolated and actual quaternion.
    
    Args:
        point (np.ndarray): Actual quaternion at this timestep, shape [4] (x,y,z,w).
        quat_start (np.ndarray): Starting waypoint quaternion, shape [4].
        quat_end (np.ndarray): Ending waypoint quaternion, shape [4].
        t (int): Current timestep within the segment.
        total (int): Total number of timesteps in the segment.
    
    Returns:
        float: Rotation magnitude in radians between actual and interpolated orientation.
    """
    # Interpolate quaternion using SLERP
    pred_point = quat_slerp(quat_start, quat_end, fraction=t / total)
    
    # Compute rotation error magnitude
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat


def geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """
    Compute trajectory reconstruction error using full pose (position + orientation).
    
    This function reconstructs a trajectory by linearly interpolating positions
    and SLERP-ing quaternions between waypoints, then computes the error against
    ground truth states.
    
    Args:
        actions (np.ndarray): Action sequence of shape [T, D] where first 3 columns
            are end-effector positions (eef_pos).
        gt_states (list[dict]): Ground truth states, each containing:
            - 'eef_pos': end-effector position as np.ndarray of shape [3]
            - 'eef_quat': end-effector quaternion as np.ndarray of shape [4] (x,y,z,w)
        waypoints (list[int]): Indices of waypoints in the trajectory. Must include 0.
        return_list (bool, optional): If True, returns (error, error_list). If False,
            returns only error. Default: False.
    
    Returns:
        float or tuple: Maximum trajectory error across all timesteps. If return_list=True,
            returns (max_error, list_of_errors).
    
    Notes:
        - Total error at each timestep is sum of position error and rotation error.
        - Position error uses Euclidean distance to interpolated line segment.
        - Rotation error uses rotation magnitude between quaternions.
    """

    # Ensure waypoints start at index 0 for proper geometric reconstruction
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    
    # Extract positions and quaternions from ground truth states
    gt_pos = [p["eef_pos"] for p in gt_states]
    gt_quat = [p["eef_quat"] for p in gt_states]

    # Extract waypoint positions and orientations
    keypoints_pos = [actions[k, :3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []
    n_segments = len(waypoints) - 1

    # Iterate through each segment between consecutive waypoints
    for i in range(n_segments):
        # Get start and end waypoints for this segment
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        # Extract ground truth points within this segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        # Compute error for each ground truth point against the interpolated trajectory
        for j in range(len(segment_points_pos)):
            # Position error: distance from point to line segment
            pos_err = point_line_distance(
                segment_points_pos[j], start_keypoint_pos, end_keypoint_pos
            )
            # Rotation error: quaternion distance to SLERP interpolation
            rot_err = point_quat_distance(
                segment_points_quat[j],
                start_keypoint_quat,
                end_keypoint_quat,
                j,
                len(segment_points_quat),
            )
            # Total error is sum of position and rotation errors
            state_err.append(pos_err + rot_err)

    if return_list:
        return total_traj_err(state_err), state_err
    return total_traj_err(state_err)


def pos_only_geometric_waypoint_trajectory(
    actions, gt_states, waypoints, return_list=False
):
    """
    Compute trajectory reconstruction error using position-only data.
    
    This function reconstructs a trajectory by linearly interpolating positions
    between waypoints, then computes the error against ground truth positions.
    This is faster than full pose reconstruction and suitable for 2D/3D position data.
    
    Args:
        actions (np.ndarray): Action/position sequence of shape [T, D] where T is
            number of timesteps and D is dimensionality (2 for 2D, 3 for 3D).
        gt_states (np.ndarray): Ground truth positions of shape [T, D], same format
            as actions.
        waypoints (list[int]): Indices of waypoints in the trajectory. Must include 0.
        return_list (bool, optional): If True, returns (error, error_list). If False,
            returns only error. Default: False.
    
    Returns:
        float or tuple: Maximum trajectory error across all timesteps. If return_list=True,
            returns (max_error, list_of_errors).
    
    Notes:
        - Error metric is Euclidean distance from each ground truth point to the
          interpolated line segment between adjacent waypoints.
        - More efficient than full pose reconstruction when orientation is not needed.
    """

    # Ensure waypoints start at index 0 for proper geometric reconstruction
    if waypoints[0] != 0:
        waypoints = [0] + waypoints

    # Extract waypoint positions
    keypoints_pos = [actions[k] for k in waypoints]
    state_err = []
    n_segments = len(waypoints) - 1

    # Iterate through each segment between consecutive waypoints
    for i in range(n_segments):
        # Get start and end waypoints for this segment
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]

        # Extract ground truth points within this segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_states[start_idx:end_idx]

        # Compute error for each ground truth point against the interpolated line segment
        for j in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[j], start_keypoint_pos, end_keypoint_pos
            )
            state_err.append(pos_err)

    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)


def total_state_err(err_dict):
    """
    Compute total state error from error dictionary.
    
    Args:
        err_dict (dict): Dictionary containing 'err_pos' and 'err_quat' keys.
    
    Returns:
        float: Sum of position and quaternion errors.
    """
    return err_dict["err_pos"] + err_dict["err_quat"]


def total_traj_err(err_list):
    """
    Compute total trajectory error from list of per-timestep errors.
    
    Uses maximum error as the aggregation metric (worst-case error).
    
    Args:
        err_list (list[float]): List of errors at each timestep.
    
    Returns:
        float: Maximum error across all timesteps.
    
    Notes:
        - Alternative would be mean error (commented out), but max provides
          stronger guarantees on reconstruction quality.
    """
    # return np.mean(err_list)  # Alternative: use mean error
    return np.max(err_list)


def compute_state_error(gt_state, pred_state):
    """
    Compute error between ground truth and predicted robot states.
    
    Args:
        gt_state (dict): Ground truth state containing:
            - 'eef_pos': end-effector position, shape [3]
            - 'eef_quat': end-effector quaternion, shape [4] (x,y,z,w)
            - 'joint_pos': joint positions
        pred_state (dict): Predicted state with same structure.
    
    Returns:
        dict: Error dictionary containing:
            - 'err_pos': Euclidean distance between positions
            - 'err_quat': Rotation magnitude between quaternions (radians)
            - 'err_joint_pos': Euclidean distance between joint positions
    """
    err_pos = np.linalg.norm(gt_state["eef_pos"] - pred_state["eef_pos"])
    err_quat = (
        Rotation.from_quat(gt_state["eef_quat"])
        * Rotation.from_quat(pred_state["eef_quat"]).inv()
    ).magnitude()
    err_joint_pos = np.linalg.norm(
        gt_state["joint_pos"] - pred_state["joint_pos"]
    )
    state_err = dict(err_pos=err_pos, err_quat=err_quat, err_joint_pos=err_joint_pos)
    return state_err


def dynamic_time_warping(seq1, seq2, idx1=0, idx2=0, memo=None):
    """
    Align two state sequences using Dynamic Time Warping (DTW).
    
    Finds the optimal alignment between two sequences by allowing elements
    to be matched one-to-one or skipped, minimizing total alignment error.
    
    Args:
        seq1 (list[dict]): First state sequence (typically ground truth).
        seq2 (list[dict]): Second state sequence (typically predicted/recorded).
        idx1 (int, optional): Current index in seq1. Default: 0.
        idx2 (int, optional): Current index in seq2. Default: 0.
        memo (dict, optional): Memoization dictionary for caching results. Default: None.
    
    Returns:
        tuple: (total_error, aligned_indices) where:
            - total_error (float): Sum of state errors for the alignment
            - aligned_indices (list[int]): Indices from seq2 that align with seq1
    
    Notes:
        - Uses memoization for efficient dynamic programming.
        - Returns infinity if seq2 is exhausted before seq1.
        - Each element in seq1 must be matched to some element in seq2.
    """
    if memo is None:
        memo = {}

    if idx1 == len(seq1):
        return 0, []

    if idx2 == len(seq2):
        return float("inf"), []

    if (idx1, idx2) in memo:
        return memo[(idx1, idx2)]

    distance_with_current = total_state_err(compute_state_error(seq1[idx1], seq2[idx2]))
    error_with_current, subseq_with_current = dynamic_time_warping(
        seq1, seq2, idx1 + 1, idx2 + 1, memo
    )
    error_with_current += distance_with_current

    error_without_current, subseq_without_current = dynamic_time_warping(
        seq1, seq2, idx1, idx2 + 1, memo
    )

    if error_with_current < error_without_current:
        memo[(idx1, idx2)] = error_with_current, [idx2] + subseq_with_current
    else:
        memo[(idx1, idx2)] = error_without_current, subseq_without_current

    return memo[(idx1, idx2)]