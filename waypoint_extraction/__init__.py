"""
Waypoint Extraction Module.

This module provides tools for extracting waypoints from robot trajectories using
dynamic programming-based geometric methods. The main function is dp_waypoint_selection,
which finds the minimum set of waypoints needed to reconstruct a trajectory within
a specified error threshold.

Main Functions:
    - dp_waypoint_selection: Extract waypoints from a trajectory using DP optimization

Example:
    >>> from waypoint_extraction import dp_waypoint_selection
    >>> import numpy as np
    >>> positions = np.random.rand(100, 2)  # 100 timesteps, 2D positions
    >>> waypoints = dp_waypoint_selection(
    ...     actions=positions,
    ...     states=positions,
    ...     err_threshold=4.0,
    ...     pos_only=True
    ... )
    >>> print(f"Extracted {len(waypoints)} waypoints from 100 timesteps")
"""

from .extract_waypoints import dp_waypoint_selection

__all__ = ['dp_waypoint_selection']