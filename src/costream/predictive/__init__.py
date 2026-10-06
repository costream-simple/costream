"""Predictive behavior: generated-video point tracks -> task-frame nominal trajectory."""

from .scale import resolve_metric_scale
from .tracks import TrackSet, load_spatracker_npz, project
from .lift import (centroid_motion, deviation_from_waypoints, lift_to_task_frame, rigid_motion,
                   select_tracks)
