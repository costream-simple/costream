"""Reactive behavior: tactile frames -> bounded body-frame corrections."""

from .maps import TactileMaps, TactileReconstructor, erode_contact, gradient_to_normal
from .registration import InsufficientOverlapError, register
from .tracker import KeyframeTracker, TrackResult
from .compensation import SlipCompensator, TactilePipeline
