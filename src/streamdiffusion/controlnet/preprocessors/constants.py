"""Shared constant mappings and colour tables for MediaPipe-based preprocessors.

Keeping these heavy structures in a separate module avoids re-parsing them for
 every process import of the main preprocessing classes and keeps those files
 more readable.
"""
from __future__ import annotations

# MediaPipe pose (33-kp) → OpenPose body (25-kp) mapping
MEDIAPIPE_TO_OPENPOSE_MAP = {
    # 0 = Nose is identical; 1 = Neck is derived later, etc.
    1: None,  # Neck (calculated from shoulders)
    2: 12,  # RShoulder → RightShoulder
    3: 14,  # RElbow → RightElbow
    4: 16,  # RWrist → RightWrist
    5: 11,  # LShoulder → LeftShoulder
    6: 13,  # LElbow → LeftElbow
    7: 15,  # LWrist → LeftWrist
    8: None,  # MidHip (calculated from hips)
    9: 24,  # RHip → RightHip
    10: 26,  # RKnee → RightKnee
    11: 28,  # RAnkle → RightAnkle
    12: 23,  # LHip → LeftHip
    13: 25,  # LKnee → LeftKnee
    14: 27,  # LAnkle → LeftAnkle
    19: 31,  # LBigToe → LeftFootIndex
    20: 31,  # LSmallToe → LeftFootIndex (approx.)
    21: 29,  # LHeel → LeftHeel
    22: 32,  # RBigToe → RightFootIndex
    23: 32,  # RSmallToe → RightFootIndex (approx.)
    24: 30,  # RHeel → RightHeel
}

# OpenPose limb pairs used for body skeleton rendering
OPENPOSE_LIMB_SEQUENCE = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
    [13, 14], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24],
]

# Standard OpenPose colours (BGR)
OPENPOSE_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 0], [255, 85, 0],
    [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
]

# OpenPose face (70-kp) connection pairs
OPENPOSE_FACE_CONNECTIONS = [
    # Jawline
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Left & Right eyebrows
    (17, 18), (18, 19), (19, 20), (20, 21),
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nose bridge / lower
    (27, 28), (28, 29), (29, 30),
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Eyes
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Lips outer
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
    # Lips inner
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
   # Pupils (68-69)
    (68, 68), (69, 69)
]    

# Colour per face-connection group (BGR)
FACE_COLORS = list(
    [(255, 255, 255)] * 16 +  # jaw
    [(0, 255, 0)] * 4 +      # right brow
    [(0, 255, 0)] * 4 +      # left brow
    [(255, 0, 255)] * 3 +    # nose bridge
    [(255, 0, 255)] * 4 +    # nose lower
    [(0, 0, 255)] * 6 +      # right eye
    [(0, 0, 255)] * 6 +      # left eye
    [(255, 0, 0)] * 12 +     # outer lips
    [(255, 0, 0)] * 8 +      # inner lips
    [(255, 0, 0)] * 2        # pupils
)

# MediaPipe 468-kp → OpenPose 70-kp face mapping
MEDIAPIPE_TO_OPENPOSE_FACE_MAP = {
    # Jawline
    0: 127, 1: 234, 2: 93, 3: 132, 4: 58, 5: 172, 6: 136, 7: 150,
    8: 152, 9: 400, 10: 365, 11: 397, 12: 435, 13: 401, 14: 323, 15: 454, 16: 356,
    # Eyebrows
    17: 55, 18: 65, 19: 52, 20: 53, 21: 46,
    22: 285, 23: 295, 24: 282, 25: 283, 26: 276,
    # Nose
    27: 168, 28: 197, 29: 5, 30: 4,
    31: 166, 32: 44, 33: 19, 34: 457, 35: 455,
    # Eyes
    36: 33, 37: 160, 38: 158, 39: 155, 40: 145, 41: 163,
    42: 463, 43: 385, 44: 388, 45: 263, 46: 373, 47: 381,
    # Lips outer
    48: 185, 49: 39, 50: 37, 51: 0, 52: 267, 53: 270, 54: 409,
    55: 321, 56: 314, 57: 17, 58: 181, 59: 146,
    # Lips inner
    60: 78, 61: 81, 62: 13, 63: 311, 64: 409, 65: 402, 66: 14, 67: 178,
    # Pupils
    68: 468, 69: 473,
}
