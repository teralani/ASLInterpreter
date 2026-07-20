import torch

def build_skeleton_adjacency(num_joints:int = 59):
    """
    Build an adjacency matrix sized to `num_joints`.

    - Assumes MediaPipe pose uses 33 pose keypoints.
    - The remaining joints are split equally between left/right hands.
    - Hand-edge templates are applied only where indices exist (safe truncation).
    """
    NUM_POSE = 33

    if num_joints < NUM_POSE:
        raise ValueError(f"num_joints ({num_joints}) must be >= {NUM_POSE}")

    # compute hand size per side (integer division)
    NUM_HAND = (num_joints - NUM_POSE) // 2
    NUM_JOINTS = num_joints

    adj = torch.zeros((NUM_JOINTS, NUM_JOINTS))

    # -------------------------------
    # Pose edges (MediaPipe Pose)
    # -------------------------------
    pose_edges = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 12),            # shoulders
        (23, 24),            # hips
        (11, 23), (12, 24),  # torso
    ]

    for i, j in pose_edges:
        if i < NUM_POSE and j < NUM_POSE:
            adj[i, j] = adj[j, i] = 1

    # -------------------------------
    # Hand edges template (up to 21 indices)
    # Only apply edges where the hand has those indices.
    # -------------------------------
    hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9,10), (10,11), (11,12),      # middle
        (0,13), (13,14), (14,15), (15,16),     # ring
        (0,17), (17,18), (18,19), (19,20),     # pinky
    ]

    # Left hand offset
    offset_left = NUM_POSE

    for i, j in hand_edges:
        if i < NUM_HAND and j < NUM_HAND:
            adj[offset_left + i, offset_left + j] = 1
            adj[offset_left + j, offset_left + i] = 1

    # Right hand offset
    offset_right = NUM_POSE + NUM_HAND

    for i, j in hand_edges:
        if i < NUM_HAND and j < NUM_HAND:
            adj[offset_right + i, offset_right + j] = 1
            adj[offset_right + j, offset_right + i] = 1

    return adj