import torch
"""
Layout of the J axis in every extracted (T, J, 3) array and (T, J) mask:
    [0                                   : NUM_POSE_JOINTS]                     pose joints
    [NUM_POSE_JOINTS                     : NUM_POSE_JOINTS + NUM_HAND_JOINTS]   left hand  (local idx 0 = wrist)
    [NUM_POSE_JOINTS + NUM_HAND_JOINTS   : NUM_POSE_JOINTS + NUM_HAND_JOINTS*2] right hand (local idx 0 = wrist)
"""
# Fixed order for pose joints
POSE_JOINT_INDICES =    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, # Face
                        11, 12,      # Shoulders
                        13, 14,      # Elbows
                        15, 16,      # Wrists
                        23, 24       # Hips
                        ]

# POSE_JOINT_INDICES = face(11) + shoulders(2) + elbows(2) + wrists(2) + hips(2)
NUM_POSE_JOINTS = len(POSE_JOINT_INDICES)
NUM_HAND_JOINTS = 21
NUM_JOINTS = NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2  # 61
 
LEFT_HAND_START = NUM_POSE_JOINTS                     # 19
RIGHT_HAND_START = NUM_POSE_JOINTS + NUM_HAND_JOINTS  # 40
 
# Local indices (within the pose slice, 0..NUM_POSE_JOINTS-1) of the wrists,
LEFT_WRIST_POSE_IDX = 15
RIGHT_WRIST_POSE_IDX = 16

LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12
LEFT_ELBOW_IDX, RIGHT_ELBOW_IDX = 13, 14
LEFT_WRIST_POSE_IDX = 15
RIGHT_WRIST_POSE_IDX = 16

# Note: the hip indices are actually 23 & 24 in the MediaPipe pose model.
# Here, the hip indices are concatenated to avoid the disconnection
LEFT_HIP_IDX, RIGHT_HIP_IDX = 17, 18
 
RAW_CHANNELS = 3     # x, y, z as stored on disk
MODEL_CHANNELS = 6   # position + velocity, produced inside STPoseModel by add_velocity()
 
# Matches extracted data
WINDOW_FRAMES = 60
WINDOW_DURATION_MS = 2000.0
FRAME_INTERVAL_MS = WINDOW_DURATION_MS / WINDOW_FRAMES

def build_skeleton_adjacency():
    """
    Build an adjacency matrix sized to `num_joints`.
    """

    adj = torch.zeros((NUM_JOINTS, NUM_JOINTS))

    pose_edges = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # Face
        (LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX), (LEFT_ELBOW_IDX, LEFT_WRIST_POSE_IDX),      # left arm
        (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX), (RIGHT_ELBOW_IDX, RIGHT_WRIST_POSE_IDX),  # right arm
        (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX),                                        # shoulders
        (LEFT_HIP_IDX, RIGHT_HIP_IDX),                                                  # hips
        (LEFT_SHOULDER_IDX, LEFT_HIP_IDX), (RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX),          # torso
    ]
    for i, j in pose_edges:
        if i < NUM_POSE_JOINTS and j < NUM_POSE_JOINTS:
            adj[i, j] = adj[j, i] = 1

    hand_edges = [
        (0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
    ]

    # Left hand offset
    offset_left = LEFT_HAND_START

    for i, j in hand_edges:
        if i < NUM_HAND_JOINTS and j < NUM_HAND_JOINTS:
            adj[offset_left + i, offset_left + j] = 1
            adj[offset_left + j, offset_left + i] = 1

    # Right hand offset
    offset_right = RIGHT_HAND_START

    for i, j in hand_edges:
        if i < NUM_HAND_JOINTS and j < NUM_HAND_JOINTS:
            adj[offset_right + i, offset_right + j] = 1
            adj[offset_right + j, offset_right + i] = 1

    # Connect pose to hand landmarks
    adj[offset_left, 15], adj[15, offset_left] = 1, 1
    adj[offset_right, 16], adj[16, offset_right] = 1, 1

    # self attention
    for i in range(NUM_JOINTS):
        adj[i, i] = 1

    return adj