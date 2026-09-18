import torch
"""
Layout of the J axis in every extracted (T, J, 3) array and (T, J) mask:
    [0                                   : NUM_POSE_JOINTS]                     pose joints
    [NUM_POSE_JOINTS                     : NUM_POSE_JOINTS + NUM_HAND_JOINTS]   left hand  (local idx 0 = wrist)
    [NUM_POSE_JOINTS + NUM_HAND_JOINTS   : NUM_POSE_JOINTS + NUM_HAND_JOINTS*2] right hand (local idx 0 = wrist)
"""
POSE_JOINT_INDICES =    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, # face
                        11, 12,      # shoulders
                        13, 14,      # elbows
                        15, 16,      # wrists
                        23, 24       # hips
                        ]

# POSE_JOINT_INDICES = face(11) + shoulders(2) + elbows(2) + wrists(2) + hips(2)
NUM_POSE_JOINTS = len(POSE_JOINT_INDICES)
NUM_HAND_JOINTS = 21
NUM_JOINTS = NUM_POSE_JOINTS + NUM_HAND_JOINTS * 2  # 61
 
LEFT_HAND_START = NUM_POSE_JOINTS                     # 19
RIGHT_HAND_START = NUM_POSE_JOINTS + NUM_HAND_JOINTS  # 40
 
# Local indices for wrists
LEFT_WRIST_POSE_IDX = 15
RIGHT_WRIST_POSE_IDX = 16

LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12
LEFT_ELBOW_IDX, RIGHT_ELBOW_IDX = 13, 14
LEFT_WRIST_POSE_IDX = 15
RIGHT_WRIST_POSE_IDX = 16

# hip indices concatenated (usually 23 & 24)
LEFT_HIP_IDX, RIGHT_HIP_IDX = 17, 18
 
RAW_CHANNELS = 3     # x, y, z
MODEL_CHANNELS = 12   # position + velocity
 
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
    adj.fill_diagonal_(1)

    return adj

def build_left_right_swap():
    """
    Build the joint mapping used for left-right mirror augmentation.

    Face and pose joints are swapped by side.
    Left and right hand blocks are swapped joint-for-joint. 
    The x-coordinate is negated separately.
    """
    swap = list(range(NUM_JOINTS))
 
    face_swap_pairs = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)]
    for a, b in face_swap_pairs:
        swap[a], swap[b] = b, a
 
    limb_swap_pairs = [
        (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX),
        (LEFT_ELBOW_IDX, RIGHT_ELBOW_IDX),
        (LEFT_WRIST_POSE_IDX, RIGHT_WRIST_POSE_IDX),
        (LEFT_HIP_IDX, RIGHT_HIP_IDX),
    ]
    for a, b in limb_swap_pairs:
        swap[a], swap[b] = b, a
 
    for k in range(NUM_HAND_JOINTS):
        left_idx = LEFT_HAND_START + k
        right_idx = RIGHT_HAND_START + k
        swap[left_idx], swap[right_idx] = right_idx, left_idx
 
    return swap

def build_bone_parents():
    """
    Returns a LongTensor (NUM_JOINTS,) where entry j is the parent joint index used in bone[j] = x[j] - x[parent[j]].
    Root joints point to themselves.

    Tree (root = RIGHT_HIP_IDX):
      RIGHT_HIP (root)
        LEFT_HIP
        LEFT_SHOULDER  <- LEFT_HIP
          LEFT_ELBOW -> LEFT_WRIST_POSE -> left hand root -> finger chains
          face root (nose) -> face chains
        RIGHT_SHOULDER <- RIGHT_HIP
          RIGHT_ELBOW -> RIGHT_WRIST_POSE -> right hand root -> finger chains
    """
    parent = list(range(NUM_JOINTS)) # default: self (root)

    # torso
    parent[LEFT_HIP_IDX] = RIGHT_HIP_IDX
    parent[LEFT_SHOULDER_IDX] = LEFT_HIP_IDX
    parent[RIGHT_SHOULDER_IDX] = RIGHT_HIP_IDX

    # arms
    parent[LEFT_ELBOW_IDX] = LEFT_SHOULDER_IDX
    parent[RIGHT_ELBOW_IDX] = RIGHT_SHOULDER_IDX
    parent[LEFT_WRIST_POSE_IDX] = LEFT_ELBOW_IDX
    parent[RIGHT_WRIST_POSE_IDX] = RIGHT_ELBOW_IDX

    # left chain: 0-1-2-3-7, right chain: 0-4-5-6-8, mouth: 0-9-10
    parent[0] = LEFT_SHOULDER_IDX  # attach face root to body
    for child, p in [(1, 0), (2, 1), (3, 2), (7, 3),
                     (4, 0), (5, 4), (6, 5), (8, 6),
                     (9, 0), (10, 9)]:
        parent[child] = p

    # attach to standard 5 finger chains
    finger_chains = [
        (1, 2, 3, 4),      # thumb
        (5, 6, 7, 8),      # index
        (9, 10, 11, 12),   # middle
        (13, 14, 15, 16),  # ring
        (17, 18, 19, 20),  # pinky
    ]
    for offset, pose_wrist in [(LEFT_HAND_START, LEFT_WRIST_POSE_IDX),
                                (RIGHT_HAND_START, RIGHT_WRIST_POSE_IDX)]:
        parent[offset] = pose_wrist  # hand wrist <- pose wrist
        for chain in finger_chains:
            prev = offset  # hand wrist local 0
            for local_idx in chain:
                parent[offset + local_idx] = prev
                prev = offset + local_idx

    return torch.tensor(parent, dtype=torch.long)


BONE_PARENTS = build_bone_parents()