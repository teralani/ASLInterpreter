import torch

def build_skeleton_adjacency(num_joints:int = 51):
    """
    Docstring for build_skeleton_adjacency
    
    :args num_joints: The total number of joints in the adjacency matrix
    :type num_joints: int

    :returns An adjacency matrix with dimensions of num_joints x num_joints:
    """
    adj = torch.zeros(num_joints, num_joints)

    # Pose (Mediapipe)

    POSE_EDGES = [
        (11, 13), (13, 15), # left arm
        (12, 14), (14, 16), # right arm
        (11, 12),           # shoulders
        (23, 24),           # hips
    ]

    HAND_EDGES = [(i, i+1) for i in range(0, 20)]

    # Combined graph is as such: [Pose (17) | Left Hand (21) | Right Hand (21)]

    # Starting index for left and right hands:
    offset_left = 17
    offset_right = 38

    # In the 2D adjacency matrix, 1 is a connection and 0 is a lack of one

    for i, j in POSE_EDGES:
        adj[i, j] = adj[j, i] = 1

    for i, j in HAND_EDGES:
        adj[offset_left + i, offset_left + j] = 1
        adj[offset_right + i, offset_right + j] = 1

    return adj