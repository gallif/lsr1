import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PinholeCamera(nn.Module):
    def __init__(self, fx, fy, cx, cy, R, t):
        super().__init__()

        K = np.asarray([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]], dtype=np.float32)
        Rt = np.asarray([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                         [R[1, 0], R[1, 1], R[1, 2], t[1]],
                         [R[2, 0], R[2, 1], R[2, 2], t[2]]], dtype=np.float32)
        P = K @ Rt
        self.register_buffer("P", torch.from_numpy(P), persistent=False)

    def forward(self, x):
        x = F.pad(x, (0, 1), "constant", 1)
        proj = torch.einsum("bij,kj->bik", x, self.P)
        proj[:, :, 0] *= -1
        return proj[:, :, :2] / proj[:, :, 2, None]
    

def project_3d_to_2d(joints_3d, camera):
    """
    Project 3D joints to 2D using camera parameters.
    
    Args:
        joints_3d (numpy.ndarray): (N, 3) array of 3D joints in world coordinates.
        R (numpy.ndarray): (3, 3) Rotation matrix.
        t (numpy.ndarray): (3,) Translation vector.
        f (tuple): (fx, fy) Focal length.
        c (tuple): (cx, cy) Principal point.
    
    Returns:
        numpy.ndarray: (N, 2) array of projected 2D points.
    """
    
    joints_3d = np.array(joints_3d).reshape(3,-1).T
    R = np.array(camera['R'])
    t = np.array(camera['t'])
    f = np.array(camera['f'])
    c = np.array(camera['c'])
    
    # Convert to camera coordinates: X_c = R * X_w + t
    joints_cam = (R @ joints_3d.T).T + t  # Shape: (N, 3)

    # Perspective division (normalize by depth Z_c)
    X_c, Y_c, Z_c = joints_cam[:, 0], joints_cam[:, 1], joints_cam[:, 2]
    x_norm = X_c / Z_c
    y_norm = Y_c / Z_c

    # Convert to pixel coordinates
    u = f[0] * x_norm + c[0]
    v = f[1] * y_norm + c[1]

    return np.stack([u, v], axis=-1)  # Shape: (N, 2)    
