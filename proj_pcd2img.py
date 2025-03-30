import torch
import numpy as np
from copy import deepcopy

def K3x3_to_4x4(K_3x3):
    """Convert a 3x3 intrinsic matrix to a 4x4 matrix."""
    if isinstance(K_3x3,np.ndarray):
        K_4x4 = np.eye(4) 
        K_4x4[:2, :3] = K_3x3[:2, :3]  # Copy first two rows
        K_4x4[2, :3] = K_3x3[2, :3]    # Copy third row
    elif isinstance(K_3x3, torch.Tensor):
        device = K_3x3.device
        dtype = K_3x3.dtype
        if K_3x3.dim() == 2:
            # Single Batch Case (3x3)
            K_4x4 = torch.eye(4, dtype=dtype, device=device) 
            K_4x4[:2, :3] = K_3x3[:2, :3]  # Copy first two rows
            K_4x4[2, :3] = K_3x3[2, :3]    # Copy third row
        else:
            # Batched Case (B, 3, 3)
            B = K_3x3.shape[0]
            if B == 1:
                K_4x4 = torch.eye(4, dtype=dtype, device=device).unsqueeze(0)
            else:
                K_4x4 = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(B).clone()
            print(K_4x4.shape)
            K_4x4[:, :2, :3] = K_3x3[:,:2, :3]  # Copy first two rows
            K_4x4[:,2, :3] = K_3x3[:,2, :3]    # Copy third row
    else:
        raise NotImplementedError
    return K_4x4

def project_3d_to_2d_np(points_3d, K, extrinsic):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points (X, Y, Z).
        K: (3, 3) intrinsic matrix.
        extrinsic: (4, 4) extrinsic matrix [R | t].
    
    Returns:
        points_2d: (N, 2) array of 2D image coordinates (u, v).
        depths: (N,) array of depths (Z in camera space).
    """
    N = points_3d.shape[0]
    points_hom = np.hstack([points_3d, np.ones((N, 1))])
    P = deepcopy(K3x3_to_4x4(K) @ extrinsic)
    points_2d =  points_hom @ P.T
    depths = points_2d[:, 2]
    points_2d = points_2d[:, :2] / depths[:, None]
    return points_2d

def project_3d_to_2d_torch(points_3d, K, extrinsic):
    """
    PyTorch version of 3D-to-2D projection with batched operations.
    
    Args:
        points_3d: (N, 3) or (B, N, 3) tensor of 3D points
        K: (3, 3) or (B, 3, 3) camera intrinsic matrix
        extrinsic: (4, 4) or (B, 4, 4) camera extrinsic matrix
    
    Returns:
        points_2d: (N, 2) or (B, N, 2) projected 2D coordinates
        depths: (N,) or (B, N) depths in camera space
    """
    device = points_3d.device
    dtype = points_3d.dtype
    K = torch.tensor(K, dtype=dtype, device=device) if isinstance(K, np.ndarray) else K.clone().detach().to(dtype=dtype, device=device)
    extrinsic = torch.tensor(extrinsic, dtype=dtype, device=device) if isinstance(extrinsic, np.ndarray) else extrinsic.clone().detach().to(dtype=dtype, device=device)
    if points_3d.dim() == 2:
        # Single batch case (N, 3)
        P = K3x3_to_4x4(K) @ extrinsic  
        points_hom = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)  # (N, 4)
        points_2d_hom = points_hom @ P.T
    else:
        # Batched case (B, N, 3)
        P =  K3x3_to_4x4(K) @ extrinsic  # (B, 3, 4)
        points_hom = torch.cat([points_3d, torch.ones_like(points_3d[:, :, :1])], dim=-1)  # (B, N, 4)
        points_2d_hom = torch.matmul(points_hom, P.transpose(-1, -2)) # (B, N, 3)
    
    depths = points_2d_hom[..., 2]
    points_2d = points_2d_hom[..., :2] / depths.unsqueeze(-1).clamp(min=1e-6)
    del P, points_hom, points_2d_hom
    return points_2d

def project_3d_to_2d(points_3d, K, extrinsic):
    if isinstance(points_3d,np.ndarray):
        return project_3d_to_2d_np(points_3d, K, extrinsic).astype(np.int32)
    elif isinstance(points_3d, torch.Tensor):
        return project_3d_to_2d_torch(points_3d, K, extrinsic).long()
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    K = np.array([
        [500, 0, 320],  # fx, skew, cx
        [0, 500, 240],   # fy, cy
        [0, 0, 1]        # Homogeneous scaling
    ])
    
    # Extrinsic matrix (4x4) [R | t]
    extrinsic = np.array([
        [1, 0, 0, 0.1],  # Translation in X
        [0, 1, 0, 0.2],  # Translation in Y
        [0, 0, 1, 0.5],  # Translation in Z
        [0, 0, 0, 1]     # Homogeneous row
    ])
    
    # 3D points (N, 3)
    points_3d = np.array([
        [1, 0, 2],
        [0, 1, 3],
        [-1, 0, 5]
    ])
    
    # Test Numpy
    pcd_uv = project_3d_to_2d(points_3d, K, extrinsic)
    print(pcd_uv[:,0].max(), pcd_uv[:,1].max())

    # Test Torch
    # (N, 3)
    pcd_uv = project_3d_to_2d(torch.tensor(points_3d), K, extrinsic)
    print(pcd_uv[:,0].max(), pcd_uv[:,1].max())
    
    # (B, N, 3)
    pcd_uv = project_3d_to_2d(torch.tensor(points_3d).unsqueeze(0), torch.tensor(K).unsqueeze(0), torch.tensor(extrinsic).unsqueeze(0))
    print(pcd_uv[:,:,0].max(), pcd_uv[:,:,1].max())
    