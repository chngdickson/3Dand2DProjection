import torch

def project_2d_to_3d_torch(points_2d, depths, K, extrinsic):
    """
    Projects 2D image points to 3D world coordinates.
    
    Args:
        points_2d: (N, 2) or (B, N, 2) tensor of image coordinates (u, v)
        depths: (N,) or (B, N) tensor of depth values (Z in camera space)
        K: (3, 3) or (B, 3, 3) camera intrinsic matrix
        extrinsic: (4, 4) or (B, 4, 4) camera extrinsic matrix [R | t]
    
    Returns:
        points_3d: (N, 3) or (B, N, 3) 3D points in world coordinates
    """
    # Ensure tensors are batched consistently
    if points_2d.dim() == 2:
        K = K.unsqueeze(0)  # (1, 3, 3)
        extrinsic = extrinsic.unsqueeze(0)  # (1, 4, 4)
        points_2d = points_2d.unsqueeze(0)  # (1, N, 2)
        depths = depths.unsqueeze(0)  # (1, N)
    
    # Convert to homogeneous coordinates (B, N, 3)
    points_2d_hom = torch.cat([
        points_2d, 
        torch.ones_like(points_2d[..., :1])
    ], dim=-1)
    
    # Compute inverse intrinsics (B, 3, 3)
    K_inv = torch.inverse(K)
    
    # Transform to camera space (B, N, 3)
    points_cam = torch.matmul(points_2d_hom, K_inv.transpose(-1, -2))  # (B, N, 3)
    points_cam = points_cam * depths.unsqueeze(-1)  # Scale by depth
    
    # Convert to homogeneous (B, N, 4)
    points_cam_hom = torch.cat([
        points_cam,
        torch.ones_like(points_cam[..., :1])
    ], dim=-1)
    
    # Transform to world coordinates (B, N, 4)
    points_world_hom = torch.matmul(points_cam_hom, extrinsic.transpose(-1, -2))
    
    # Return 3D coordinates (B, N, 3)
    points_3d = points_world_hom[..., :3]
    
    return points_3d.squeeze(0) if points_2d.dim() == 3 else points_3d

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example data (single image)
    points_2d = torch.tensor([
        [320, 240],  # Principal point
        [370, 240],  # Right-shifted point
    ], dtype=torch.float32, device=device)
    
    depths = torch.tensor([2.5, 3.0], device=device)  # Depth values
    
    K = torch.tensor([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    extrinsic = torch.tensor([
        [1, 0, 0, 0.1],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.5],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # Project to 3D
    points_3d = project_2d_to_3d_torch(points_2d, depths, K, extrinsic)
    print("3D Points:\n", points_3d.cpu().numpy())