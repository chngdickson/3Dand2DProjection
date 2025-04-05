import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
def rasterize_3dto2D(
    pointcloud, 
    mask_2d: torch.Tensor=None, 
    img_shape: tuple=None,
    min_xyz: tuple=None,  # (min_x, min_y, min_z)
    max_xyz: tuple=None,  # (max_x, max_y, max_z)
    axis='z', 
    highest_first=True,
    depth_weighting=True  
):
    """
    Rasterize point cloud with explicit bounds and depth-based weighting.
    
    Args:
        pointcloud: (N, 3) tensor of 3D points.
        mask_2d: (H, W) binary mask (True = keep point).
        img_shape: (H, W) if mask_2d is not None
        min_xyz: Tuple of (min_x, min_y, min_z) bounds.
        max_xyz: Tuple of (max_x, max_y, max_z) bounds.
        axis: 'xy', 'xz', or 'yz' (projection plane).
        highest_first: If True, prioritize farthest points; else closest.
        depth_weighting: If True, farther points have lower values.
    
    Returns:
        filtered_pointcloud: (M, 3) tensor (M <= N).
        raster_image: (H, W) float tensor with depth-weighted values (0-1).
        raster_filtered_img: (H, W) float tensor of masked points with depth weighting.
    """
    if isinstance(pointcloud, torch.Tensor):
        filtered_pointcloud, raster_image, raster_filtered_img = rasterize_3dto2D_torch(
            pointcloud=pointcloud,
            mask_2d=mask_2d,
            img_shape=img_shape,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            axis=axis,
            highest_first=highest_first,
            depth_weighting=depth_weighting
        )
        return filtered_pointcloud.detach().cpu(), raster_image.detach().cpu().numpy(), raster_filtered_img.detach().cpu().numpy()
    elif isinstance(pointcloud, np.ndarray):
        filtered_pointcloud, raster_image, raster_filtered_img =  rasterize_3dto2D_numpy(
            pointcloud=pointcloud,
            mask_2d=mask_2d,
            img_shape=img_shape,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            axis=axis,
            highest_first=highest_first,
            depth_weighting=depth_weighting
        )
        return filtered_pointcloud, raster_image, raster_filtered_img
    else:
        assert NotImplementedError
    
def rasterize_3dto2D_torch(
    pointcloud, 
    mask_2d: np.ndarray=None, 
    img_shape: tuple=None,
    min_xyz: tuple=None,  # (min_x, min_y, min_z)
    max_xyz: tuple=None,  # (max_x, max_y, max_z)
    axis='z', 
    highest_first=True,
    depth_weighting=True
):
    """
    Rasterize point cloud with explicit bounds and depth-based weighting.
    Higher values → Red
    Mid values → Blue
    Lowest values → Green
    
    Args:
        pointcloud: (N, 3) array of 3D points.
        mask_2d: (H, W) binary mask (True = keep point).
        img_shape: (H, W) if mask_2d is not None.
        min_xyz: Tuple of (min_x, min_y, min_z) bounds.
        max_xyz: Tuple of (max_x, max_y, max_z) bounds.
        axis: 'xy', 'xz', or 'yz' (projection plane).
        highest_first: If True, prioritize farthest points; else closest.
        depth_weighting: If True, farther points have lower values.
    
    Returns:
        filtered_pointcloud: (M, 3) array (M <= N).
        raster_image: (H, W) binary | (H, W, 3) RGB image (Red=High, Blue=Mid, Green=Low).
        raster_filtered_img: (H, W) binary of masked points | (H, W, 3) RGB image of masked points.
    """
    assert not(mask_2d is None and img_shape is None), "mask_2d or img_shape must be present for rasterization"
    device = pointcloud.device
    dtype = pointcloud.dtype
    if mask_2d is not None:
        H, W = mask_2d.shape
        if isinstance(mask_2d,np.ndarray):
            mask_2d = torch.tensor(mask_2d, dtype=torch.bool, device=device)
    else:
        H, W = img_shape
    
    xyz = pointcloud[:,:3]
    if axis == 'z':
        depth = pointcloud[:, 2]  # Z-axis for XY projection
        coords = xyz[:, :2]
        min_coord = torch.tensor([min_xyz[0], min_xyz[1]], device=device) if min_xyz is not None else coords.min(dim=0)[0]
        max_coord = torch.tensor([max_xyz[0], max_xyz[1]], device=device) if max_xyz is not None else coords.max(dim=0)[0]
    elif axis == 'y':
        depth = pointcloud[:, 1]  # Y-axis for XZ projection
        coords = xyz[:, [0, 2]]
        coords[:,1] = coords[:,1] * -1
        min_coord = torch.tensor([min_xyz[0], min_xyz[2]], device=device) if min_xyz is not None else coords.min(dim=0)[0]
        max_coord = torch.tensor([max_xyz[0], max_xyz[2]], device=device) if max_xyz is not None else coords.max(dim=0)[0]
    elif axis == 'x':
        depth = pointcloud[:, 0]  # X-axis for YZ projection
        coords = xyz[:, [1, 2]]
        coords[:,1] = coords[:,1] * -1
        min_coord = torch.tensor([min_xyz[1], min_xyz[2]], device=device) if min_xyz is not None else coords.min(dim=0)[0]
        max_coord = torch.tensor([max_xyz[1], max_xyz[2]], device=device) if max_xyz is not None else coords.max(dim=0)[0]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # Normalize to [0, 1] 
    coords_normalized = (coords - min_coord) / (max_coord - min_coord + 1e-6)
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    if highest_first:
        norm_depth = 1.0 - norm_depth
    
    # Scale to mask dimensions and round to integer indices
    u = (coords_normalized[:, 0] * (W - 1)).long()
    v = (coords_normalized[:, 1] * (H - 1)).long()
    
    # --- Step 3: Filter points using the mask ---
    # Raster_filtered use sorted, 
    valid_within_bounds = (0 <= u) & (u < W) & (0 <= v) & (v < H)
    if mask_2d is None:
        valid_within_bounds_n_mask = valid_within_bounds.nonzero().squeeze(1)
    else:
        valid_within_bounds_n_mask = torch.zeros_like(u, dtype=torch.bool)
        valid_within_bounds_n_mask[valid_within_bounds.nonzero().squeeze(1)] = mask_2d[v[valid_within_bounds],u[valid_within_bounds]]
        valid_within_bounds_n_mask = valid_within_bounds_n_mask.nonzero().squeeze(1)
    
    filtered_pointcloud = pointcloud[valid_within_bounds_n_mask]
    # Apply depth weighting if enabled
    if depth_weighting:
        cmap = plt.get_cmap('rainbow')
        raster_image = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        raster_filtered_img = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        
        
        # --- Filter points within bounds ---
        u_valid, v_valid = u[valid_within_bounds], v[valid_within_bounds]
        norm_depth_valid = norm_depth[valid_within_bounds]
        
        # --- Filter points within bounds and 2dMask ---
        u_valid_mask, v_valid_mask = u[valid_within_bounds_n_mask], v[valid_within_bounds_n_mask]
        norm_depth_valid_mask = norm_depth[valid_within_bounds_n_mask]
        
        # --- Vectorized occlusion handling within bounds---
        # Sort by highest first        
        _ ,unique_indices = torch.unique(v_valid * W + u_valid, return_inverse=True, return_counts=False, dim=0)
        u_valid, v_valid = u_valid[unique_indices], v_valid[unique_indices]
        raster_image[v_valid, u_valid] = torch.tensor(
                cmap(norm_depth_valid[unique_indices].detach().cpu().numpy())[:, :3]*255, dtype=torch.uint8, device=device
            )
        
        # --- Vectorized occlusion handling within bounds and mask---
        if mask_2d is None:
            raster_filtered_img = raster_image.clone()
        else:
            # Find the first occurence of each pixel
            _, unique_indices = torch.unique(v_valid_mask * W + u_valid_mask, return_inverse=True, return_counts=False, dim=0)
            u_valid_mask, v_valid_mask = u_valid_mask[unique_indices], v_valid_mask[unique_indices]
            raster_filtered_img[v_valid_mask, u_valid_mask] = torch.tensor(
                cmap(norm_depth_valid_mask[unique_indices].detach().cpu().numpy())[:, :3] *255, dtype=torch.uint8, device=device
            )
    else:
        # Binary version (original behavior)
        raster_image = torch.zeros((H, W), dtype=torch.bool, device=device)
        raster_filtered_img = torch.zeros((H, W), dtype=torch.bool, device=device)
        raster_image[v[valid_within_bounds], u[valid_within_bounds]] = True
        if mask_2d is None:
            raster_filtered_img = raster_image.clone()
        else:
            raster_filtered_img[v[valid_within_bounds_n_mask], u[valid_within_bounds_n_mask]] = True
    
    del u_valid_mask, v_valid_mask, u_valid, v_valid, valid_within_bounds, valid_within_bounds_n_mask
    del u, v, unique_indices, coords_normalized, norm_depth
    del depth, min_coord, max_coord, xyz, coords
    return filtered_pointcloud, raster_image, raster_filtered_img


def rasterize_3dto2D_numpy(
    pointcloud, 
    mask_2d: np.ndarray=None, 
    img_shape: tuple=None,
    min_xyz: tuple=None,  # (min_x, min_y, min_z)
    max_xyz: tuple=None,  # (max_x, max_y, max_z)
    axis='z', 
    highest_first=True,
    depth_weighting=True
):
    """
    Rasterize point cloud with explicit bounds and depth-based weighting.
    Higher values → Red
    Mid values → Blue
    Lowest values → Green
    
    Args:
        pointcloud: (N, 3) array of 3D points.
        mask_2d: (H, W) binary mask (True = keep point).
        img_shape: (H, W) if mask_2d is not None.
        min_xyz: Tuple of (min_x, min_y, min_z) bounds.
        max_xyz: Tuple of (max_x, max_y, max_z) bounds.
        axis: 'xy', 'xz', or 'yz' (projection plane).
        highest_first: If True, prioritize farthest points; else closest.
        depth_weighting: If True, farther points have lower values.
    
    Returns:
        filtered_pointcloud: (M, 3) array (M <= N).
        raster_image: (H, W) binary | (H, W, 3) RGB image (Red=High, Blue=Mid, Green=Low).
        raster_filtered_img: (H, W) binary of masked points | (H, W, 3) RGB image of masked points.
    """
    assert not(mask_2d is None and img_shape is None), "mask_2d or img_shape must be present for rasterization"
    if mask_2d is not None:
        H, W = mask_2d.shape
    else:
        H, W = img_shape
    
    xyz = pointcloud[:,:3]
    if axis == 'z':
        depth = pointcloud[:, 2]  # Z-axis for XY projection
        coords = xyz[:, :2]
        min_coord = np.array([min_xyz[0], min_xyz[1]]) if min_xyz is not None else coords.min(axis=0)
        max_coord = np.array([max_xyz[0], max_xyz[1]]) if max_xyz is not None else coords.max(axis=0)
    elif axis == 'y':
        depth = pointcloud[:, 1]  # Y-axis for XZ projection
        coords[:,1] = coords[:,1] * -1
        min_coord = np.array([min_xyz[0], min_xyz[2]]) if min_xyz is not None else coords.min(axis=0)
        max_coord = np.array([max_xyz[0], max_xyz[2]]) if max_xyz is not None else coords.max(axis=0)
    elif axis == 'x':
        depth = pointcloud[:, 0]  # X-axis for YZ projection
        coords[:,1] = coords[:,1] * -1
        min_coord = np.array([min_xyz[1], min_xyz[2]]) if min_xyz is not None else coords.min(axis=0)
        max_coord = np.array([max_xyz[1], max_xyz[2]]) if max_xyz is not None else coords.max(axis=0)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # Normalize to [0, 1] 
    coords_normalized = (coords - min_coord) / (max_coord - min_coord + 1e-6)
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    if highest_first:
        norm_depth = 1.0 - norm_depth
    
    # Scale to mask dimensions and round to integer indices
    u = (coords_normalized[:, 0] * (W - 1)).astype(int)
    v = (coords_normalized[:, 1] * (H - 1)).astype(int)
    
    # --- Step 3: Filter points using the mask ---
    # Raster_filtered use sorted, 
    valid_within_bounds = (0 <= u) & (u < W) & (0 <= v) & (v < H)
    if mask_2d is None:
        valid_within_bounds_n_mask = valid_within_bounds.nonzero()[0]
    else:
        valid_within_bounds_n_mask = np.zeros_like(u)
        valid_within_bounds_n_mask[valid_within_bounds.nonzero()] = mask_2d[v[valid_within_bounds],u[valid_within_bounds]]
        valid_within_bounds_n_mask = valid_within_bounds_n_mask.nonzero()[0]
    
    filtered_pointcloud = pointcloud[valid_within_bounds_n_mask]
    # Apply depth weighting if enabled
    if depth_weighting:
        cmap = plt.get_cmap('rainbow')
        raster_image = np.zeros((H, W, 3), dtype=np.uint8)
        raster_filtered_img = np.zeros((H, W,3), dtype=np.uint8)
        
        
        # --- Filter points within bounds ---
        u_valid, v_valid = u[valid_within_bounds], v[valid_within_bounds]
        norm_depth_valid = norm_depth[valid_within_bounds]
        
        # --- Filter points within bounds and 2dMask ---
        u_valid_mask, v_valid_mask = u[valid_within_bounds_n_mask], v[valid_within_bounds_n_mask]
        norm_depth_valid_mask = norm_depth[valid_within_bounds_n_mask]
        
        # --- Vectorized occlusion handling within bounds---
        pixel_keys = v_valid * W + u_valid
        
        # 1. Sort by highest first
        sort_order = np.lexsort((pixel_keys, -norm_depth_valid)) 
        u_valid, v_valid = u_valid[sort_order], v_valid[sort_order]
        norm_depth_valid = norm_depth_valid[sort_order]
        
        # 2. Find the first occurence of each pixel
        _, unique_indices = np.unique(v_valid * W + u_valid, return_index=True)
        u_valid, v_valid = u_valid[unique_indices], v_valid[unique_indices]
        
        # 3. Finally Plot the colors
        raster_image[v_valid, u_valid] = cmap(norm_depth_valid[unique_indices])[:, :3] * 255
        
        # --- Vectorized occlusion handling within bounds and mask---
        if mask_2d is None:
            raster_filtered_img = raster_image.copy()
        else:
            pixel_keys = v_valid_mask * W + u_valid_mask
            
            # 1. Sort by highest first
            sort_order = np.lexsort((pixel_keys, -norm_depth_valid_mask)) 
            u_valid_mask, v_valid_mask = u_valid_mask[sort_order], v_valid_mask[sort_order]
            norm_depth_valid_mask = norm_depth_valid_mask[sort_order]
            
            # 2. Find the first occurence of each pixel
            _, unique_indices = np.unique(v_valid_mask * W + u_valid_mask, return_index=True)
            u_valid_mask, v_valid_mask = u_valid_mask[unique_indices], v_valid_mask[unique_indices]
            
            # 3. Finally Plot the colors
            raster_filtered_img[v_valid_mask, u_valid_mask] = cmap(norm_depth_valid_mask[unique_indices])[:, :3] * 255
    else:
        # Binary version (original behavior)
        raster_image = np.zeros((H, W), dtype=np.bool)
        raster_filtered_img = np.zeros((H, W), dtype=np.bool)
        raster_image[v[valid_within_bounds], u[valid_within_bounds]] = True
        if mask_2d is None:
            raster_filtered_img = raster_image.copy()
        else:
            raster_filtered_img[v[valid_within_bounds_n_mask], u[valid_within_bounds_n_mask]] = True
    
    
    return filtered_pointcloud, raster_image, raster_filtered_img
if __name__ == "__main__":
    import open3d as o3d
    import matplotlib.pyplot as plt
    pcd = o3d.io.read_point_cloud("multi_tree.ply")
    
    # Create a dummy mask (H, W) (e.g., circle mask)
    H, W = 480, 640
    y, x = np.ogrid[:H, :W]
    center = (W//2, H//2)
    radius = 200
    mask_2d = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    min_pts = np.array(pcd.points).min(axis=0) -1
    max_pts = np.array(pcd.points).max(axis=0) -2
    
    # Usage 1 : Using Segmentation Mask for rasterization
    filtered_pc_highest, raster_img, raster_img_filtered = rasterize_3dto2D(
        np.array(pcd.points), 
        mask_2d=mask_2d, 
        axis='z',
        min_xyz=min_pts,
        max_xyz=max_pts,
        highest_first=False,
        depth_weighting=True
    )
    
    # Usage 2 : Using Nothing for rasterization
    filtered_pc_highest, raster_img, raster_img_filtered = rasterize_3dto2D(
        np.array(pcd.points), 
        img_shape=(H,W),
        axis='z',
        min_xyz=min_pts,
        max_xyz=max_pts,
        highest_first=False,
        depth_weighting=True
    )
    
    # # Usage 3: Not Using min_xyz
    filtered_pc_highest, raster_img, raster_img_filtered = rasterize_3dto2D(
        torch.tensor(np.array(pcd.points)), 
        img_shape=(H,W),
        axis='z',
        highest_first=False,
        depth_weighting=True
    )
    
    plt.imshow(raster_img, cmap="gray")
    plt.show()
    plt.imshow(raster_img_filtered*255, cmap="gray")
    plt.show()
    # Output Preview
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(pcd.points)
    pcd_original.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
    
    pcd_highest = o3d.geometry.PointCloud()
    pcd_highest.points = o3d.utility.Vector3dVector(filtered_pc_highest)
    pcd_highest.paint_uniform_color([1, 0, 0])  # Red (farthest kept)
    o3d.visualization.draw_geometries([pcd, pcd_highest])