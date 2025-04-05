import cv2
import torch
import numpy as np

from proj_pcd2img import project_3d_to_2d
from ransac_circle import ransac_circle
def create_2d_mask(width, height, annotations):
    imgRGBWithAnnotations = np.zeros([height, width])
    imgRGBWithAnnotations = cv2.fillPoly(imgRGBWithAnnotations, [annotations], 1)
    return imgRGBWithAnnotations

def segment3D_With2DMask(pcd, mask_2d, K, extrinsic):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Project 3d_to_2d
    pcd_uv = project_3d_to_2d(torch.tensor(np.array(pcd.points)).to(device=device), K, extrinsic)
    w, h = get_image_size_from_intrinsic(K)
    pcd_within_bounds, _ = removal_boundary(pcd, pcd_uv, w=w, h=h)
    pcd_uv = project_3d_to_2d(torch.tensor(np.array(pcd_within_bounds.points)).to(device=device), K, extrinsic)
    
    if isinstance(mask_2d, np.ndarray):
        mask_2d = torch.tensor(mask_2d, device=device).bool()
    
    mask_2d_v2 = torch.zeros(mask_2d.shape).to(device).bool()
    mask_2d_v2[pcd_uv[:,1], pcd_uv[:,0]] = True
    
    return pcd_within_bounds, mask_2d_v2#, test_ransac

def segment3D_with2DAnno(pcd, segment2d_anno, im_w, im_h, K, extrinsic):
    mask_2d = create_2d_mask(width=im_w, height=im_h, annotations=segment2d_anno)
    pcd_selected = segment3D_With2DMask(pcd, mask_2d, K, extrinsic)
    return pcd_selected

def removal_boundary(pcd, pcd_uv, w,h):
    x, y = pcd_uv[:, 0], pcd_uv[:, 1]
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return (pcd.select_by_index(mask.nonzero().cpu().numpy()), pcd_uv[mask])

def get_image_size_from_intrinsic(K):
    """
    Compute image width and height from intrinsic matrix K.
    
    Args:
        K: (3, 3) intrinsic matrix.
    
    Returns:
        w: Image width (pixels).
        h: Image height (pixels).
    """
    c_x = K[0, 2]
    c_y = K[1, 2]
    w = int(2 * c_x)
    h = int(2 * c_y)
    return w, h
if __name__ == "__main__":
    pass