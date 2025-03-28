import numpy as np
import pandas as pd
import cv2
import time
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Not to self. The function is slow because of the sorting algorithm
In the future, Please use the Rapids aka Pandas on GPU
I've tried an implementation of numpy array but it's slower

"""
def limit_point_cloud(xyzrgbc:np.ndarray, w_lim, y_lim):
    x,y = xyzrgbc[0], xyzrgbc[1]
    # find the mask where x<w_lim
    mask = (-w_lim < x) & (x < w_lim) &\
           (-y_lim < y) & (y < y_lim)
    # get the points which satisfy the mask for xyzrgb
    xyzrgbc = xyzrgbc[:,mask]
    return xyzrgbc

def min_max_normalize(data, min_data, max_data):
    """
    Normalize an array using min-max normalization.
    Args:
        data: The array to normalize.
        min_data: The minimum value of the data.
        max_data: The maximum value of the data.
    Returns:
        Z: The normalized array.
    """
    return (data - min_data) / (max_data - min_data)

def cloud_to_gray_np(dim1_arr, dim2_arr, dim_depth, req_arr, stepsize, max_xyz, highest=True):
    """
    Convert the point cloud data to a 2D image.
    
    Args:
        dim1_arr: The x-coordinates of the point cloud data.
        dim2_arr: The y-coordinates of the point cloud data.
        dim_depth: The depth of the point cloud data.
        req_arr: The array to convert to the image.
        stepsize: The step size for the image.
        max_xyz: The maximum x, y, and z values.
        highest: Whether to keep the highest or lowest values.
    Returns:
        rtn: The 2D image
    """
    dim1_min, dim2_min, depth_max = -max_xyz[0], -max_xyz[1], abs(max_xyz[2])
    
    a = np.round((dim1_arr-dim1_min)/stepsize).astype(np.int32)
    b = np.round((dim2_arr-dim2_min)/stepsize).astype(np.int32)
    
    if len(req_arr.shape) == 1:
        req_arr = req_arr.reshape(-1,1)
        columnlst = [1]
    else:
        columnlst = [i for i in range(req_arr.shape[0])]
        req_arr = req_arr.T
    df = pd.DataFrame(req_arr, columns=columnlst)
    df["depth"] = dim_depth
    df["ab"] = list(zip(a,b)) # combine a,b to tuples
    df.sort_values(['ab', 'depth'], inplace=True, ascending=(not highest))
    df = df.drop_duplicates('ab')
    
    depth1d = min_max_normalize(df["depth"].to_numpy(), -depth_max, depth_max)*255
    depth1d = np.expand_dims(depth1d, 1)
    
    rtn = np.vstack(df[["ab"]].to_numpy().T[0]) # Convert tuples back to np arr
    rtn = np.hstack((rtn, df[columnlst].to_numpy(), depth1d)).T
    return rtn

def pcd2img_np_3d(xyz, stepsize:float, rgb_or_others, xyz_ext, highest=True)->np.ndarray:
    """
    :param pcd      : xyz
    :param stepsize : float [in meters]
    :return:        : numpy.ndarray [2D image]
    """
    if xyz.shape[1] != 3:
        xyz = xyz.T
    assert xyz.shape[1] == 3, "Invalid PointCloudData"
    len_pcd = xyz.shape[0]
    if rgb_or_others is None:
        rgb_or_others = np.ones((len_pcd))*255
        output_dim = 1
    assert len(rgb_or_others.shape) <= 2, f"Invalid RGBData, Ideal shape should be (3,n) or (4,n)"
    if len(rgb_or_others.shape) > 1:
        if rgb_or_others.shape[1] != len_pcd:
            rgb_or_others = rgb_or_others.T
        assert rgb_or_others.shape[1] == len_pcd, "Invalid RGBData"
        output_dim = int(rgb_or_others.shape[0])
    
    x,y,z = xyz[:,0], xyz[:,1], xyz[:,2]

    gv = cloud_to_gray_np(
        x,y,
        z,
        rgb_or_others, 
        stepsize,
        xyz_ext,
        highest=highest
    )
    n_ch = gv.shape[0]
    img_width = round((xyz_ext[0]-(-xyz_ext[0]))/stepsize)
    img_height = round((xyz_ext[1]-(-xyz_ext[1]))/stepsize)
    rtn_img = np.zeros((int(img_height)+1,int(img_width)+1, int(output_dim+1)), dtype=np.int32)
    rtn_img[np.int32(gv[1]), np.int32(gv[0])]= gv[2:n_ch].T
    
    rgb, seg, grey = rtn_img[:,:,0:3], rtn_img[:,:,3], rtn_img[:,:,4]
    return rgb, seg, grey



def cloud_to_gray_torch(
        dim1_arr, dim2_arr, dim_depth, req_arr, stepsize, max_xyz, rtn_img, highest=True):
    """
    Convert the point cloud data to a 2D image.
    
    Args:
        dim1_arr: The x-coordinates of the point cloud data.
        dim2_arr: The y-coordinates of the point cloud data.
        dim_depth: The depth of the point cloud data.
        req_arr: The array to convert to the image.
        stepsize: The step size for the image.
        max_xyz: The maximum x, y, and z values.
        highest: Whether to keep the highest or lowest values.
    Returns:
        rtn: The 2D image
    """
    if not dim1_arr.is_cuda:
        dim1_arr, dim2_arr, dim_depth, req_arr, rtn_img= map(lambda x: torch.asarray(x).to(DEVICE), (dim1_arr, dim2_arr, dim_depth, req_arr, rtn_img))
    
    dim1_min, dim2_min, depth_max = -abs(max_xyz[0]), -abs(max_xyz[1]), abs(max_xyz[2])
    
    dim1_arr = torch.round((dim1_arr-dim1_min)/stepsize).type(torch.int32)
    dim2_arr = torch.round((dim2_arr-dim2_min)/stepsize).type(torch.int32)
    
    if len(req_arr.shape) == 1:
        req_arr = req_arr.view(-1,1)
        req_arr = req_arr.t()

    ind = torch.argsort(dim_depth.t())
    dim_depth = min_max_normalize(dim_depth, -depth_max, depth_max)#*255
    rtn = torch.vstack((dim1_arr, dim2_arr, req_arr, dim_depth))
    rtn = rtn[:,ind].float()
    # Reverse the rtn array
    if highest:
        rtn = torch.flip(rtn,(1,))
    rtn_img[rtn[0].long(), rtn[1].long()] = rtn[2:].T
    # Create a mask array which values are there
    w,h,ch = rtn_img.shape
    mask = torch.zeros((w,h))
    mask[rtn[0].long(), rtn[1].long()] = 1
    
    if req_arr.shape[0] == 1:
        seg, grey = rtn_img[:,:,0], rtn_img[:,:,1]
        return mask.bool().cpu().detach().numpy() ,seg.int().cpu().detach().numpy(), grey.float().cpu().detach().numpy()
    else:
        rgb, seg, grey = rtn_img[:,:,0:3], rtn_img[:,:,3], rtn_img[:,:,4]
        return rgb.int().cpu().detach().numpy(), seg.int().cpu().detach().numpy(), grey.float().cpu().detach().numpy()

def pcd2img_torch_v2(xyz, w_h_img, max_x_y_z, rgbc=None, highest=True):
    """
    This requires that the pointcloud is BOUNDED 
    MEANING x,y more than a MAX_X_Y Has been removed
    Helper func located at occlusion.py World2Drone2VehicleV2
    Args:
        xyz: np.array of shape (N,3)
        w_h_img: tuple of (w,h)
        max_x_y_z: tuple of (max_x, max_y, max_z)
        rgb_or_others: np.array of shape (3,N) or (4,N)
        highest: bool
    Returns:
        rgb: np.array of shape (w,h,3)
        seg: np.array of shape (w,h)
        grey: np.array of shape (w,h)
    """
    len_pcd = xyz.shape[1]
    if rgbc is None:
        rgbc = torch.ones((len_pcd))*255
        output_dim = 1
    assert len(rgbc.shape) <= 2, f"Invalid RGBData, Ideal shape should be (3,n) or (4,n)"
    if len(rgbc.shape) > 1:
        if rgbc.shape[1] != len_pcd:
            rgbc = rgbc.T
        assert rgbc.shape[1] == len_pcd, "Invalid RGBData"
        output_dim = int(rgbc.shape[0])
    else:
        output_dim = 1
    
    w,h = w_h_img
    max_x, max_y, max_z = max_x_y_z
    stepsize_x, stepsize_y = 2*max_x/(h-1), 2*max_y/(w-1)
    stepsize = stepsize_x
    x,y,z = -xyz[0], -xyz[1], xyz[2]
    x_capped = torch.clip(x, -max_x, max_x)
    y_capped = torch.clip(y, -max_y, max_y)
    z_capped = torch.clip(z, -max_z, max_z).float()

    return cloud_to_gray_torch(
        x_capped,y_capped,
        z_capped,
        rgbc, 
        stepsize,
        max_x_y_z,
        rtn_img = torch.zeros((int(h),int(w), int(output_dim+1))).to(DEVICE),
        highest=highest
    )

def blur_n_inpaint_fill(rgb_np, gry_msk, blur=10, radius=0.2):
    rgb_np = rgb_np.astype(np.uint8)
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    w,h, ch= rgb_np.shape
    w_blur, h_blur = int(w/blur), int(h/blur)
    gray = cv2.blur(gray, (blur,blur)) # blur the image
    mask = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8) # Initialize mask

    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # creating convex hull object for each contour
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
        cv2.drawContours(mask, hull, i, 255, -1)
    for i in range(len(contours)):
        cv2.drawContours(mask, hull, i, 255, -1)

    mask_new = np.zeros_like(gray)
    mask_new[(mask == 255) & (gry_msk!=255)] = True
    mask_n = np.uint8(mask_new)

    
    dst_NS = cv2.inpaint(rgb_np,mask_n,radius,cv2.INPAINT_NS)
    return dst_NS, mask