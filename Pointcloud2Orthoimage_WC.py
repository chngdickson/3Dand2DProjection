#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang, Ph.D.(http://www.yuhanjiang.com)
# Date:         4/2/2022
# Discriptions : pointcloud to orthoimage Waste Management Version
# Major updata :
import copy
import gc
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
from mpl_toolkits import mplot3d
import cv2 as cv
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from itertools import repeat
from multiprocessing import Process
from multiprocessing import pool


class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(pool.Pool):
    Process = NoDaemonProcess
def rotate(img,angle):#旋转功能 逆时针
    h, w =img.shape[0:2]# 获取图像尺寸
    center = (w/2, h/2)# 将图像中心设为旋转中心  @04.21.2019 either 4/2=2 or 5/2=2.5
    M = cv.getRotationMatrix2D(center, angle, 1) #执行旋转
    rotated = cv.warpAffine(img,M,(w,h))
    return rotated# 返回旋转后的图像
def newdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        #print(path+'   Successful')
        return True
    else:
        #print(path+'   Exists')
        return False
def generateGridImageUisngMultiCPU(X,Y,Z):
    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))

    grid_x,grid_y=np.mgrid[X.min():X.max():(x_range*1j),Y.min():Y.max():(y_range*1j)]  # create the grid size

    grid_z = griddata((X,Y), Z, (grid_x, grid_y), method='linear')#{‘linear’, ‘nearest’, ‘cubic’}, optional
    try:
        return grid_z
    finally:
        del X,Y,Z,grid_z,grid_x,grid_y
        gc.collect()

def PointCloud2OrthoimageNonrgb(PCD,downsample=10,GSDmm2px=5):
    pts = np.array(PCD.points).T
    
    if downsample>0:
        X=pts[0][::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=pts[2][::downsample]*1000/GSDmm2px  # [::10] downsample 1/10
        Z=pts[1][::downsample]*1000/GSDmm2px
        #R=(PCD.red[::downsample]/(2**16-1)*255).astype('uint8')  # covert 16-bit color to 8-bit
        #G=(PCD.green[::downsample]/(2**16-1)*255).astype('uint8')
        #B=(PCD.blue[::downsample]/(2**16-1)*255).astype('uint8')
        # R=(PCD.red[::downsample])#.astype('uint8')  # keep 16-bit
        # G=(PCD.green[::downsample])#.astype('uint8')
        # B=(PCD.blue[::downsample])#.astype('uint8')
        # print('[DownSamplePCShape]',X.shape,Y.shape,Z.shape)
    else:
        X=pts[0]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=pts[2]*1000/GSDmm2px  # [::10] downsample 1/10
        Z=pts[1]*1000/GSDmm2px
        #R=(PCD.red[::downsample]/(2**16-1)*255).astype('uint8')  # covert 16-bit color to 8-bit
        #G=(PCD.green[::downsample]/(2**16-1)*255).astype('uint8')
        #B=(PCD.blue[::downsample]/(2**16-1)*255).astype('uint8')
        # R=PCD.red  #.astype('uint8')  # keep 16-bit
        # G=PCD.green  #.astype('uint8')
        # B=PCD.blue#.astype('uint8')

    # print("[RGBColorRange]",R,G,B)

    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))
    # print("[ImageFrameSize]",x_range,y_range)

    ele_max=Z.max()
    ele_min=Z.min()
    z_range=[ele_min,ele_max]

    print('[x,y,z range in mm]',x_range*GSDmm2px,y_range*GSDmm2px,z_range)

    #print(X.shape,Y.shape,Z.shape)

    #grid_ele = griddata((X,Y), Z, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_R = griddata((X,Y), R, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_G = griddata((X,Y), G, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_B = griddata((X,Y), B, (grid_x, grid_y), method='cubic').astype(np.float)
    EleRGB=[Z,R,G,B]
    pool=MyPool(4)  #//5+1)
    grid_Mutiple=pool.starmap(generateGridImageUisngMultiCPU,zip(repeat(X),repeat(Y),EleRGB))
    pool.close()

    grid_ele=grid_Mutiple[0].astype('float')
    grid_R=grid_Mutiple[1].astype('uint16')
    grid_G=grid_Mutiple[2].astype('uint16')
    grid_B=grid_Mutiple[3].astype('uint16')

    grid_RGB=np.zeros((grid_ele.shape[0],grid_ele.shape[1],3)).astype('uint16')
    grid_RGB[:,:,0]=grid_R
    grid_RGB[:,:,1]=grid_G
    grid_RGB[:,:,2]=grid_B

    print('[RGB,Ele imageShape]',grid_RGB.shape,grid_ele.shape)
    return grid_RGB,grid_ele,[ele_min,ele_max]

def PointCloud2Orthoimage(PCD,downsample=10,GSDmm2px=5):
    print('[PointCloudShape X,Y,Z]',PCD.x.shape,PCD.y.shape,PCD.z.shape)

    if downsample>0:
        X=PCD.x[::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=PCD.z[::downsample]*1000/GSDmm2px  # [::10] downsample 1/10
        Z=PCD.y[::downsample]*1000/GSDmm2px
        #R=(PCD.red[::downsample]/(2**16-1)*255).astype('uint8')  # covert 16-bit color to 8-bit
        #G=(PCD.green[::downsample]/(2**16-1)*255).astype('uint8')
        #B=(PCD.blue[::downsample]/(2**16-1)*255).astype('uint8')
        R=(PCD.red[::downsample])#.astype('uint8')  # keep 16-bit
        G=(PCD.green[::downsample])#.astype('uint8')
        B=(PCD.blue[::downsample])#.astype('uint8')
        print('[DownSamplePCShape]',X.shape,Y.shape,Z.shape)
    else:
        X=PCD.x*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=PCD.z*1000/GSDmm2px  # [::10] downsample 1/10
        Z=PCD.y*1000/GSDmm2px
        #R=(PCD.red[::downsample]/(2**16-1)*255).astype('uint8')  # covert 16-bit color to 8-bit
        #G=(PCD.green[::downsample]/(2**16-1)*255).astype('uint8')
        #B=(PCD.blue[::downsample]/(2**16-1)*255).astype('uint8')
        R=PCD.red  #.astype('uint8')  # keep 16-bit
        G=PCD.green  #.astype('uint8')
        B=PCD.blue#.astype('uint8')

    print("[RGBColorRange]",R,G,B)

    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))
    print("[ImageFrameSize]",x_range,y_range)

    ele_max=Z.max()
    ele_min=Z.min()
    z_range=[ele_min,ele_max]

    print('[x,y,z range in mm]',x_range*GSDmm2px,y_range*GSDmm2px,z_range)

    #print(X.shape,Y.shape,Z.shape)

    #grid_ele = griddata((X,Y), Z, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_R = griddata((X,Y), R, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_G = griddata((X,Y), G, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_B = griddata((X,Y), B, (grid_x, grid_y), method='cubic').astype(np.float)
    EleRGB=[Z,R,G,B]
    pool=MyPool(4)  #//5+1)
    grid_Mutiple=pool.starmap(generateGridImageUisngMultiCPU,zip(repeat(X),repeat(Y),EleRGB))
    pool.close()

    grid_ele=grid_Mutiple[0].astype('float')
    grid_R=grid_Mutiple[1].astype('uint16')
    grid_G=grid_Mutiple[2].astype('uint16')
    grid_B=grid_Mutiple[3].astype('uint16')

    grid_RGB=np.zeros((grid_ele.shape[0],grid_ele.shape[1],3)).astype('uint16')
    grid_RGB[:,:,0]=grid_R
    grid_RGB[:,:,1]=grid_G
    grid_RGB[:,:,2]=grid_B

    print('[RGB,Ele imageShape]',grid_RGB.shape,grid_ele.shape)
    return grid_RGB,grid_ele,[ele_min,ele_max]
def PointCloud2Orthoimage2(points,colors,downsample=10,GSDmm2px=5):
    print('[PointCloudShape XYZ RGB]',points.shape,colors.shape)
    if downsample>0:
        X=points[:,0][::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=points[:,1][::downsample]*-1000/GSDmm2px  # [::10] downsample 1/10
        if X.max()>Y.max():
            print('[Rotated]')
            X=points[:,1][::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
            Y=points[:,0][::downsample]*-1000/GSDmm2px  # [::10] downsample 1/10
        Z=points[:,2][::downsample]*1000# elevation in mm
        R=(colors[:,0][::downsample])# keep 16-bit
        G=(colors[:,1][::downsample])
        B=(colors[:,2][::downsample])
        print('[DownSamplePCShape]',X.shape,Y.shape,Z.shape)
    else:
        X=points[:,0]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=points[:,1]*-1000/GSDmm2px  # [::10] downsample 1/10
        if X.max()>Y.max():
            print('[Rotated]')
            X=points[:,1]*1000/GSDmm2px  # 1000 means:1mm to 1 px
            Y=points[:,0]*-1000/GSDmm2px  # [::10] downsample 1/10
        Z=points[:,2]*1000#elevation in mm
        R=colors[:,0]# keep 16-bit
        G=colors[:,1]
        B=colors[:,2]
    print("[RGBColorRange]",R,G,B)
    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))
    print("[ImageFrameSize]",x_range,y_range)
    ele_max=Z.max()
    ele_min=Z.min()
    z_range=[ele_min,ele_max]
    print('[x,y,z range in mm]',x_range*GSDmm2px,y_range*GSDmm2px,z_range)

    EleRGB=[Z,R,G,B]
    pool=MyPool(4)
    grid_Mutiple=pool.starmap(generateGridImageUisngMultiCPU,zip(repeat(X),repeat(Y),EleRGB))
    pool.close()

    grid_ele=grid_Mutiple[0].astype('float')
    grid_R=grid_Mutiple[1].astype('uint16')
    grid_G=grid_Mutiple[2].astype('uint16')
    grid_B=grid_Mutiple[3].astype('uint16')

    grid_RGB=np.zeros((grid_ele.shape[0],grid_ele.shape[1],3)).astype('uint16')
    grid_RGB[:,:,0]=grid_R
    grid_RGB[:,:,1]=grid_G
    grid_RGB[:,:,2]=grid_B

    print('[RGB,Ele imageShape]',grid_RGB.shape,grid_ele.shape)
    print('[GSD: mm/px]',GSDmm2px)
    try:
        return grid_RGB,grid_ele,[ele_min,ele_max]
    finally:
        del grid_B,grid_G,grid_R,grid_RGB,grid_ele,grid_Mutiple,EleRGB,X,Y,Z,R,G,B,pool,points,colors


def cameraSelector(v):
    camera=[]
    camera.append(v.get('eye'))
    camera.append(v.get('phi'))
    camera.append(v.get('theta'))
    camera.append(v.get('r'))
    return np.concatenate(camera).tolist()

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def get_floor_plane(pcd, dist_threshold=0.02, num_iterations=2000,bool_visualize=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=3,
                                             num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    inlier_cloud=pcd.select_by_index(inliers)
    if bool_visualize:
        inlier_cloud.paint_uniform_color([1.0,0,0])
        outlier_cloud=pcd.select_by_index(inliers,invert=True)
        o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud],window_name='FloorPlane_wc@Red ',width=1920//3*2,height=1080//3*2)
    il_points=np.array(inlier_cloud.points)
    plane_ele_mean=il_points[:,2].mean()
    print('[A wcPlaneRange@center]',il_points[:,2].min(),il_points[:,2].max(),plane_ele_mean)
    try:
        return plane_model,plane_ele_mean
    finally:
        del pcd,inliers,inlier_cloud,il_points
def align_wc_surface(pcd,bool_visualize=False,bool_repeate=True,dist_threshold=0.02, num_iterations=2000):
    downpcd=copy.deepcopy(pcd).voxel_down_sample(voxel_size=0.05)
    floor,plane_ele_mean=get_floor_plane(downpcd,bool_visualize=bool_visualize,dist_threshold=dist_threshold, num_iterations=num_iterations)
    a,b,c,d=floor
    # Translate plane to z-coordinate = 0
    pcd.translate((0,0,-plane_ele_mean))
    # Calculate rotation angle between plane normal & z-axis
    plane_normal=tuple(floor[:3])
    z_axis=(0,0,1)
    rotation_angle=vector_angle(plane_normal,z_axis)
    # Calculate rotation axis
    plane_normal_length=math.sqrt(a**2+b**2+c**2)
    u1=b/plane_normal_length
    u2=-a/plane_normal_length
    rotation_axis=(u1,u2,0)
    # Generate axis-angle representation
    optimization_factor=1  #1.4
    axis_angle=tuple([x*rotation_angle*optimization_factor for x in rotation_axis])
    # Rotate point cloud
    R=pcd.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(R,center=(0,0,0))
    if bool_repeate:
        bool_repeate=False
        return align_wc_surface(pcd,bool_visualize=False,bool_repeate=bool_repeate,dist_threshold=0.02*2, num_iterations=2000)
    else:
        try:
            return pcd
        finally:
            del pcd,downpcd

def main(glb_file_path,pointName='5mm_18_34_56',downsample=10,GSDmm2px=5,bool_alignOnly=False,b='win',bool_generate=False):
    print('$',pointName)
    bool_confirm=False
    if b=='win':
        #import pptk
        import open3d as o3d
        axis_mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()  #o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=5.0,origin=np.array([0.,0.,0.]))
        pcd=o3d.io.read_point_cloud(glb_file_path+pointName+'.pts', format='xyzrgb') # xyz(double)rgb(256)normal(double)
        points=(np.asarray(pcd.points))
        colors=(np.asarray(pcd.colors))
        #region get_the_min_rotated_boundingbox
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(points)
        pcd.colors=o3d.utility.Vector3dVector(colors/256)
        #pcd.normals=o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd,axis_mesh],window_name='OrginalPCD_WC'+pointName,width=1920//3*2,height=1080//3*2)
        #region get floor plane# Get the plane equation of the floor → ax+by+cz+d = 0
        #pcd=align_wc_surface(pcd)
        #get num of point > 0, and num of point <0
        zzz=np.array(pcd.points)[:,2]#[::2000]
        print(zzz.max(),zzz.min())
        plane_ele_mean=0.035#.3
        print(np.sum(zzz>plane_ele_mean),np.sum(zzz<-plane_ele_mean))
        if abs(zzz.max())>abs(zzz.min()) and np.sum(zzz>plane_ele_mean)>np.sum(zzz<-plane_ele_mean):
            upmodel=1
        else:
            upmodel=-1
        #o3d.visualization.draw_geometries([pcd,axis_mesh],window_name='FloorPlane_wc '+pointName,width=1920//3*2,height=1080//3*2)
        #endregion
        #
        #obb=pcd.get_oriented_bounding_box()
        #obb.color=(0,1,0)  #obbBounding box is green
        #center=obb.center
        #extent=obb.extent
        #print('#[BoundingBox]',obb)
        #
        obb=pcd.get_oriented_bounding_box()
        obb.color=(0,1,0)  #obbBounding box is green
        center=obb.center
        extent=obb.extent
        R=np.matrix(obb.R)
        #print(center,extent,R,R.I)
        R=R.I
        #R[:,2]=1
        pcd_r=copy.deepcopy(pcd).rotate(R,center=center)#rotation
        #pcd_r=align_wc_surface(pcd_r)
        #get num of point > 0, and num of point <0
        points_r=np.array(pcd_r.points)
        zzz_r=points_r[:,2]#[::2000]
        print(zzz_r.max(),zzz_r.min())
        print(np.sum(zzz_r>plane_ele_mean),np.sum(zzz_r<-plane_ele_mean))
        if abs(zzz_r.max())>abs(zzz_r.min()) and np.sum(zzz_r>plane_ele_mean)>np.sum(zzz_r<-plane_ele_mean):
            upmodel_r=1
        else:
            upmodel_r=-1
        if True:#upmodel+upmodel_r==0:
            back_pcd_r=copy.deepcopy(pcd_r)
            print('[Flipped Z-axis]')
            points_r[:,2]=points_r[:,2]*-1
            points_r[:,1]=points_r[:,1]*-1
            pcd_r.points=o3d.utility.Vector3dVector(points_r)
            o3d.visualization.draw_geometries([pcd_r,axis_mesh],window_name='Flipped_wc '+pointName,width=1920//3*2,height=1080//3*2)  # if only show red box, then the green box is been covered. then the results is correct.
            bool_confirm=input('Confirm the filp? y/n:')
            if  not bool_confirm in ['y',"Y"]:
                print('Discard filp')
                pcd_r=copy.deepcopy(back_pcd_r)
                bool_confirm=True
        obb_r=pcd_r.get_oriented_bounding_box()
        #obb_r.color=(0,1,0)  #obbBounding box is green
        center_xy=np.array(obb_r.center)
        center_xy[2]=0
        pcd_t=copy.deepcopy(pcd_r).translate(-center_xy)
        aabb=pcd_t.get_axis_aligned_bounding_box()
        aabb.color=(1,0,0)  #aabb bounding box is red
        obb_t=pcd_t.get_oriented_bounding_box()
        obb_t.color=(0,1,0)  #obbBounding box is green
        print(aabb)
        print(obb_t)
        if bool_confirm==False:
            o3d.visualization.draw_geometries([pcd_t,aabb,obb_t,axis_mesh],window_name='Tanslated_wc '+pointName,width=1920//3*2,height=1080//3*2)  # if only show red box, then the green box is been covered. then the results is correct.

        if bool_alignOnly and bool_generate:
            print('[Start]---/...')
            #o3d.io.write_point_cloud(glb_file_path+pointName+"aligned.pcd",pcd_t)
            df=np.hstack([np.array(pcd_t.points),np.asarray(pcd_t.colors)])
            df=pd.DataFrame(df)
            df.to_csv(glb_file_path+pointName+"aligned.csv",index=False,header=False)
            print('[Saved]',glb_file_path+pointName+"aligned.csv")

    if b=='server':

        pc=pd.read_csv(glb_file_path+pointName+"aligned.csv",index_col=False,header=None)
        pc=np.array(pc)
        print(['CSV pointcloud formate'],pc.shape)
        points=pc[:,0:3]#*-1
        colors=pc[:,3:6]
        #endregion
    if bool_alignOnly:
        print('[Piontcloud aligment only]')
        return False
    #region ouput
    if b=='win':
        grid_RGB,grid_ele,(ele_min,ele_max)=PointCloud2Orthoimage2(np.array(pcd_t.points),np.asarray(pcd_t.colors)*65535,downsample=downsample,GSDmm2px=GSDmm2px)  #PointCloud2Orthoimage(PCD,downsample=0,GSDmm2px=5)
    if b=='server':
        grid_RGB,grid_ele,(ele_min,ele_max)=PointCloud2Orthoimage2(np.array(points),np.asarray(colors)*65535,downsample=downsample,GSDmm2px=GSDmm2px)  #PointCloud2Orthoimage(PCD,downsample=0,GSDmm2px=5)
    grid_RGB=(grid_RGB/(2**16-1)*255).astype('uint8')
    grid_map=((grid_ele-ele_min)/(ele_max-ele_min)*255).astype('uint8')
    #if grid_ele.shape[0]>grid_ele.shape[1]:# alway keep width larger than height
    #    grid_ele=rotate(grid_ele,90)
    #    grid_RGB=rotate(grid_RGB,90)
    try:
        return grid_RGB,grid_ele,grid_map,(ele_min,ele_max),GSDmm2px
    finally:
        newdir(glb_file_path+'/Demo/'+pointName+'/')
        cv.imwrite(glb_file_path+'/Demo/'+pointName+'/'+pointName+'RGB.jpg',cv.cvtColor(grid_RGB,cv.COLOR_RGB2BGR),[int(cv.IMWRITE_JPEG_QUALITY),100])  # agg 1 binder 128 air 255
        cv.imwrite(glb_file_path+'/Demo/'+pointName+'/'+pointName+'DEM.jpg',grid_map,[int(cv.IMWRITE_JPEG_QUALITY),100])  # agg 1 binder 128 air 255
        print('[Done]',glb_file_path+'/Demo/'+pointName+'/'+pointName+'RGB/DEM.jpg')
        try:
            del PCD,point_cloud,points,colors,pcd_t,pcd_r
        except:
            del pc,points,colors
        gc.collect()

#-------
if __name__ == '__main__':
    if os.path.exists('D:/'):
        glb_file_path='D:/CentOS/WC/'  # screenshot saving path
        b='win'
        cpu=3
        import open3d as o3d
    elif os.path.exists('/data/'):
        glb_file_path='/data/wc/'  # screenshot saving path
        b='server'
        cpu=4
    if b=='win':
        PC_Name=["wc"]
        #PC_Name.reverse()
        for i in PC_Name:
            main(pointName=i,glb_file_path=glb_file_path,GSDmm2px=5,bool_alignOnly=1,b=b,bool_generate=1)# default 5
    else:
        main(pointName='1_21_48_14',glb_file_path=glb_file_path,GSDmm2px=5,bool_alignOnly=False,b=b)
    #if b=='server':
    #    exec(open('/scratch/py3d/wc/SD_Joints_Extraction_Tool.py').read())
    #else:
    #    exec(open('F:\OneDrive\PyCharmProject\Py3D\wc\SD_Joints_Extraction_Tool.py',encoding='utf-8').read())
