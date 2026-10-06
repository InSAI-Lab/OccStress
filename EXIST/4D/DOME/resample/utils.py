# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(2560, 1440))
# display.start()
# from mayavi import mlab
# import mayavi
# mlab.options.offscreen = True
# print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
import numpy as np
import os
from pyquaternion import Quaternion
from functools import reduce 
import open3d as o3d
from tqdm import tqdm
import scipy.interpolate as scipy_interpolate
from sklearn.neighbors import NearestNeighbors
import cv2


colors = np.array(
    [
        [255, 120,  50, 255],       # barrier              orange
        [255, 192, 203, 255],       # bicycle              pink
        [255, 255,   0, 255],       # bus                  yellow
        [  0, 150, 245, 255],       # car                  blue
        [  0, 255, 255, 255],       # construction_vehicle cyan
        [255, 127,   0, 255],       # motorcycle           dark orange
        [255,   0,   0, 255],       # pedestrian           red
        [255, 240, 150, 255],       # traffic_cone         light yellow
        [135,  60,   0, 255],       # trailer              brown
        [160,  32, 240, 255],       # truck                purple                
        [255,   0, 255, 255],       # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [ 75,   0,  75, 255],       # sidewalk             dard purple
        [150, 240,  80, 255],       # terrain              light green          
        [230, 230, 250, 255],       # manmade              white
        [  0, 175,   0, 255],       # vegetation           green
        # [  0, 255, 127, 255],       # ego car              dark cyan
        # [255,  99,  71, 255],       # ego car
        # [  0, 191, 255, 255]        # ego car
    ]
).astype(np.uint8)

def pass_print(*args, **kwargs):
    pass


def get_grid_coords(dims, resolution,use_center=True,indexing='xy'):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz,indexing=indexing)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    # coords_grid = (coords_grid * resolution) + resolution / 2
    offset=0
    if use_center:
        offset=0.5
    coords_grid = (coords_grid+offset) * resolution

    return coords_grid

def draw(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        # fov_voxels[:, 1],
        # fov_voxels[:, 0],
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=1.0 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=16, # 16
    )

    global  colors
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    dst=os.path.join(save_dir, f'vis_{timestamp}.png')
    mlab.savefig(dst)
    mlab.close()
    return dst

def write_pc(pc,dst,c=None):
    # pc=pc
    import open3d as o3d 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if c is not None:
        pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud(dst, pcd)

def visualize_point_cloud(
    all_pred,
    abs_pose=None,
    abs_rot=None,
    abs_trans=None,
    vox_origin=[-40, -40, -1],
    resolution=0.4,  #voxel size
    cmp_dir="./",
    key='gt',
    frame_type='e+w',
    save_box=False,
    save_label=False,
):
    if abs_trans is not None:
        abs_pose=abs_trans[:,:3,3]
        abs_rot=abs_trans[:,:3,:3]
    assert len(all_pred)==len(abs_pose)==len(abs_rot)
    os.makedirs(cmp_dir,exist_ok=True)
    all_occ,all_color=[],[]
    cls_label=[]
    bboxs=[]
    pose_mesh=[]
    for i,(occ,pose,rot) in enumerate(zip(all_pred,abs_pose,abs_rot)):
        occ=occ.reshape(-1)#.flatten()
        # mask=(occ>=1)&(occ<16)  # ignore GO   0:GO 17:empty
        mask=occ<17  # ignore GO   0:GO 17:empty
        cc=colors[occ[mask]-1][:,:3]/255.0 #[...,::-1]
        cls_label.append(occ[mask])
        # occ_x,occ_y,occ_z=np.meshgrid(np.arange(200),np.arange(200),np.arange(16))
        # occ_x=occ_x.flatten()
        # occ_y=occ_y.flatten()
        # occ_z=occ_z.flatten()
        # occ_xyz=np.concatenate([occ_x[:,None],occ_y[:,None],occ_z[:,None]],axis=1)
        # occ_xyz=(occ_xyz * resolution) + resolution / 2 # to center
        # occ_xyz+=np.array([-40,-40,-1]) # to ego
        # Compute the voxels coordinates in ego frame
        occ_xyz = get_grid_coords(
            [200,200,16], [resolution]*3
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3]) #[80,80,6.4] [-40, -40 , -1, 40,40 5.4]
        if 'e' in frame_type:
            write_pc(occ_xyz[mask],os.path.join(cmp_dir, f'vis_{key}_{i}_e.ply'),c=cc)                      

        # ego to world
        if rot.shape==(3,3):
            rot_m=rot
        else:
            rot_m=Quaternion(rot).rotation_matrix[:3,:3]
        # rot_m=rr@rot_m

        trans_mat=np.eye(4)
        trans_mat[:3,:3]=rot_m
        trans_mat[:3,3]=pose
        rr=np.array([
            [0,1,0],
            [1,0,0],
            [0,0,1]
        ])
        occ_xyz=occ_xyz@rr.T
        occ_xyz=occ_xyz@rot_m.T +pose

        if 'w' in frame_type:
            write_pc(occ_xyz[mask],os.path.join(cmp_dir, f'vis_{key}_{i}_w.ply'),c=cc)                      

        all_occ.append(occ_xyz[mask])
        all_color.append(cc)
        
        if save_box:
            bboxs.append(draw_bbox_3d(trans_mat,80,80,3))
        pose_mesh.append(get_pose_mesh(trans_mat))
        # aaa=create_bbox(occ_xyz)
        # __import__('ipdb').set_trace()
        # print(type(aaa))
        # bboxs.append(aaa)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(occ_xyz)
        # obb=pcd.get_oriented_bounding_box()
        # import ipdb;ipdb.set_trace()
        # bboxs.append(obb.R)


    # all_occ=all_occ[-1:]
    # all_color=all_color[-1:]
    all_occ=np.concatenate(all_occ, axis=0)
    all_color=np.concatenate(all_color, axis=0)
    cls_label=np.concatenate(cls_label, axis=0)

    if save_label:
        np.save(os.path.join(cmp_dir, f'cls_label_{key}_all_w.npy'),cls_label)
    # save mesh
    if save_box:
        o3d.io.write_line_set(os.path.join(cmp_dir, f'vis_{key}_all_w_box.ply'), merge_mesh(bboxs))
    # o3d.io.write_line_set(os.path.join(cmp_dir, f'vis_{key}_all_w_box.ply'), merge_mesh(bboxs[-1:]))
    # o3d.io.write_triangle_mesh(os.path.join(cmp_dir, f'vis_{key}_all_w_box.ply'),merge_mesh(bboxs))
    write_pc(all_occ,os.path.join(cmp_dir, f'vis_{key}_all_w.ply'),c=all_color)
    o3d.io.write_triangle_mesh(os.path.join(cmp_dir, f'vis_{key}_all_w_traj.ply'),merge_mesh(pose_mesh))
    

def visualize_point_cloud_no_pose(
    all_pred,
    vox_origin=[-40, -40, -1],
    resolution=0.4,  #voxel size
    cmp_dir="./",
    key='000000',
    key2='gt',
    offset=0,
):
    os.makedirs(cmp_dir,exist_ok=True)
    for i,occ in enumerate(all_pred):
        occ_d=occ.copy()
        occ=occ.reshape(-1)#.flatten()
        mask=(occ>=1)&(occ<16)  # ignore GO  
        cc=colors[occ[mask]-1][:,:3]/255.0 #[...,::-1]

        occ_xyz = get_grid_coords(
            [200,200,16], [resolution]*3
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
        write_pc(occ_xyz[mask],os.path.join(cmp_dir, f'vis_{key}_{i+offset:02d}_e_{key2}.ply'),c=cc)                      

        np.save(os.path.join(cmp_dir, f'vis_{key}_{i+offset:02d}_e_{key2}.npy'),occ_d) 



def draw_bbox_3d(
  trans_mat,
    w,
    h,
    d,
):
    import open3d as o3d
    points =np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]).astype(np.float64)
    points=points-0.5
    # __import__('ipdb').set_trace()
    points*=np.array([w,h,d])
    
  
    points=points@trans_mat[:3,:3].T +trans_mat[:3,3]

    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_bbox(occ_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occ_xyz)
    obb=pcd.get_oriented_bounding_box()
    # Convert OrientedBoundingBox to a mesh
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=obb.extent[0], height=obb.extent[1], depth=obb.extent[2])
    # __import__('ipdb').set_trace()
    box_mesh.rotate(obb.R)# center=False)
    box_mesh.translate(obb.center)
    return box_mesh 

def merge_mesh(meshes):
    return reduce(lambda x,y:x+y, meshes)



def get_pose_mesh(trans_mat,s=5):

    # Create a coordinate frame with x-axis (red), y-axis (green), and z-axis (blue)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=s, origin=[0, 0, 0])
    mesh_frame.transform(trans_mat)
    # Save the coordinate frame to a file
    return mesh_frame

def vis_pose_mesh(poses,s=5,cmp_dir="./",fn='vis_pose.ply'):
    os.makedirs(cmp_dir,exist_ok=True)
    pose_mesh=[]
    for pose in poses:
        pose_mesh.append(get_pose_mesh(pose))
    o3d.io.write_triangle_mesh(os.path.join(cmp_dir, fn),merge_mesh(pose_mesh))
    # 'vis_{key}_all_w_traj.ply'



def get_inliers_outliers(pose,w,h,d, pcd):
   # Create an oriented bounding box with the specified pose
    center,rotation=pose[:3,3],pose[:3,:3]
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=[w,h,d])
    inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
    # select inside points = cropped
    inliers_pcd = pcd.select_by_index(inliers_indices, invert=False)
    outliers_pcd = pcd.select_by_index(
        inliers_indices, invert=True)  # select outside points
    # print('@len(inliers_pcd.points)',len(inliers_pcd.points))
    return inliers_pcd, outliers_pcd

def sample_points_in_roi(roi_max_x,roi_max_y, num_points,resolution,mask,
        min_distance_threshold = 0.1,
        seed=None,
        verbose=False,
        fail_cnt_thres=1000
    ):
    if seed is not None:
        np.random.seed(seed)

    assert mask.sum() > num_points, "Not enough points in the mask"
    points_pair = []
    if verbose:
        pbar=tqdm(total=num_points)
    fail_cnt=0
    while len(points_pair) < num_points:
        if fail_cnt>fail_cnt_thres:
            assert False, "fail to sample points in roi"
        points = np.random.uniform(low=[0, 0], high=[roi_max_x, roi_max_y], size=(2, 2))
        points = np.floor(points).astype(np.int32)
        
        if not np.all(mask[points[:, 0], points[:, 1]] == 1):
            fail_cnt+=1
            continue
        distance = np.linalg.norm(points[0] - points[1]) * resolution
        if distance < min_distance_threshold:
            fail_cnt+=1
            continue
        fail_cnt=0
        # points_pair.append((points[0], points[1]))
        points_pair.append((points[0], points[1],distance))
        if verbose:
            pbar.update(1)
        
        

    # points_pair = np.array(points_pair)
    return points_pair

def sample_points_in_roi_v2(roi_max_x, roi_max_y, num_points, resolution, mask,
        mask_path,
        min_distance_threshold=0.1,
        seed=None,
        verbose=False,
        fail_cnt_thres=1000
    ):
    if seed is not None:
        np.random.seed(seed)

    assert mask_path.sum() > 0, "Not enough points in the mask_path"
    assert mask.sum() > num_points, "Not enough points in the mask"
    points_pair = []
    if verbose:
        pbar = tqdm(total=num_points)
    fail_cnt = 0
    valid_indices_path = np.argwhere(mask_path == 1)
    valid_indices_mask = np.argwhere(mask == 1)

    while len(points_pair) < num_points:
        if fail_cnt > fail_cnt_thres:
            raise ValueError("Failed to sample points in ROI after maximum attempts")
        
        # Sample point 1 from mask_path
        point1 = valid_indices_path[np.random.randint(len(valid_indices_path))]
        
        # Sample point 2 from mask
        point2 = valid_indices_mask[np.random.randint(len(valid_indices_mask))]
        
        distance = np.linalg.norm(point1 - point2) * resolution
        if distance < min_distance_threshold:
            fail_cnt += 1
            continue

        fail_cnt = 0
        points_pair.append((point1, point2, distance))
        if verbose:
            pbar.update(1)

    return points_pair

def approximate_b_spline_path(x, y, n_path_points, degree=3, with_derivatives=False):
    t = np.arange(len(x))
    x_tup = scipy_interpolate.splrep(t, x, k=degree)
    y_tup = scipy_interpolate.splrep(t, y, k=degree)

    x_list = list(x_tup)
    x_list[1] = np.concatenate([x, np.zeros(degree)])
    
    y_list = list(y_tup)
    y_list[1] = np.concatenate([y, np.zeros(degree)])

    ipl_t = np.linspace(0.0, len(x) - 1, n_path_points)
    rx = scipy_interpolate.splev(ipl_t, x_list)
    ry = scipy_interpolate.splev(ipl_t, y_list)

    path = np.column_stack((rx, ry))

    if with_derivatives:
        rx_der = scipy_interpolate.splev(ipl_t, x_list, der=1)
        ry_der = scipy_interpolate.splev(ipl_t, y_list, der=1)
        derivatives = np.column_stack((rx_der, ry_der))
        return path, derivatives
    else:
        return path


def sampling_occ_from_pc(scene_points,cls_label, 
        pc_range=[-40, -40, -1, 40, 40, 5.4], 
        voxel_size=0.4, 
        occ_size=[200, 200, 16],
        empty_cls=17
    ):
    """
    scene_points: n*3 , points in World
    cls_label: n , semantic of points
    pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    voxel_size: 0.5
    occ_size: [200, 200, 16]

    return: dense_voxels_with_semantic, voxel_semantic
    dense_voxels_with_semantic: n*4 (x,y,z,semantic)
    voxel_semantic: 200*200*16
    """
    ################## remain points with a spatial range ##############
    mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
           & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
    assert mask.sum() > 0, "No points in the scene"       
    scene_points = scene_points[mask]

    ################## convert points to voxels ##############
    # ego -> voxel
    pcd_np = scene_points.copy()
    pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
    pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
    pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
    pcd_np = np.floor(pcd_np).astype(np.int32)
    voxel = np.zeros(occ_size)
    voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

    ################## convert voxel coordinates to LiDAR system  ##############
    gt_ = voxel
    x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
    y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
    z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    fov_voxels = vv[gt_ > 0]
    fov_voxels_idx = vv[gt_ > 0].astype(np.int32)
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]

    ################## Nearest Neighbor to assign semantics ##############
    dense_voxels = fov_voxels
    sparse_voxels_semantic = cls_label[mask]

    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(scene_points)
    dist, indices = knn.kneighbors(dense_voxels)

    dense_semantic = sparse_voxels_semantic[indices[:, 0]]
    dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

    # to voxel coordinate
    pcd_np = dense_voxels_with_semantic
    pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
    pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
    pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
    dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int32)

    voxel_semantic = np.ones(occ_size).astype(np.int8)*empty_cls
    voxel_semantic[fov_voxels_idx[:,0],fov_voxels_idx[:,1],fov_voxels_idx[:,2]]=dense_semantic

    return dense_voxels_with_semantic, voxel_semantic



def get_3d_pose_from_2d(path,grad,origin,resolution,z=0,plane_model=None):
    res = []
    for p, d in zip(path, grad):
        p_w=(p + 0.5) * resolution + np.array(origin)
        pose = np.eye(4)
        d_normalized = d / np.linalg.norm(d)
        pose[:2, 3] = p_w
        if plane_model is not None:
            z=-(plane_model[3]+plane_model[0]*p_w[0]+plane_model[1]*p_w[1])/plane_model[2]
        pose[2,3]=z
        pose[:2, 0] = d_normalized
        pose[:2, 1] = [-d_normalized[1], d_normalized[0]]  # Perpendicular vector
        res.append(pose)
    return np.array(res)


def create_bev_from_pc(occ_pc_road, resolution_2d, max_dist):
    occ_pc_road = occ_pc_road[:,:2]
    x_min, x_max = occ_pc_road[:,0].min(), occ_pc_road[:,0].max()
    y_min, y_max = occ_pc_road[:,1].min(), occ_pc_road[:,1].max()
    n_x = int((x_max - x_min) / resolution_2d)
    n_y = int((y_max - y_min) / resolution_2d)

    # Create grid coordinates efficiently
    x_coords = np.linspace(x_min + resolution_2d/2, x_max - resolution_2d/2, n_x)
    y_coords = np.linspace(y_min + resolution_2d/2, y_max - resolution_2d/2, n_y)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords,indexing='ij')
    grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Fit KNN model and compute distances
    knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', n_jobs=-1).fit(occ_pc_road)
    distances, _ = knn.kneighbors(grid_coords)
    mean_distances = distances.mean(axis=1)

    # Create and populate nd and occ_2d arrays
    nearest_distance = mean_distances.reshape(n_x,n_y)
    bev_road = (nearest_distance < max_dist).astype(np.uint8)

    return bev_road, nearest_distance, (x_min, y_min,n_x,n_y)


def ransac(point_cloud, distance_threshold=0.33, ransac_n=3, num_iterations=100):
    """
    RANSAC-based plane segmentation for a point cloud.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        distance_threshold (float, optional): The maximum distance a point can be from the plane to be considered an inlier.
            Default is 0.33.
        ransac_n (int, optional): The number of points to randomly sample for each iteration of RANSAC. Default is 3.
        num_iterations (int, optional): The number of RANSAC iterations to perform. Default is 100.

    Returns:
        open3d.geometry.PointCloud, open3d.geometry.PointCloud: Two point clouds representing the inliers and outliers
        of the segmented plane, respectively.
    """
    # Perform plane segmentation using RANSAC
    if isinstance(point_cloud,np.ndarray):
        point_cloud=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n,
                                                     num_iterations=num_iterations)

    # Extract inlier and outlier point clouds
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    # Color the outlier cloud red and the inlier cloud blue
    outlier_cloud.paint_uniform_color([0.8, 0.2, 0.2])  # Red
    inlier_cloud.paint_uniform_color([0.25, 0.5, 0.75])  # Blue

    return outlier_cloud, inlier_cloud, inliers ,plane_model

def downsample_pc_with_label(pc, voxel_size, cls_label,return_np=True):
    if isinstance(pc,np.ndarray):
        pc=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
    # Voxel downsample the point cloud and trace the original indices
    pc_ds = pc.voxel_down_sample(voxel_size)
    print(f'@before downsampled pc {len(pc.points)}', f'@after downsampled pc {len(pc_ds.points)}')

    _, indices = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.asarray(pc.points)).kneighbors(np.asarray(pc_ds.points))
    # Update the cls_label to match the downsampled point cloud
    cls_label_ds = cls_label[indices[:, 0]]
    if return_np:
        return np.asarray(pc_ds.points), cls_label_ds
    else:
        return pc_ds, cls_label_ds
    
def get_mask_from_path(path, occ_size, expand_radius=2):
    mask = np.zeros(occ_size, dtype=np.uint8)
    for p in path:
        x, y = p[0], p[1]
        x_min, x_max = max(0, x - expand_radius), min(occ_size[0], x + expand_radius + 1)
        y_min, y_max = max(0, y - expand_radius), min(occ_size[1], y + expand_radius + 1)
        mask[x_min:x_max, y_min:y_max] = 1
    # smooth mask
    mask=cv2.GaussianBlur(mask,(5,5),0)
    # binary mask
    mask=cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)[1] 
    return mask