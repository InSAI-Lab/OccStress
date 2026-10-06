from utils import (
    visualize_point_cloud,
    vis_pose_mesh,
    colors, write_pc,
    get_inliers_outliers,
    sample_points_in_roi,
    sample_points_in_roi_v2,
    approximate_b_spline_path,
    sampling_occ_from_pc,
    get_3d_pose_from_2d,
    create_bev_from_pc,
    ransac,
    downsample_pc_with_label,
    get_mask_from_path
)
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from astar import GridWithWeights, a_star_search, reconstruct_path
import time
import os
from tqdm import tqdm
from functools import partial
import joblib
import cv2
import pickle
import shutil
from pyquaternion import Quaternion


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--imageset', type=str, default='../data/nuscenes_infos_train_temporal_v3_scene.pkl')
    parser.add_argument('--input_dataset', type=str, default='gts')
    parser.add_argument('--data_path', type=str, default='../data/nuscenes')
    

    return parser.parse_args()

# Configuration
args = parse_args()
idx = args.idx
dst_dir = args.dst
os.makedirs(dst_dir, exist_ok=True)


moving_cls_id = [
    2,  # 'bicycle'
    3,  # 'bus'
    4,  # 'car'
    6,  # motorcycle
    7,  # 'pedestrian'
    9,  # trailer
    10, # 'truck'
]
road_cls_id = 11

## voxelization param
pc_voxel_downsample=0.2
# resolution_2d=1
resolution_2d=2
max_dist=0.5


## st en point sampling param
path_expand_radius=2
seed=0
n_sample_pair=10
min_distance_st_en=10

## A* param
distance_cost_weigth=250
n_traj_point_ds=4

## traj valid check
min_min_traj_len_st_en=2
delta_min_traj_len_st_en=10
min_traj_len_st_en=30
max_traj_len_st_en=50
fail_cnt_thres=10

## resampling occ from pc
n_sample_occ=40 # each traj only sample 40
voxel_size= 0.4
pc_range=  [-40, -40, -1, 40, 40, 5.4]
occ_size=  [200, 200, 16]

###################################

# data=np.load(f'./occ_{idx}.npz')
# occ,trans,rot=data['occ'],data['e2g_rel0_t'],data['e2g_rel0_r']
def load_occ(args):
    with open(args.imageset, 'rb') as f:
        data = pickle.load(f)
    nusc_infos = data['infos']
    assert args.idx<len(nusc_infos)
    scene_name = list(nusc_infos.keys())[args.idx]
    occs = []
    e2g_t=[]
    e2g_r=[]
    poses=[]
    for idx,item in enumerate(nusc_infos[scene_name]):
        token = item['token']
        label_file = os.path.join(args.data_path, f'{args.input_dataset}/{scene_name}/{token}/labels.npz')
        label = np.load(label_file)
        occs.append(label['semantics'])
        e2g_t.append(item['ego2global_translation'])
        e2g_r.append(item['ego2global_rotation'])
        # # save front cam for debug
        # front_cam_src=os.path.join(args.data_path,os.path.join(item['cams']['CAM_FRONT']['data_path']).replace('./data/nuscenes/',''))
        # shutil.copyfile(front_cam_src,f'{dst_dir}/front_cam_{idx}.png')
        # save pose
        pose=np.eye(4)
        pose[:3,:3]=Quaternion(item['ego2global_rotation']).rotation_matrix
        pose[:3,3]=item['ego2global_translation']
        poses.append(pose)
        # save raw_scene
        dir=f'{dst_dir}/raw_scene/traj-{idx:06d}'
        os.makedirs(dir,exist_ok=True)
        np.savez(f'{dir}/labels.npz',semantics=label['semantics'],pose=pose)

    occs = np.stack(occs, dtype=np.int64)
    e2g_t=np.array(e2g_t)
    e2g_r=np.array(e2g_r)
    poses=np.stack(poses)
    return occs, e2g_t, e2g_r,poses

occ,trans,rot,poses=load_occ(args)

visualize_point_cloud(
    occ,
    trans,
    rot,
    cmp_dir=dst_dir,
    frame_type='',
    # frame_type='e+w',
    key='gt',
    save_label=True,
)

occ_pc_all=o3d.io.read_point_cloud(f'{dst_dir}/vis_gt_all_w.ply')
cls_label=np.load(f'{dst_dir}/cls_label_gt_all_w.npy')

occ_pc_all_np=np.asarray(occ_pc_all.points)
occ_pc_road=occ_pc_all_np[cls_label==road_cls_id]

mask_static=~np.isin(cls_label,moving_cls_id)
occ_pc_static=occ_pc_all_np[mask_static]
cls_label=cls_label[mask_static]
occ_pc_static,cls_label=downsample_pc_with_label(occ_pc_static,pc_voxel_downsample,cls_label)
write_pc(occ_pc_static,f'{dst_dir}/vis_gt_all_w_ds.ply',colors[cls_label-1][:,:3]/255.0)

# pca get thrid dim
# z_road=np.mean(occ_pc_road[:,2])
_, inlier_cloud, _ ,plane_model=ransac(occ_pc_road)
# o3d.io.write_point_cloud(f'{dst_dir}/vis_gt_road_plane.ply',inlier_cloud)  #debug
# exit(0)
assert len(occ_pc_road)>0

# save pc
# repeat n times
# write_pc(occ_pc_road,f'{dst_dir}/vis_gt_road.ply',colors[None,road_cls_id-1].repeat(len(occ_pc_road),axis=0)[:,:3]/255.0) #debug

# create bev map
bev_road, nearest_distance, (x_min, y_min,n_x,n_y) = create_bev_from_pc(occ_pc_road, resolution_2d, max_dist)
distance_cost=distance_cost_weigth*nearest_distance
origin_path=trans.copy()
origin_path=((origin_path-np.array([x_min,y_min,0]))/resolution_2d).astype(np.int32)

# Generate mask from the original path
mask_origin_path = get_mask_from_path(origin_path, (n_x, n_y), expand_radius=path_expand_radius)

# Refine the mask by considering only areas that are both in the original path and on the road
mask_origin_path = np.logical_and(mask_origin_path > 0, bev_road > 0).astype(np.uint8)

# Calculate map Paramaters
bev_map=GridWithWeights.from_voxel_bev_map(bev_road,cost_map=distance_cost)

n_r=n_c=np.ceil(n_sample_pair**0.5).astype(np.int32)
fig,ax=plt.subplots(n_r,n_c,figsize=(5*n_c,5*n_r))
ax=ax.flatten()

sampled_trajs = []
valid_path_count=0
np.random.seed(seed)

# Plotting
ax[0].imshow(bev_road)
# draw raw path
ax[1].imshow(mask_origin_path)
ax[1].plot(origin_path[:,1], origin_path[:,0], linewidth=1.5, color='k', zorder=0,alpha=0.8)
ax[1].scatter(origin_path[0,1], origin_path[0,0], marker='o', c='r')
ax[1].scatter(origin_path[-1,1], origin_path[-1,0], marker='x', c='b')
ax[1].set_title(f'origin path')#, area {mask_origin_path.sum()}')
plt.tight_layout()
plt.savefig(f'{dst_dir}/sampled_trajectories.png', dpi=300, bbox_inches='tight')

fail_cnt=0
fail_v2_flag=False
pbar=tqdm(total=n_sample_pair,desc='sampling traj')
while valid_path_count<n_sample_pair:
    if fail_cnt > fail_cnt_thres:
        if min_traj_len_st_en < min_min_traj_len_st_en:
            raise ValueError('Failed too many times: unable to generate valid trajectory')
        elif fail_v2_flag:
            print('Sampling method v1 failed, attempting to reduce min_distance_st_en')
            if min_traj_len_st_en > min_min_traj_len_st_en:
                min_traj_len_st_en = max(min_traj_len_st_en - delta_min_traj_len_st_en, min_min_traj_len_st_en)
            else:
                raise ValueError('Failed too many times: unable to generate valid trajectory')
            fail_cnt = 0
        else:
            print('Sampling method v2 failed, switching to v1')
            fail_v2_flag = True
            fail_cnt = 0
    # sample points in roi
    if not fail_v2_flag:
    # if mask_origin_path.sum()>min_path_area:
        st, en, dist=sample_points_in_roi_v2(bev_road.shape[0],bev_road.shape[1],num_points=1,resolution=resolution_2d,mask=bev_road,
                                         mask_path=mask_origin_path,
                                         min_distance_threshold=min_distance_st_en,verbose=False)[0]
    else:
        st, en, dist=sample_points_in_roi(bev_road.shape[0],bev_road.shape[1],num_points=1,resolution=resolution_2d,mask=bev_road,min_distance_threshold=min_distance_st_en,verbose=False)[0]
        
    start, goal = tuple(st), tuple(en)
    # tic = time.time()
    came_from, cost_so_far = a_star_search(bev_map, start, goal)
    # print('A* Time:', time.time() - tic)
    
    if len(came_from) <= 1 or goal not in came_from:
        print('@filtered, no solution')
        fail_cnt+=1
        continue
    
    path = np.array(reconstruct_path(came_from, start=start, goal=goal))
    path_raw=path.copy()
    
    print(f'@len traj {len(path_raw)} @dist {dist}')
    if len(path_raw) < min_traj_len_st_en:
        print('@filtered, too short traj')
        fail_cnt+=1
        continue
    elif len(path_raw) > max_traj_len_st_en:
        print('@filtered, too long traj')
        fail_cnt+=1
        continue
    
    fail_cnt=0
    if n_traj_point_ds>1:   
        path = np.concatenate([path[:-1:n_traj_point_ds], path[-1:]], axis=0)
    degree=min(5,len(path)-1)
    path_smooth = approximate_b_spline_path(path[:, 0], path[:, 1], n_path_points=1000, degree=degree)
    path_occ, dd = approximate_b_spline_path(path[:, 0], path[:, 1], n_path_points=n_sample_occ, degree=degree,with_derivatives=True)
    sampled_trajs.append(get_3d_pose_from_2d(path_occ, dd, [x_min, y_min], resolution_2d,plane_model=plane_model))
    pbar.update(1)
    # Plotting
    ax[valid_path_count+2].imshow(bev_road)
    # draw raw path
    # ax[valid_path_count].plot(origin_path[:,1], origin_path[:,0], linewidth=1.5, color='k', zorder=0,alpha=0.8)
    ax[valid_path_count+2].plot(path_raw[:,1], path_raw[:,0], linewidth=1.5, color='g', zorder=0,alpha=0.8)
    ax[valid_path_count+2].plot(path[:,1], path[:,0], linewidth=1.5, color='b', zorder=0,alpha=0.8)
    ax[valid_path_count+2].plot(path_smooth[:,1], path_smooth[:,0], linewidth=1.5, color='r', zorder=0,alpha=0.8)
    ax[valid_path_count+2].scatter(st[1], st[0], marker='o', c='r')
    ax[valid_path_count+2].scatter(en[1], en[0], marker='x', c='b')
    ax[valid_path_count+2].set_title(f'resample {valid_path_count}\ndist: {dist:.2f}, traj: {len(path_raw)}')
    valid_path_count += 1
    # break #debug
    

plt.tight_layout()
plt.savefig(f'{dst_dir}/sampled_trajectories.png', dpi=300, bbox_inches='tight')

sampled_trajs=np.array(sampled_trajs)
np.save(f'{dst_dir}/sampled_trajectories.npy',sampled_trajs)
assert len(sampled_trajs)==n_sample_pair

# ## debug use orgin
# sampled_trajs=np.array([np.eye(4)]*len(trans))
# sampled_trajs[:,:3,:3]=np.array([Quaternion(r).rotation_matrix for r in rot])
# sampled_trajs[:,:3,3]=trans
# sampled_trajs=sampled_trajs[None]
# n_sample_pair=1

def process_trajectory(j, sampled_traj, occ_pc_static, cls_label, pc_range, voxel_size, occ_size, dst_dir_i):
    dst_dir_j = f'{dst_dir_i}/traj-{j:06d}'  # TODO: use local id
    os.makedirs(dst_dir_j, exist_ok=True)
    
    # Transform to ego coordinates
    w2e = np.linalg.inv(sampled_traj)
    occ_pc_static_e = occ_pc_static @ w2e[:3,:3].T + w2e[:3,3]
    
    dense_voxels_with_semantic, voxel_semantic = sampling_occ_from_pc(occ_pc_static_e, cls_label, pc_range, voxel_size, occ_size)
    np.savez(f'{dst_dir_j}/labels.npz', semantics=voxel_semantic,pose=sampled_traj)
    # write_pc(dense_voxels_with_semantic[:,:3],f'{dst_dir_j}/occ_vis.ply',colors[dense_voxels_with_semantic[:,3]-1][:,:3]/255.0)  #debug
    return voxel_semantic

for i in tqdm(range(n_sample_pair),desc='resampling occ'):
    # if i>0:
    #     break #debug

    voxel_semantic_all=[]
    # vis_pose_mesh(sampled_trajs[i],cmp_dir=dst_dir,fn=f'vis_resample_all_w_traj_sample_{i}.ply')  #debug
    dst_dir_i=f'{dst_dir}/scene-{i:04d}'  #TODO global id

    process_func = partial(process_trajectory, 
                           occ_pc_static=occ_pc_static, 
                           cls_label=cls_label, 
                           pc_range=pc_range, 
                           voxel_size=voxel_size, 
                           occ_size=occ_size, 
                           dst_dir_i=dst_dir_i)
    
    voxel_semantic_all = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_func)(j, sampled_traj)
        for j, sampled_traj in enumerate(sampled_trajs[i])
    )
    # for j in range(len(sampled_trajs[i])):
    #     tic=time.time()
    #     voxel_semantic_all.append(process_func(j, sampled_trajs[i][j]))
    #     print(f'@process_func {time.time()-tic}')
        
    visualize_point_cloud(
        voxel_semantic_all,
        None,
        None,
        abs_trans=sampled_trajs[i],
        cmp_dir=dst_dir_i,
        frame_type='',
        # frame_type='e+w',
        key='resample'
    )
