import os
import argparse
from multiprocessing import Process

from nuscenes.nuscenes import NuScenes

from lidar.lidar_anno import nuScenesLidarSeg
from helper import *


def gen_seg_map(start_idx, end_idx, nusc, lidar_seg_nus, down_sample, proj_lidar=False, save_dir=None):
    for i, scene in enumerate(nusc.scene[start_idx:end_idx]):
        sample = nusc.get('sample', scene['first_sample_token'])
        while True:
            lidar_seg = lidar_seg_nus.get_lidar_seg(sample["token"])
            process_one_sample(nusc,
                               sample,
                               down_sample,
                               lidar_seg=lidar_seg,
                               proj_lidar=proj_lidar,
                               save_dir=save_dir)
            if sample['next'] == '':
                break
            sample = nusc.get('sample', sample['next'])


def gen_labels(nusc, lidar_seg_nus, down_sample, parallel=1, proj_lidar=False, visible_level=2, save_dir=None):
    total_n = len(nusc.scene)
    interval = total_n // parallel
    processes = []
    for i in range(parallel):
        start_idx = i * interval
        end_idx = total_n if i == parallel - 1 else (i + 1) * interval
        p = Process(target=gen_seg_map,
                    args=(start_idx, end_idx,
                          nusc, lidar_seg_nus, down_sample, proj_lidar, save_dir
                          )
                    )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 2d images seg label')
    parser.add_argument('data_root', help='data root of nuscenes')
    parser.add_argument('--down_sample', type=int, default=8, help='down sample seg img')
    parser.add_argument('--parallel', type=int, default=1, help='parallel processing num')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root
    version = "v1.0-trainval"
    save_dir = os.path.join(args.data_root, "imgseg")
    os.makedirs(save_dir, exist_ok=True)
    down_sample = args.down_sample
    parallel = args.parallel
    nusc = NuScenes(version=version,
                    dataroot=data_root,
                    verbose=True)
    lidar_seg_nus = nuScenesLidarSeg(nusc=nusc, data_path=data_root, version=version)
    gen_labels(nusc, lidar_seg_nus, down_sample=down_sample, parallel=parallel,
               proj_lidar=True, save_dir=save_dir)
