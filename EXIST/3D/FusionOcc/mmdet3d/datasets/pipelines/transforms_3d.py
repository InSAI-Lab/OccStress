import torch
import numpy as np
from pyquaternion import Quaternion

from ..builder import PIPELINES


@PIPELINES.register_module()
class PointsLidar2Ego(object):
    def __call__(self, input_dict):
        points = input_dict['points']
        lidar2ego_rots = torch.tensor(Quaternion(input_dict['curr']['lidar2ego_rotation']).rotation_matrix).float()
        lidar2ego_trans = torch.tensor(input_dict['curr']['lidar2ego_translation']).float()
        points.tensor[:, :3] = (
                points.tensor[:, :3] @ lidar2ego_rots.T
        )
        points.tensor[:, :3] += lidar2ego_trans
        input_dict['points'] = points
        return input_dict


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        eps = 0.001
        self.pcd_range = [
            self.pcd_range[0] + eps, self.pcd_range[1] + eps, self.pcd_range[2] + eps,
            self.pcd_range[3] - eps, self.pcd_range[4] - eps, self.pcd_range[5] - eps
        ]

        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str
