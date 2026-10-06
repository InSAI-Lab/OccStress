# Copyright (c) OpenMMLab. All rights reserved.
from .indoor_eval import indoor_eval
from .instance_seg_eval import instance_seg_eval
from .kitti_utils import kitti_eval, kitti_eval_coco_style
from .seg_eval import seg_eval

try:
    from .lyft_eval import lyft_eval
except ImportError as exc:
    _LYFT_IMPORT_ERROR = exc

    def lyft_eval(*args, **kwargs):
        raise ImportError(
            'lyft_dataset_sdk is required for Lyft evaluation, but it is not '
            'installed in the current environment.') from _LYFT_IMPORT_ERROR

__all__ = [
    'kitti_eval_coco_style', 'kitti_eval', 'indoor_eval', 'lyft_eval',
    'seg_eval', 'instance_seg_eval'
]
