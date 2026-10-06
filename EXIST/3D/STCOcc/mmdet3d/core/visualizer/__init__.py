# Copyright (c) OpenMMLab. All rights reserved.
try:
    from .show_result import (show_multi_modality_result, show_result,
                              show_seg_result)
    __all__ = ['show_result', 'show_seg_result', 'show_multi_modality_result']
except ModuleNotFoundError as exc:
    if exc.name != 'trimesh':
        raise

    def _missing_trimesh(*args, **kwargs):
        raise ModuleNotFoundError(
            'trimesh is required for visualization utilities in '
            'mmdet3d.core.visualizer.show_result')

    show_result = _missing_trimesh
    show_seg_result = _missing_trimesh
    show_multi_modality_result = _missing_trimesh
    __all__ = ['show_result', 'show_seg_result', 'show_multi_modality_result']
