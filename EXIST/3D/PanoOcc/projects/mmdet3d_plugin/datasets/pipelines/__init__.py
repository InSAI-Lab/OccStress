# Register mmdet3d transforms into mmdet PIPELINES for PanoOcc CustomCompose.
from mmdet.datasets.builder import PIPELINES as _MMDET_PIPELINES
from mmdet3d.datasets.pipelines import (
    LoadMultiViewImageFromFiles, DefaultFormatBundle3D, MultiScaleFlipAug3D)
for _cls in (LoadMultiViewImageFromFiles, DefaultFormatBundle3D, MultiScaleFlipAug3D):
    if _cls.__name__ not in _MMDET_PIPELINES.module_dict:
        _MMDET_PIPELINES.register_module(module=_cls)

from .transform_3d import (
    PadMultiViewImage, NormalizeMultiviewImage,ResizeCropFlipImage,RandomMultiScaleImageMultiViewImage,
    PhotoMetricDistortionMultiViewImage, CustomCollect3D, RandomScaleImageMultiViewImage)
from .formating import CustomDefaultFormatBundle3D
from .loading import LoadDenseLabel
__all__ = [
    'PadMultiViewImage', 'NormalizeMultiviewImage', 'ResizeCropFlipImage','RandomMultiScaleImageMultiViewImage','LoadDenseLabel',
    'PhotoMetricDistortionMultiViewImage', 'CustomDefaultFormatBundle3D', 'CustomCollect3D', 'RandomScaleImageMultiViewImage'
]