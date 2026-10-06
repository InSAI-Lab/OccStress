import os

from .builder import custom_build_dataset
from .waymo_temporal_zlt import CustomWaymoDataset_T

__all__ = ["CustomWaymoDataset_T"]

if os.environ.get("CVTOCC_WAYMO_ONLY") != "1":
    from .nuscenes_dataset import CustomNuScenesDataset
    from .nuscenes_occ import NuSceneOcc

    __all__ += ["CustomNuScenesDataset", "NuSceneOcc"]
