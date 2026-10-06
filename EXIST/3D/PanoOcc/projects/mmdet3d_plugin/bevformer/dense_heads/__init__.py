from .pano_occ_head import PanoOccHead

# PanoOcc_small occupancy eval does not need panoptic/sparse heads.
try:
    from .panoseg_occ_head import PanoSegOccHead
except Exception:
    PanoSegOccHead = None
try:
    from .panoseg_occ_sparse_head import SparseOccupancyHead
except Exception:
    SparseOccupancyHead = None

__all__ = ["PanoOccHead"]
if PanoSegOccHead is not None:
    __all__.append("PanoSegOccHead")
if SparseOccupancyHead is not None:
    __all__.append("SparseOccupancyHead")
