import os

from .occformer_waymo import CVTOccWaymo

if os.environ.get("CVTOCC_WAYMO_ONLY") != "1":
    from .occformer import CVTOcc
    from .centerpoint_solo import CenterPoint_solo
    from .bevdet_solo import BEVDet_solo
    from .solofusion import SOLOFusion
