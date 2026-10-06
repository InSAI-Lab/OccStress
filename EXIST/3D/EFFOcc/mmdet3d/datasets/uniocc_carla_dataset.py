# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmcv
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class UniOccCarlaDataset(Custom3DDataset):
    """Single-frame UniOcc-CARLA occupancy dataset.

    Training stays in CARLA's native 11-class ontology. The converter mirrors
    CARLA's left-handed y-right geometry into the model's right-handed y-left
    frame; the Occ3D class mapping is applied only for OccStress export.
    """

    CLASSES = (
        "undefined",
        "car",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "vegetation",
        "road",
        "terrain",
        "building",
        "free",
    )

    def load_annotations(self, ann_file):
        payload = mmcv.load(ann_file, file_format="pkl")
        if isinstance(payload, dict):
            self.metadata = payload.get("metadata", {})
            infos = payload.get("infos")
        else:
            self.metadata = {}
            infos = payload
        if not isinstance(infos, list):
            raise TypeError("UniOcc-CARLA annotations must contain an infos list")
        return infos

    def get_data_info(self, index):
        info = deepcopy(self.data_infos[index])
        required = {
            "sample_idx",
            "pts_filename",
            "occ_gt_path",
            "curr",
            "scene_token",
            "timestamp",
        }
        missing = required - set(info)
        if missing:
            raise KeyError(f"CARLA info {index} is missing {sorted(missing)}")
        info.setdefault("file_name", info["pts_filename"])
        info.setdefault("sample_token", info["sample_idx"])
        return info

    @staticmethod
    def _confusion(prediction, target, mask, num_classes):
        prediction = np.asarray(prediction).reshape(-1)
        target = np.asarray(target).reshape(-1)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        valid = mask & (target >= 0) & (target < num_classes)
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {prediction.shape} != target {target.shape}"
            )
        encoded = num_classes * target[valid].astype(np.int64)
        encoded += prediction[valid].astype(np.int64)
        return np.bincount(
            encoded, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    def evaluate(self, occ_results, runner=None, show_dir=None, **kwargs):
        del runner, show_dir, kwargs
        if len(occ_results) != len(self.data_infos):
            raise ValueError(
                f"received {len(occ_results)} predictions for "
                f"{len(self.data_infos)} samples"
            )

        num_classes = len(self.CLASSES)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for prediction, info in tqdm(
            zip(occ_results, self.data_infos),
            total=len(self.data_infos),
            desc="UniOcc-CARLA evaluation",
        ):
            with np.load(info["occ_gt_path"], allow_pickle=False) as payload:
                target = payload["semantics"]
                mask = payload["mask_camera"]
            confusion += self._confusion(prediction, target, mask, num_classes)

        intersection = np.diag(confusion).astype(np.float64)
        union = confusion.sum(0) + confusion.sum(1) - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.full(num_classes, np.nan, dtype=np.float64),
            where=union > 0,
        )
        occupied_present = np.flatnonzero((union > 0) & (np.arange(num_classes) != 10))

        occupied_tp = confusion[:-1, :-1].sum()
        occupied_fp = confusion[-1, :-1].sum()
        occupied_fn = confusion[:-1, -1].sum()
        binary_union = occupied_tp + occupied_fp + occupied_fn

        metrics = {
            f"IoU/{name}": float(iou[index] * 100.0)
            for index, name in enumerate(self.CLASSES)
            if np.isfinite(iou[index])
        }
        metrics["mIoU/present_occupied"] = float(
            np.nanmean(iou[occupied_present]) * 100.0
        )
        metrics["IoU/binary_occupied"] = float(
            occupied_tp / binary_union * 100.0 if binary_union else 0.0
        )
        return metrics
