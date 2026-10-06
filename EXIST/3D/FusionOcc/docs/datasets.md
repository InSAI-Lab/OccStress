# Prepare Datasets
Currently supported datasets: Occ3D-nuScenes.

## Occ3D-nuScenes
Download nuScenes V1.0 full from [here](https://www.nuscenes.org/download) to `data/nuscenes`, nuScenes-lidarseg from [here](https://www.nuscenes.org/download), GTs of Occ(gts only) from [here](https://github.com/Tsinghua-MARS-Lab/Occ3D). \
Prepare nuScenes dataset as below,

```
FusionOcc
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg
│   │   ├── gts
|   |   ├── v1.0-trainval
```

Create the pkl file:
```python
python tools/create_data_fusionocc.py
```
Generate the image segmentation labels (takes a long time) by running:
```shell
python img_seg/gen_segmap.py data/nuscenes --parallel=32
```

After processing, the data structure is as follows:
```
FusionOcc
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg
│   │   ├── imgseg
│   │   ├── gts
|   |   ├── v1.0-trainval
|   |   ├── fusionocc-nuscenes_infos_train.pkl
|   |   ├── fusionocc-nuscenes_infos_val.pkl
```