# FusionOcc
> **FusionOcc: Multi-Modal Fusion for 3D Occupancy Prediction, MM 2024** [[paper](https://dl.acm.org/doi/10.1145/3664647.3681293)]

## INTRODUCTION
FusionOcc is a new multi-modal fusion network for 3D occupancy prediction by fusing features of LiDAR point clouds and surround-view images. The model fuses features of these two modals in 2D and 3D space, respectively. Semi-supervised method is utilized to generate dense depth map, which is integrated by BEV images via a cross-modal fusion module. Features of voxelized point clouds are aligned and merged with BEV images' features converted by a view-transformer in 3D space. FusionOcc establishes a new baseline for further research in multi-modal fusion for 3D occupancy prediction, while achieves the new state-of-the-art on Occ3D-nuScenes dataset.

![pipeline](assets/pipeline.png)

## Getting Started

- [Installation](docs/install.md)
```
# main prerequisites 
Python = 3.8
nuscenes-devkit = 1.1.11
PyTorch = 1.10.0
torch-scatter = 2.0.9
opencv-python = 4.9.0
Pillow = 10.0.1
mmcv-ful = 1.5.3
mmdetection = 2.25.1
```

- [Datasets](docs/datasets.md) 

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


## Model Zoo

| Backbone | Config | Mask | Pretrain | mIoU | Checkpoints | 
| :-------: | :---: | :---: | :---: | :---: | :---: |
| Swin-Base | [Base](configs/fusion_occ) | ✖️ | ImageNet, nuImages | 56.62 | [BaseWoMask](https://drive.google.com/file/d/16ELoDLoDkCYheREJUPiBz2905MHhuVHv/view) |
<!-- | Swin-Base | [Base](configs/) | ✔️ | ImageNet | 35.94 | [BaseMask](checkpoints/) |
-->
<!-- | ViT-Tiny | [Light](configs/) | ✔️ |  |  |  |
| ViT-Tiny | [Light](configs/) | ✖️ |  |  |  | -->

## Evaluation

We provide instructions for evaluating our pretrained models. Download checkpoints above first.

the config file is here [fusion_occ.py](configs/fusion_occ/fusion_occ.py )

Run:
```bash
./tools/dist_test.sh $config $checkpoint num_gpu
```

## Training

Modify the "load_from" path at the end of the config file to load pre-trained weights, run:

```bash
./tools/dist_train.sh $config num_gpu
```

To obtain the version without using mask, simply modify the use_mask field in the config file to False and train several epochs.

You can also acquire pre-trained weights from [BEVDet](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docker/Dockerfile)
 to start training from the very beginning.



## Acknowledgement

Thanks a lot to these excellent open-source projects, our code is based on them:
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [Occ3d](https://github.com/Tsinghua-MARS-Lab/Occ3D), [CVPR23-Occ-Chanllege](https://github.com/CVPR2023-3D-Occupancy-Prediction)

Some other related projects for Occ3d prediction:
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc), [TPVFormer](https://github.com/wzzheng/TPVFormer)
- [PanoOcc](https://github.com/Robertwyq/PanoOcc), [RenderOcc](https://github.com/pmj110119/RenderOcc)


## BibTeX

If this work is helpful for your research, please consider citing the following paper:

```
@inproceedings{
    zhang2024fusionocc,
    title={FusionOcc: Multi-Modal Fusion for 3D Occupancy Prediction},
    author={Shuo Zhang and Yupeng Zhai and Jilin Mei and Yu Hu},
    booktitle={ACM Multimedia 2024},
    year={2024},
    url={https://openreview.net/forum?id=xX66hwZJWa}
}
