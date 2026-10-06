_base_ = ['./alocc_3d_r50_256x704_bevdet_preatrain_16f.py']

data = dict(
    val=dict(
        use_corner_case_data='../nuScenes-C/raw/image/nuScenes-c/ColorQuant',
        corner_case_degree='hard',
    ),
    test=dict(
        use_corner_case_data='../nuScenes-C/raw/image/nuScenes-c/ColorQuant',
        corner_case_degree='hard',
    ),
)
