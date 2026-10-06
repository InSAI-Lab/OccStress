**II-World OccStress Integration**

这套接入是增量的，不覆盖 clean 路径。

**新增文件**

- `EXIST/4D/II-World/mmdet3d/datasets/occstress_world_dataset.py`
- `EXIST/4D/II-World/mmdet3d/datasets/pipelines/loading_occstress.py`
- `EXIST/4D/II-World/mmdet3d/models/ii_world/scene_tokenizer/ii_tokenizer_occstress.py`
- `EXIST/4D/II-World/configs/scene_tokenizer/ii_scene_tokenizer_occstress.py`
- `EXIST/4D/II-World/configs/world_model/ii_generate_world_occstress.py`

**新增数据索引**

- `EXIST/4D/II-World/data/nuscenes/world-nuscenes_infos_trainval.pkl`

这个索引仍然保留，但当前主评测已经切换到：

- strict `4519`-sample clean `val` backbone
- `data/OccStress/protocols/*_val_backbone.pkl`

**clean 路径保持不变**

- clean tokenizer 权重：
  - `EXIST/4D/II-World/work_dirs/ii_scene_tokenizer_4f/iter_47976_ema.pth`
- clean world 权重：
  - `EXIST/4D/II-World/work_dirs/ii_generate_world_4gpu_bs16/iter_95952_ema.pth`
- clean latent token：
  - `EXIST/4D/II-World/data/nuscenes/save_dir/token_4f`

**OccStress 路径单独存**

- protocol：
  - `data/OccStress/protocols/*.pkl`
- OccStress tokenizer latent：
  - `EXIST/4D/II-World/data/OccStress/save_dir/<protocol_name>/token_4f/<scene>/<protocol_sample_id>.npz`

**当前接入方式**

1. `OccStressNuScenesTokenizerDataset`
   - 读取 `OccStress` protocol。
   - 把每个 protocol clip 展平成 `4 history + 1 current` 的流式序列。
   - `OccStressIISceneTokenizer` 只在 anchor 帧保存 token。
   - 保存文件名用 `protocol_sample_id`，不会覆盖 clean token。

2. `OccStressNuScenesWorldDataset`
   - 读取 `OccStress` protocol。
   - 当前帧 latent 从 OccStress token 根目录读取。
   - future latent 继续从 clean `token_4f` 读取。
   - GT occupancy 从 protocol 指向的 clean target/future 路径读取。

**验证状态**

- `configs/scene_tokenizer/ii_scene_tokenizer_occstress.py`
  - dataset 能正常 `build_dataset + __getitem__`
- `configs/world_model/ii_generate_world_occstress.py`
  - dataset 能正常 `build_dataset + get_data_info`
  - 如果没有先生成 OccStress token，`__getitem__` 会显式报缺失文件，这是预期行为

**建议使用顺序**

1. 先选一个 protocol
2. 用 `ii_scene_tokenizer_occstress.py` 生成该 protocol 的 OccStress token
3. 再用 `ii_generate_world_occstress.py` 跑 world eval

**当前 protocol 语义**

- `all_frame`: 4 帧 history 被污染，current 保持 clean
- `history_k1`: 最近一帧 history 和 current 一起被污染
- `current`: 仅 current input 被污染，current target 保持 clean

**当前默认 protocol**

- `semantic_easy_history_k1_H4_F6_val_backbone`

如果切换 protocol，至少同时改这几个变量：

- `protocol_path`
- `occstress_save_root` 或 `occstress_token_root`

不要改 clean 的：

- `data/nuscenes/save_dir/token_4f`
- clean checkpoints
