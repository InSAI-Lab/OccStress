import os

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadOracleFlowGT(object):
    """Load oracle flow GT for pairwise current/previous fusion.

    This transform is intended to run after ``LoadStreamOcc3D`` so the current
    occupancy tensor shape is already known. It loads the current sample's
    backward flow, which maps current-grid locations back to the previous
    frame. That backward flow can be consumed directly by ``grid_sample`` when
    transporting the previous latent into the current frame.
    """

    def __init__(self, flow_root, flow_key='flow_backward',
                 valid_key='flow_valid_backward',
                 dynamic_key='dynamic_mask',
                 static_flow_key='flow_backward_static',
                 static_valid_key='flow_valid_backward_static',
                 dynamic_residual_flow_key='flow_backward_dynamic_residual',
                 dynamic_residual_valid_key='flow_valid_backward_dynamic_residual'):
        self.flow_root = flow_root
        self.flow_key = flow_key
        self.valid_key = valid_key
        self.dynamic_key = dynamic_key
        self.static_flow_key = static_flow_key
        self.static_valid_key = static_valid_key
        self.dynamic_residual_flow_key = dynamic_residual_flow_key
        self.dynamic_residual_valid_key = dynamic_residual_valid_key

    def _zero_payload(self, results):
        if 'voxel_semantics_clean' in results:
            occ_shape = tuple(results['voxel_semantics_clean'].shape[-3:])
        elif 'voxel_semantics' in results:
            occ_shape = tuple(results['voxel_semantics'].shape[-3:])
        else:
            occ_shape = (200, 200, 16)
        return dict(
            oracle_flow=np.zeros(occ_shape + (3,), dtype=np.float32),
            oracle_flow_valid=np.zeros(occ_shape, dtype=np.bool_),
            oracle_dynamic_mask=np.zeros(occ_shape, dtype=np.bool_),
            oracle_static_flow=np.zeros(occ_shape + (3,), dtype=np.float32),
            oracle_static_flow_valid=np.zeros(occ_shape, dtype=np.bool_),
            oracle_static_flow_forward=np.zeros(occ_shape + (3,), dtype=np.float32),
            oracle_static_flow_valid_forward=np.zeros(occ_shape, dtype=np.bool_),
            oracle_dynamic_residual_flow=np.zeros(occ_shape + (3,), dtype=np.float32),
            oracle_dynamic_residual_flow_valid=np.zeros(occ_shape, dtype=np.bool_),
            oracle_dynamic_residual_flow_forward=np.zeros(occ_shape + (3,), dtype=np.float32),
            oracle_dynamic_residual_flow_valid_forward=np.zeros(occ_shape, dtype=np.bool_),
            oracle_dynamic_mask_source=np.zeros(occ_shape, dtype=np.bool_),
            oracle_has_flow=False,
        )

    def _decomposed_arrays(self, flow_data, full_flow, full_valid, dynamic_mask):
        if (
            self.static_flow_key in flow_data.files
            and self.static_valid_key in flow_data.files
            and self.dynamic_residual_flow_key in flow_data.files
            and self.dynamic_residual_valid_key in flow_data.files
        ):
            return (
                flow_data[self.static_flow_key].astype(np.float32),
                flow_data[self.static_valid_key].astype(np.bool_),
                flow_data[self.dynamic_residual_flow_key].astype(np.float32),
                flow_data[self.dynamic_residual_valid_key].astype(np.bool_),
            )

        # Backward-compatible fallback for legacy archives: treat the old full
        # flow as the static field and expose a zero residual branch.
        static_flow = full_flow.astype(np.float32)
        static_valid = full_valid.astype(np.bool_)
        dynamic_residual_flow = np.zeros_like(full_flow, dtype=np.float32)
        dynamic_residual_valid = np.zeros_like(dynamic_mask, dtype=np.bool_)
        return static_flow, static_valid, dynamic_residual_flow, dynamic_residual_valid

    def __call__(self, results):
        current_token = str(results['sample_idx'])
        current_scene = str(results['scene_name'])
        prev_token = results.get('prev', None)
        has_real_previous = bool(prev_token)
        if has_real_previous:
            prev_token = str(prev_token)
            has_real_previous = prev_token != current_token
        elif results.get('previous_occ_path'):
            previous_paths = results.get('previous_occ_path', [])
            if len(previous_paths) > 0:
                prev_token = os.path.basename(previous_paths[-1].rstrip('/'))
                has_real_previous = prev_token != current_token

        if not has_real_previous:
            results.update(self._zero_payload(results))
            return results

        flow_path = os.path.join(self.flow_root, current_scene, f'{current_token}.npz')
        prev_flow_path = os.path.join(self.flow_root, current_scene, f'{prev_token}.npz')
        if not os.path.exists(flow_path):
            results.update(self._zero_payload(results))
            return results

        flow_data = np.load(flow_path)
        flow_arr = flow_data[self.flow_key].astype(np.float32)
        valid_arr = flow_data[self.valid_key].astype(np.bool_)
        dynamic_arr = flow_data[self.dynamic_key].astype(np.bool_)
        static_flow_arr, static_valid_arr, dynamic_residual_flow_arr, dynamic_residual_valid_arr = self._decomposed_arrays(
            flow_data, flow_arr, valid_arr, dynamic_arr)
        results['oracle_flow'] = flow_arr
        results['oracle_flow_valid'] = valid_arr
        results['oracle_dynamic_mask'] = dynamic_arr
        results['oracle_static_flow'] = static_flow_arr
        results['oracle_static_flow_valid'] = static_valid_arr
        results['oracle_dynamic_residual_flow'] = dynamic_residual_flow_arr
        results['oracle_dynamic_residual_flow_valid'] = dynamic_residual_valid_arr
        if os.path.exists(prev_flow_path):
            prev_flow_data = np.load(prev_flow_path)
            results['oracle_static_flow_forward'] = prev_flow_data['flow_forward_static'].astype(np.float32)
            results['oracle_static_flow_valid_forward'] = prev_flow_data['flow_valid_forward_static'].astype(np.bool_)
            results['oracle_dynamic_residual_flow_forward'] = prev_flow_data['flow_forward_dynamic_residual'].astype(np.float32)
            results['oracle_dynamic_residual_flow_valid_forward'] = prev_flow_data['flow_valid_forward_dynamic_residual'].astype(np.bool_)
            results['oracle_dynamic_mask_source'] = prev_flow_data[self.dynamic_key].astype(np.bool_)
        else:
            zero_payload = self._zero_payload(results)
            results['oracle_static_flow_forward'] = zero_payload['oracle_static_flow_forward']
            results['oracle_static_flow_valid_forward'] = zero_payload['oracle_static_flow_valid_forward']
            results['oracle_dynamic_residual_flow_forward'] = zero_payload['oracle_dynamic_residual_flow_forward']
            results['oracle_dynamic_residual_flow_valid_forward'] = zero_payload['oracle_dynamic_residual_flow_valid_forward']
            results['oracle_dynamic_mask_source'] = zero_payload['oracle_dynamic_mask_source']
        results['oracle_has_flow'] = True
        results['oracle_flow_path'] = flow_path
        return results


@PIPELINES.register_module()
class LoadOracleFlowGTSequence(object):
    """Load adjacent oracle flow GT for a full history window.

    The loaded occupancy sequence from ``LoadStreamOcc3D`` is ordered as:
      `[t-(N-1), ..., t-1, t]`

    This transform returns the sequence of *backward* flows for the current
    step of each adjacent pair:
      `[(t-(N-2))<- (t-(N-1)), ..., t <- (t-1)]`

    Concretely, the output has ``N-1`` flow tensors. Entry ``i`` corresponds
    to the current frame token at ``seq_tokens[i + 1]`` and maps its grid back
    to the previous token ``seq_tokens[i]``.
    """

    def __init__(self,
                 flow_root,
                 flow_key='flow_backward',
                 valid_key='flow_valid_backward',
                 dynamic_key='dynamic_mask',
                 static_flow_key='flow_backward_static',
                 static_valid_key='flow_valid_backward_static',
                 dynamic_residual_flow_key='flow_backward_dynamic_residual',
                 dynamic_residual_valid_key='flow_valid_backward_dynamic_residual'):
        self.flow_root = flow_root
        self.flow_key = flow_key
        self.valid_key = valid_key
        self.dynamic_key = dynamic_key
        self.static_flow_key = static_flow_key
        self.static_valid_key = static_valid_key
        self.dynamic_residual_flow_key = dynamic_residual_flow_key
        self.dynamic_residual_valid_key = dynamic_residual_valid_key

    def _occ_shape(self, results):
        if 'voxel_semantics_clean' in results:
            return tuple(results['voxel_semantics_clean'].shape[-3:])
        if 'voxel_semantics' in results:
            return tuple(results['voxel_semantics'].shape[-3:])
        return (200, 200, 16)

    def _zero_flow(self, occ_shape):
        return (
            np.zeros(occ_shape + (3,), dtype=np.float32),
            np.zeros(occ_shape, dtype=np.bool_),
            np.zeros(occ_shape, dtype=np.bool_),
            np.zeros(occ_shape + (3,), dtype=np.float32),
            np.zeros(occ_shape, dtype=np.bool_),
            np.zeros(occ_shape + (3,), dtype=np.float32),
            np.zeros(occ_shape, dtype=np.bool_),
            np.zeros(occ_shape + (3,), dtype=np.float32),
            np.zeros(occ_shape, dtype=np.bool_),
            np.zeros(occ_shape + (3,), dtype=np.float32),
            np.zeros(occ_shape, dtype=np.bool_),
            np.zeros(occ_shape, dtype=np.bool_),
        )

    def _decomposed_arrays(self, flow_data, full_flow, full_valid, dynamic_mask):
        if (
            self.static_flow_key in flow_data.files
            and self.static_valid_key in flow_data.files
            and self.dynamic_residual_flow_key in flow_data.files
            and self.dynamic_residual_valid_key in flow_data.files
        ):
            return (
                flow_data[self.static_flow_key].astype(np.float32),
                flow_data[self.static_valid_key].astype(np.bool_),
                flow_data[self.dynamic_residual_flow_key].astype(np.float32),
                flow_data[self.dynamic_residual_valid_key].astype(np.bool_),
            )

        static_flow = full_flow.astype(np.float32)
        static_valid = full_valid.astype(np.bool_)
        dynamic_residual_flow = np.zeros_like(full_flow, dtype=np.float32)
        dynamic_residual_valid = np.zeros_like(dynamic_mask, dtype=np.bool_)
        return static_flow, static_valid, dynamic_residual_flow, dynamic_residual_valid

    def __call__(self, results):
        occ_shape = self._occ_shape(results)
        scene_name = str(results['scene_name'])
        current_token = str(results['sample_idx'])
        previous_paths = list(results.get('previous_occ_path', []))

        previous_tokens = [os.path.basename(path.rstrip('/')) for path in previous_paths]
        seq_tokens = previous_tokens + [current_token]
        num_steps = max(len(seq_tokens) - 1, 0)

        flow_seq = []
        valid_seq = []
        dynamic_seq = []
        static_flow_seq = []
        static_valid_seq = []
        static_flow_forward_seq = []
        static_valid_forward_seq = []
        dynamic_residual_flow_seq = []
        dynamic_residual_valid_seq = []
        dynamic_residual_flow_forward_seq = []
        dynamic_residual_valid_forward_seq = []
        dynamic_source_mask_seq = []
        flow_path_seq = []

        for step_idx in range(num_steps):
            prev_token = str(seq_tokens[step_idx])
            curr_token = str(seq_tokens[step_idx + 1])
            real_pair = curr_token != prev_token
            flow_path = os.path.join(self.flow_root, scene_name, f'{curr_token}.npz')
            prev_flow_path = os.path.join(self.flow_root, scene_name, f'{prev_token}.npz')

            if (not real_pair) or (not os.path.exists(flow_path)):
                (flow_arr, valid_arr, dynamic_arr,
                 static_flow_arr, static_valid_arr,
                 static_forward_arr, static_forward_valid_arr,
                 dynamic_residual_flow_arr, dynamic_residual_valid_arr,
                 dynamic_residual_forward_arr, dynamic_residual_forward_valid_arr,
                 dynamic_source_mask_arr) = self._zero_flow(occ_shape)
                flow_path_seq.append('')
            else:
                flow_data = np.load(flow_path)
                flow_arr = flow_data[self.flow_key].astype(np.float32)
                valid_arr = flow_data[self.valid_key].astype(np.bool_)
                dynamic_arr = flow_data[self.dynamic_key].astype(np.bool_)
                static_flow_arr, static_valid_arr, dynamic_residual_flow_arr, dynamic_residual_valid_arr = self._decomposed_arrays(
                    flow_data, flow_arr, valid_arr, dynamic_arr)
                if os.path.exists(prev_flow_path):
                    prev_flow_data = np.load(prev_flow_path)
                    static_forward_arr = prev_flow_data['flow_forward_static'].astype(np.float32)
                    static_forward_valid_arr = prev_flow_data['flow_valid_forward_static'].astype(np.bool_)
                    dynamic_residual_forward_arr = prev_flow_data['flow_forward_dynamic_residual'].astype(np.float32)
                    dynamic_residual_forward_valid_arr = prev_flow_data['flow_valid_forward_dynamic_residual'].astype(np.bool_)
                    dynamic_source_mask_arr = prev_flow_data[self.dynamic_key].astype(np.bool_)
                else:
                    (_, _, _, _, _,
                     static_forward_arr, static_forward_valid_arr,
                     _, _, dynamic_residual_forward_arr, dynamic_residual_forward_valid_arr,
                     dynamic_source_mask_arr) = self._zero_flow(occ_shape)
                flow_path_seq.append(flow_path)

            flow_seq.append(flow_arr)
            valid_seq.append(valid_arr)
            dynamic_seq.append(dynamic_arr)
            static_flow_seq.append(static_flow_arr)
            static_valid_seq.append(static_valid_arr)
            static_flow_forward_seq.append(static_forward_arr)
            static_valid_forward_seq.append(static_forward_valid_arr)
            dynamic_residual_flow_seq.append(dynamic_residual_flow_arr)
            dynamic_residual_valid_seq.append(dynamic_residual_valid_arr)
            dynamic_residual_flow_forward_seq.append(dynamic_residual_forward_arr)
            dynamic_residual_valid_forward_seq.append(dynamic_residual_forward_valid_arr)
            dynamic_source_mask_seq.append(dynamic_source_mask_arr)

        if num_steps == 0:
            (flow_arr, valid_arr, dynamic_arr,
             static_flow_arr, static_valid_arr,
             static_forward_arr, static_forward_valid_arr,
             dynamic_residual_flow_arr, dynamic_residual_valid_arr,
             dynamic_residual_forward_arr, dynamic_residual_forward_valid_arr,
             dynamic_source_mask_arr) = self._zero_flow(occ_shape)
            flow_seq = np.zeros((0,) + flow_arr.shape, dtype=np.float32)
            valid_seq = np.zeros((0,) + valid_arr.shape, dtype=np.bool_)
            dynamic_seq = np.zeros((0,) + dynamic_arr.shape, dtype=np.bool_)
            static_flow_seq = np.zeros((0,) + static_flow_arr.shape, dtype=np.float32)
            static_valid_seq = np.zeros((0,) + static_valid_arr.shape, dtype=np.bool_)
            static_flow_forward_seq = np.zeros((0,) + static_forward_arr.shape, dtype=np.float32)
            static_valid_forward_seq = np.zeros((0,) + static_forward_valid_arr.shape, dtype=np.bool_)
            dynamic_residual_flow_seq = np.zeros((0,) + dynamic_residual_flow_arr.shape, dtype=np.float32)
            dynamic_residual_valid_seq = np.zeros((0,) + dynamic_residual_valid_arr.shape, dtype=np.bool_)
            dynamic_residual_flow_forward_seq = np.zeros((0,) + dynamic_residual_forward_arr.shape, dtype=np.float32)
            dynamic_residual_valid_forward_seq = np.zeros((0,) + dynamic_residual_forward_valid_arr.shape, dtype=np.bool_)
            dynamic_source_mask_seq = np.zeros((0,) + dynamic_source_mask_arr.shape, dtype=np.bool_)
        else:
            flow_seq = np.stack(flow_seq, axis=0)
            valid_seq = np.stack(valid_seq, axis=0)
            dynamic_seq = np.stack(dynamic_seq, axis=0)
            static_flow_seq = np.stack(static_flow_seq, axis=0)
            static_valid_seq = np.stack(static_valid_seq, axis=0)
            static_flow_forward_seq = np.stack(static_flow_forward_seq, axis=0)
            static_valid_forward_seq = np.stack(static_valid_forward_seq, axis=0)
            dynamic_residual_flow_seq = np.stack(dynamic_residual_flow_seq, axis=0)
            dynamic_residual_valid_seq = np.stack(dynamic_residual_valid_seq, axis=0)
            dynamic_residual_flow_forward_seq = np.stack(dynamic_residual_flow_forward_seq, axis=0)
            dynamic_residual_valid_forward_seq = np.stack(dynamic_residual_valid_forward_seq, axis=0)
            dynamic_source_mask_seq = np.stack(dynamic_source_mask_seq, axis=0)

        results['oracle_flow_seq'] = flow_seq
        results['oracle_flow_valid_seq'] = valid_seq
        results['oracle_dynamic_mask_seq'] = dynamic_seq
        results['oracle_static_flow_seq'] = static_flow_seq
        results['oracle_static_flow_valid_seq'] = static_valid_seq
        results['oracle_static_flow_forward_seq'] = static_flow_forward_seq
        results['oracle_static_flow_valid_forward_seq'] = static_valid_forward_seq
        results['oracle_dynamic_residual_flow_seq'] = dynamic_residual_flow_seq
        results['oracle_dynamic_residual_flow_valid_seq'] = dynamic_residual_valid_seq
        results['oracle_dynamic_residual_flow_forward_seq'] = dynamic_residual_flow_forward_seq
        results['oracle_dynamic_residual_flow_valid_forward_seq'] = dynamic_residual_valid_forward_seq
        results['oracle_dynamic_source_mask_seq'] = dynamic_source_mask_seq
        results['oracle_flow_seq_tokens'] = seq_tokens
        results['oracle_flow_seq_paths'] = flow_path_seq

        if num_steps > 0:
            results['oracle_flow'] = flow_seq[-1]
            results['oracle_flow_valid'] = valid_seq[-1]
            results['oracle_dynamic_mask'] = dynamic_seq[-1]
            results['oracle_static_flow'] = static_flow_seq[-1]
            results['oracle_static_flow_valid'] = static_valid_seq[-1]
            results['oracle_static_flow_forward'] = static_flow_forward_seq[-1]
            results['oracle_static_flow_valid_forward'] = static_valid_forward_seq[-1]
            results['oracle_dynamic_residual_flow'] = dynamic_residual_flow_seq[-1]
            results['oracle_dynamic_residual_flow_valid'] = dynamic_residual_valid_seq[-1]
            results['oracle_dynamic_residual_flow_forward'] = dynamic_residual_flow_forward_seq[-1]
            results['oracle_dynamic_residual_flow_valid_forward'] = dynamic_residual_valid_forward_seq[-1]
            results['oracle_dynamic_mask_source'] = dynamic_source_mask_seq[-1]
            results['oracle_has_flow'] = bool(valid_seq[-1].any())
            results['oracle_flow_path'] = flow_path_seq[-1]
        else:
            (flow_arr, valid_arr, dynamic_arr,
             static_flow_arr, static_valid_arr,
             static_forward_arr, static_forward_valid_arr,
             dynamic_residual_flow_arr, dynamic_residual_valid_arr,
             dynamic_residual_forward_arr, dynamic_residual_forward_valid_arr,
             dynamic_source_mask_arr) = self._zero_flow(occ_shape)
            results['oracle_flow'] = flow_arr
            results['oracle_flow_valid'] = valid_arr
            results['oracle_dynamic_mask'] = dynamic_arr
            results['oracle_static_flow'] = static_flow_arr
            results['oracle_static_flow_valid'] = static_valid_arr
            results['oracle_static_flow_forward'] = static_forward_arr
            results['oracle_static_flow_valid_forward'] = static_forward_valid_arr
            results['oracle_dynamic_residual_flow'] = dynamic_residual_flow_arr
            results['oracle_dynamic_residual_flow_valid'] = dynamic_residual_valid_arr
            results['oracle_dynamic_residual_flow_forward'] = dynamic_residual_forward_arr
            results['oracle_dynamic_residual_flow_valid_forward'] = dynamic_residual_forward_valid_arr
            results['oracle_dynamic_mask_source'] = dynamic_source_mask_arr
            results['oracle_has_flow'] = False
            results['oracle_flow_path'] = ''

        return results
