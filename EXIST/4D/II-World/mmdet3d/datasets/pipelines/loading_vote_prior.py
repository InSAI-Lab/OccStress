import os

import numpy as np

from ..builder import PIPELINES
from .loading import LoadStreamLatentToken
from .loading_occstress import OccStressLoadStreamLatentToken


@PIPELINES.register_module()
class LoadStreamLatentVoteToken(LoadStreamLatentToken):
    """Load clean stage2 latent tokens plus a current-frame vote prior token."""

    def __init__(self,
                 data_path=None,
                 vote_data_path=None,
                 to_long=False,
                 dataset_type='occ3d',
                 load_vote_confidence=True,
                 ):
        super().__init__(data_path=data_path, to_long=to_long, dataset_type=dataset_type)
        self.vote_data_path = vote_data_path
        self.load_vote_confidence = load_vote_confidence

    def _token_path(self, root, scene_name, sample_idx, occ_path=None):
        if self.dataset_type == 'waymo':
            scene_name = str(scene_name).zfill(3)
            occ_index = occ_path.split('/')[-1].split('.')[0] if occ_path is not None else sample_idx
            return os.path.join(root, scene_name, f'{occ_index}.npz')
        return os.path.join(root, str(scene_name), f'{sample_idx}.npz')

    def __call__(self, results):
        results = super().__call__(results)
        if self.vote_data_path is None:
            return results

        vote_token_path = self._token_path(
            self.vote_data_path,
            results['scene_name'],
            results['sample_idx'],
            occ_path=results.get('occ_path'),
        )
        vote_payload = np.load(vote_token_path)
        results['vote_latent'] = vote_payload['token']
        if self.load_vote_confidence:
            if 'vote_confidence' in vote_payload:
                results['vote_confidence'] = vote_payload['vote_confidence']
            elif 'support_bev' in vote_payload:
                results['vote_confidence'] = vote_payload['support_bev']
        return results


@PIPELINES.register_module()
class OccStressLoadStreamLatentVoteToken(OccStressLoadStreamLatentToken):
    """Load OccStress current/future tokens plus a current-frame vote prior."""

    def __init__(self,
                 current_data_path,
                 future_data_path=None,
                 vote_data_path=None,
                 dataset_type='occ3d',
                 load_vote_confidence=True,
                 ):
        super().__init__(
            current_data_path=current_data_path,
            future_data_path=future_data_path,
            dataset_type=dataset_type,
        )
        self.vote_data_path = vote_data_path
        self.vote_data_paths = self._as_list(vote_data_path) if vote_data_path is not None else []
        self.load_vote_confidence = load_vote_confidence

    def __call__(self, results):
        results = super().__call__(results)
        if not self.vote_data_paths:
            return results

        scene_name = results['scene_name']
        sample_names = []
        protocol_sample_id = results.get('protocol_sample_id')
        if protocol_sample_id:
            sample_names.append(protocol_sample_id)
        sample_names.append(results['sample_idx'])

        vote_payload = None
        for sample_name in sample_names:
            _, vote_payload = self._load_first_existing(self.vote_data_paths, scene_name, sample_name)
            if vote_payload is not None:
                break
        if vote_payload is None:
            raise FileNotFoundError(
                f'Could not resolve OccStress vote latent token for scene={scene_name}. '
                f'Tried names={sample_names} under roots={self.vote_data_paths}'
            )

        results['vote_latent'] = vote_payload['token']
        if self.load_vote_confidence:
            if 'vote_confidence' in vote_payload:
                results['vote_confidence'] = vote_payload['vote_confidence']
            elif 'support_bev' in vote_payload:
                results['vote_confidence'] = vote_payload['support_bev']
        return results
