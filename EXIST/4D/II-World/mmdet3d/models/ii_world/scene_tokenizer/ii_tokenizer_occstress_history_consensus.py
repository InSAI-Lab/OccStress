# Copyright (c) OpenMMLab. All rights reserved.

from mmdet.models import DETECTORS

from .ii_tokenizer_occstress import OccStressIISceneTokenizer


@DETECTORS.register_module()
class OccStressIISceneTokenizerHistoryConsensus(OccStressIISceneTokenizer):
    """Dedicated OccStress tokenizer entry point for history-consensus fusion."""

    pass
