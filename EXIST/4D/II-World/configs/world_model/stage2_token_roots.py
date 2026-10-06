"""Canonical nuScenes stage2 token roots.

Keep baseline stage2 token paths in one place. The original stage2 training used
``save_dir`` train tokens, but its val tokens are not the locked eval set. For
val/test/eval, use the separately exported evalreg token root.
"""

BASELINE_STAGE2_TRAIN_TOKEN_ROOT = 'data/nuscenes/save_dir/token_4f'
BASELINE_STAGE2_EVAL_TOKEN_ROOT = 'data/nuscenes/save_dir_baseline_original_evalreg/token_4f'

BASELINE_STAGE2_TOKEN_ROOTS = dict(
    train=BASELINE_STAGE2_TRAIN_TOKEN_ROOT,
    val=BASELINE_STAGE2_EVAL_TOKEN_ROOT,
    test=BASELINE_STAGE2_EVAL_TOKEN_ROOT,
)

VOTE_PRIOR_BASELINE_TOKEN_ROOT = 'data/nuscenes/save_dir_vote_prior_baseline/token_4f'


def baseline_stage2_token_root(split):
    """Return the canonical baseline stage2 token root for a dataset split."""
    if split not in BASELINE_STAGE2_TOKEN_ROOTS:
        raise KeyError(f'Unsupported baseline stage2 split: {split}')
    return BASELINE_STAGE2_TOKEN_ROOTS[split]


def baseline_stage2_lookup_token_roots():
    """Return roots for sample lookup, preferring locked eval tokens over train."""
    return [
        BASELINE_STAGE2_EVAL_TOKEN_ROOT,
        BASELINE_STAGE2_TRAIN_TOKEN_ROOT,
    ]
