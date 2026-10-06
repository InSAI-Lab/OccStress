"""Temporal slot selection; the current state is slot zero."""
PATTERNS = {'current': [0], 'history_k1': [-1, 0], 'all_frame': [-4, -3, -2, -1]}
HORIZONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def history_mask_for_protocol(history_length, frame_protocol, k):
    if history_length < 0 or k < 0:
        raise ValueError('History length and k must be nonnegative')
    if frame_protocol == 'all_frame':
        return [1] * history_length
    if frame_protocol == 'history_k1':
        return [int(i >= history_length - k) for i in range(history_length)]
    if frame_protocol == 'current':
        return [0] * history_length
    raise ValueError(frame_protocol)


def target_active_for_protocol(frame_protocol):
    if frame_protocol in {'current', 'history_k1'}:
        return True
    if frame_protocol == 'all_frame':
        return False
    raise ValueError(frame_protocol)


def visible_sweep_slots(input_offsets):
    return [int(offset * 2) for offset in input_offsets]
