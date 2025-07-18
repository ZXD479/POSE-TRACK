import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        top, left, bottom, right = self.last_tlbr
        on_left   = abs(left) <= 100
        on_right  = abs(right - 1920) <= 100
        on_top    = abs(top) <= 100
        on_bottom = abs(bottom - 1080) <= 100
        if on_bottom or on_left or on_right or on_top:
            self.state = 3
        else:
            self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
    @staticmethod
    def clear_count():
        BaseTrack._count = 0
