import wandb
import numpy as np
from collections import defaultdict

from mjrl.utils.logger import DataLog


class WandbLogger(DataLog):
    """Extends logger to also log to wandb."""

    def __init__(self, *args, job_name='', **kwargs):
        super().__init__(*args, **kwargs)
        wandb.init(
            resume=job_name,
            project='semantic_imitation',
            config={},
            dir='./logs',
            entity='clvr',
            notes='',
        )
        self._log_steps = defaultdict(lambda: 1)

    def log_kv(self, key, value):
        super().log_kv(key, value)
        wandb.log({key: value}, step=self._log_steps[key])
        self._log_steps[key] += 1

    def log_videos(self, vids, name, fps=20, max_vids=3):
        """Logs video to wandb. Vids is a list of arrays of shape [T, C, H, W]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        if vids[0].max() <= 1.0: vids = [np.asarray(vid * 255.0, dtype=np.uint8) for vid in vids]
        log_dict = {name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids[:max_vids]]}
        wandb.log(log_dict, step=self._log_steps[name])
        self._log_steps[name] += 1