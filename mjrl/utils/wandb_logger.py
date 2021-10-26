import wandb

from mjrl.utils.logger import DataLog


class WandbLogger(DataLog):
    """Extends logger to also log to wandb."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wandb.init(
            resume='',
            project='semantic_imitation',
            config={},
            dir='./logs',
            entity='clvr',
            notes='',
        )

    def log_kv(self, key, value):
        super().log_kv(key, value)
        wandb.log({key: value})