from tqdm import tqdm
import sys
import pytorch_lightning as pl


class LitProgressBar(pl.callbacks.progress.ProgressBar):
    def init_validation_tqdm(self):
        """ Override this to customize the tqdm bar for val. """

        bar = tqdm(
            desc='Validating',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=True,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar
