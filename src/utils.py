from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    """
    Stable progress bar for Kaggle/Jupyter environments.
    - Closes bars cleanly after each phase to prevent ghost lines.
    NOTE: do NOT set bar.dynamic_ncols after construction — tqdm converts
    dynamic_ncols=True into a callable internally; overwriting it with a bool
    causes TypeError: 'bool' object is not callable.
    """

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False          # do NOT leave bar on screen after epoch ends
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('Eval')
        bar.leave = False
        return bar

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.set_description('Sanity Check')
        bar.leave = False
        return bar

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.train_progress_bar is not None:
            self.train_progress_bar.clear()
            self.train_progress_bar.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self.val_progress_bar is not None:
            self.val_progress_bar.clear()
            self.val_progress_bar.close()