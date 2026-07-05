from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    """
    Stable progress bar for Kaggle/Jupyter environments.
    - Disables Rich so only one bar system runs.
    - Closes bars cleanly after each phase to prevent ghost lines.
    - Prints a compact one-line summary per epoch.
    """

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = True
        bar.leave = False          # do NOT leave bar on screen after epoch ends
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('Eval')
        bar.dynamic_ncols = True
        bar.leave = False
        return bar

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.set_description('Sanity Check')
        bar.dynamic_ncols = True
        bar.leave = False
        return bar

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        # Close and clear the training bar explicitly
        if self.train_progress_bar is not None:
            self.train_progress_bar.clear()
            self.train_progress_bar.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        # Close and clear the validation bar explicitly
        if self.val_progress_bar is not None:
            self.val_progress_bar.clear()
            self.val_progress_bar.close()