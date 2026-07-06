from tqdm.auto import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

    def init_validation_tqdm(self):
        return tqdm(
            desc="Eval",
            position=self.process_position,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
        )

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.set_description("Sanity Check")
        bar.leave = False
        return bar