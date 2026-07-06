from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Eval")
        bar.leave = False
        return bar

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.set_description("Sanity Check")
        bar.leave = False
        return bar