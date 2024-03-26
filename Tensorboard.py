import pytorch_lightning
import torch


class Tensorboard:
    def __init__(self, log_dir):
        self.logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir, name="lightning_logs")
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def log_scalar(self, tag, value, step):
        print("TAG: ", tag)
        self.logger.log_metrics({tag: value}, step=step)

    def log_image(self, tag, img_tensor, step):
        self.logger.experiment.add_image(tag, img_tensor, step)

    def log_model_graph(self, model, input_tensor):
        self.logger.log_graph(model, input_tensor)


    def close(self):
        self.logger.save()
