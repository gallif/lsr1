from lsr1.data import create_dataset
from lsr1.models.lsr1_baseline import LOPTModel

import hydra
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

torch.autograd.set_detect_anomaly(True)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, trainset, valset):
        super().__init__()
        self.trainset = trainset
        self.valset = valset

    def train_dataloader(self):
        return self.trainset

    def val_dataloader(self):
        return self.valset


@hydra.main(config_path="configs", config_name="train")
def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_devices)
    pl.seed_everything(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Working directory : {}".format(os.getcwd()))
    trainset = create_dataset(opt.data.train)
    valset = create_dataset(opt.data.validate)

    datamodule = CustomDataModule(trainset, valset)
    model = LOPTModel(opt=opt.model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="ckpts/",
                                                       filename="best-model",
                                                       monitor="val/MPJPE_aligned",
                                                       save_top_k=1,
                                                       save_last=True,
                                                       every_n_epochs=1)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}/{opt.timestamp}")
    trainer = pl.Trainer(gpus=1,
                         accelerator="gpu",
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         log_every_n_steps=1,
                         flush_logs_every_n_steps=1,
                         val_check_interval=100,
                         limit_train_batches=1000
                         )

    model.train()
    trainer.fit(model, 
                datamodule=datamodule, 
                ckpt_path=hydra.utils.to_absolute_path(opt.resume_chkpt) if opt.resume_chkpt is not None else None)

if __name__ == '__main__':
    main()  
