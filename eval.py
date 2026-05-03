import os

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from lsr1.data import create_dataset
from lsr1.models.lsr1_baseline import LOPTModel
from pytorch_lightning.loggers import WandbLogger

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, testset):
        super().__init__()
        self.testset = testset

    def test_dataloader(self):
        return self.testset

def load_hydra_config_from_checkpoint(ckpt_path):
    """Loads the saved Hydra config from the checkpoint directory."""
    config_path = os.path.join(os.path.dirname(ckpt_path), ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)

@hydra.main(config_path="configs", config_name="eval")
def main(opt):
    pl.seed_everything(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_devices)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint = hydra.utils.to_absolute_path(opt.load_checkpoint) if opt.load_checkpoint is not None else None

    testset = create_dataset(opt.data.test)
    datamodule = CustomDataModule(testset)
        
    model = LOPTModel.load_from_checkpoint(checkpoint, strict=False, opt=opt.model)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}/{opt.timestamp}",tags="eval")

    model.eval()
    trainer = pl.Trainer(gpus=1, 
                         logger=logger,
                         accelerator="gpu")
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
