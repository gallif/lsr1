import hydra
import numpy as np
from torch.utils.data import Dataset
from ..utils.transforms import RandomRotateSMPL

class AMASSDataset(Dataset):
    def __init__(self, opt):
        self.dataset = np.load(hydra.utils.to_absolute_path(opt.dataset_path))

        # force loading npz data to memory
        self.dataset = {k: self.dataset[k] for k in self.dataset}
        
        if opt.frac < 1:
            num_samples_all = self.dataset["smpl_shape"].shape[0]

            # Shuffle indices
            shuffled_indices = np.random.permutation(num_samples_all)
            self.dataset["smpl_shape"][shuffled_indices]
            self.dataset["smpl_pose"][shuffled_indices]

            # Compute the number of rows to select
            num_selected = int(num_samples_all * opt.frac)
            
            # Select the first `num_selected` rows
            self.dataset["smpl_shape"] = self.dataset["smpl_shape"][:num_selected]
            self.dataset["smpl_pose"] = self.dataset["smpl_pose"][:num_selected]

        if opt.augmentation:
            self.transform = RandomRotateSMPL()
        else:
            self.transform = None

    def __getitem__(self, index):
        betas = self.dataset["smpl_shape"][index]
        thetas = self.dataset["smpl_pose"][index]
        output = {'thetas': thetas, 'betas': betas}
        if self.transform is not None:
            output = self.transform(output)
        return output

    def __len__(self):
        return self.dataset["smpl_shape"].shape[0]
