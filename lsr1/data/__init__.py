from .PW_dataset import PWDataset
from .AMASS_dataset import AMASSDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np


def find_dataset_using_name(name):
    mapping = {
        "3DPW": PWDataset,
        "AMASS": AMASSDataset, 
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}") 
    return cls


def create_dataset(opt):
    dataset_cls = find_dataset_using_name(opt.name)
    dataset = dataset_cls(opt)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=opt.drop_last,
        shuffle=opt.shuffle,
        num_workers=opt.worker,
        pin_memory=True
    )


def draw_joints_indices(joints_2d, output_path, image_size=1000, img=None):
    """
    Draws 2D joints with index labels on a white image and saves it.

    Args:
        joints_2d (np.ndarray): Array of shape (N, 2) with 2D joint coordinates.
        output_path (str): Path to save the output image.
        image_size (int): Width and height of the output image.
    """
    if img is None:
        # Create a white image
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Font settings for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (0, 0, 0)  # Black

    # Draw red circles and index labels
    for idx, (u, v) in enumerate(joints_2d):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < image_size and 0 <= v < image_size:
            cv2.circle(img, (u, v), radius=4, color=(0, 0, 255), thickness=-1)  # Red dot
            cv2.putText(img, str(idx), (u + 10, v + 10), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    # Save the image
    cv2.imwrite(output_path, img)
    print(f"Saved image with joints to {output_path}")