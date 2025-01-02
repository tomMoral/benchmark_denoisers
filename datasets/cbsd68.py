from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    from deepinv.utils.demo import load_dataset
    import torch
    import torchvision.transforms as transforms


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    name = "CBSD68"

    parameters = {
        'n_images': [None],
    }

    def get_data(self):

        img_size = 128 if torch.cuda.is_available() else 32
        transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        dataset = load_dataset(
            "CBSD68", data_dir=get_data_path(), transform=transform,
            train=False
        )
        if self.n_images is None:
            self.n_images = len(dataset)
        images = torch.concat(
            tuple(dataset[i][0][None] for i in range(self.n_images)),
        )

        return dict(images=images)
