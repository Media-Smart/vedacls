import torch


def build_dataloader(dataset, cfg_loader, shuffle=True):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg_loader.batch_size,
        shuffle=shuffle,
        num_workers=cfg_loader.workers_per_gpu,
        pin_memory=True)

    return loader
