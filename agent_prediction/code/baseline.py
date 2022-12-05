from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os
import sys
from datetime import datetime


def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model


def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs


if __name__ == '__main__':

    data_root = "/data/hc2225/prediction-dataset/"
    os.environ["L5KIT_DATA_FOLDER"] = data_root
    dm = LocalDataManager(None)
    # get config
    model_root = "/home/hc2225/planning/"
    cfg = load_config_data(os.path.join(model_root, "code", "baseline_config.yaml"))

    # ===== INIT DATASET
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])
    print(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    distributed = True
    if distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)

    print(train_dataset)

    # ==== TRAIN LOOP
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []
    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward(data, model, device, criterion)

        # multiple gpu training
        loss = loss.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
    plt.legend()
    fig_name = cfg['model_params']['model_architecture'] + '_' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.jpg'
    plt.savefig(fig_name)

    path_to_save = os.path.join(model_root, "models", "planning_model.pt")
    torch.save(model.module.state_dict(), path_to_save)
    print(f"MODEL STORED at {path_to_save}")

    # ==== EVALUATION
    # num_frames_to_chop = 100
    # eval_cfg = cfg["val_data_loader"]
    # eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
    #                             num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)

    # eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
    # eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    # eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    # eval_zarr = ChunkedDataset(eval_zarr_path).open()
    # eval_mask = np.load(eval_mask_path)["arr_0"]
    # # ===== INIT DATASET AND LOAD MASK
    # eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
    # eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
    #                             num_workers=eval_cfg["num_workers"])
    # print(eval_dataset)