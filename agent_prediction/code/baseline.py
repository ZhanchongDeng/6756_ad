from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    # model = resnet50(pretrained=True)
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # EfficientNetB3
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    num_targets = 2 * cfg["model_params"]["future_num_frames"]

    if model.__class__.__name__ == "ResNet":
        model.conv1 = nn.Conv2d(
            num_in_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
        # change output size to (X, Y) * number of future states
        model.fc = nn.Linear(in_features=2048, out_features=num_targets)
    elif model.__class__.__name__ == "EfficientNet":
        first_layer = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            num_in_channels,
            first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding,
            bias=False,
        )
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_targets, bias=True)
    
    print('model loaded')

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

    data_root = "/data/hc2225/prediction-dataset"
    model_root = "/home/hc2225/av/agent_prediction"
    os.environ["L5KIT_DATA_FOLDER"] = data_root
    dm = LocalDataManager(None)
    # get config
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

    tb_writer = SummaryWriter(log_dir=os.path.join(model_root, "models"))

    distributed = True
    if distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)

    print(train_dataset)

    # ==== TRAIN LOOP

    losses_train = []
    tr_it = iter(train_dataloader)

    train_mode = 'step'
    if train_mode == 'step':
        progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
        for i in progress_bar:
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
            tb_writer.add_scalar("Loss", loss, i)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
    else:
        epoches = cfg["train_params"]["max_epoch"]
        progress_bar = tqdm(total=epoches * len(tr_it))
        for epoch in range(epoches):
            for batch_idx, data in enumerate(train_dataloader):
                model.train()
                torch.set_grad_enabled(True)
                loss, _ = forward(data, model, device, criterion)

                # multiple gpu training
                loss = loss.sum()

                tb_writer.add_scalar("Loss", loss, epoch * len(train_dataloader) + batch_idx)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_train.append(loss.item())
                progress_bar.update(1)
                # progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    tb_writer.close()

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