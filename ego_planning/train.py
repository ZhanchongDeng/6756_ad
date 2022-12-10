import os
from pathlib import Path
import argparse
from typing import Dict
from torchvision.models.resnet import resnet50
from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.geometry import transform_points
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from AggregatedDataset import AggregatedDataset
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

DATASET_DIR_PATH = "../prediction-dataset/"
SAVED_MODEL_PATH = "./models/"

def build_agent_prediction_model(config_path) -> torch.nn.Module:
    cfg = load_config_data(Path(config_path))
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

def forward_agent_prediction(data, model, device, criterion):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-prediction", action="store_true", default=False)
    parser.add_argument("--prediction-model-path", type=str, default="../agent_prediction/models/bl_it500.pt")
    parser.add_argument("--prediction-config-path", type=str, default="../agent_prediction/code/baseline_config.yaml")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--config-path", type=str, default="configs/aggregate_config.yaml")

    args = parser.parse_args()

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = DATASET_DIR_PATH
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(args.config_path)

    if args.use_gpu:
        # use cuda on linux & windows
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("mps") # can't use worker > 1 for some reason. Still faster than cpu and 16 workers.
    else:
        device = torch.device("cpu")

    # ===== INIT DATASET
    if args.use_prediction:
        agent_prediction_model = build_agent_prediction_model(args.prediction_config_path)
        agent_prediction_model.load_state_dict(torch.load(args.prediction_model_path, map_location=torch.device("cpu")))
        agent_prediction_model.to(device)
        agent_prediction_model.eval()
    
    perturb_prob = cfg["train_data_loader"]["perturb_probability"]

    # rasterisation and perturbation
    rasterizer = build_rasterizer(cfg, dm)
    mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
    std = np.array([0.5, 1.5, np.pi / 6])
    perturbation = AckermanPerturbation(
            random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

    # ===== INIT DATASET
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    if args.use_prediction:
        ego_train_dataset = AggregatedDataset(cfg, agent_dataset.dataset, rasterizer, None, agent_prediction_model, agent_dataset, device)
    else:
        ego_train_dataset = EgoDataset(cfg, train_zarr, rasterizer, perturbation)

    print("ego dataset:\n", ego_train_dataset)
    print("agent dataset:\n", agent_dataset)

    # ===== INIT MODEL
    # TODO: not sure if this is correct
    num_input_channels = rasterizer.num_channels() + 1 if args.use_prediction else rasterizer.num_channels()
    print(num_input_channels)
    ego_planning_model = RasterizedPlanningModel(
            model_arch="resnet50",
            num_input_channels=num_input_channels,
            num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
            weights_scaling= [1., 1., 1.],
            criterion=nn.MSELoss(reduction="none")
            )
    
    # ===== INIT DATALOADER
    train_cfg = cfg["train_data_loader"]
    ego_train_dataloader = DataLoader(ego_train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], num_workers= train_cfg["num_workers"])
    
    ego_planning_model = ego_planning_model.to(device)
    
    optimizer = optim.Adam(ego_planning_model.parameters(), lr=1e-3)

    # ===== TRAIN LOOP
    tr_it = iter(ego_train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []
    ego_planning_model.train()
    torch.set_grad_enabled(True)
    
    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(ego_train_dataloader)
            data = next(tr_it)
        
        # Forward pass
        data = {k: v.float().to(device) for k, v in data.items()}
        result = ego_planning_model(data)
        loss = result["loss"]
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
    
    # create save dir
    Path(SAVED_MODEL_PATH).mkdir(parents=True, exist_ok=True)

    # ===== GRAPH TRAINING LOSSES
    plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
    plt.legend()
    # save figure
    plt.savefig(Path(SAVED_MODEL_PATH) / f"losses_{cfg['train_params']['name']}.png")

    # ===== SAVE MODEL
    to_save = torch.jit.script(ego_planning_model.cpu())
    path_to_save = str(Path(SAVED_MODEL_PATH, f"planning_model_{cfg['train_params']['name']}.pt"))
    to_save.save(path_to_save)
    print(f"MODEL STORED at {path_to_save}")