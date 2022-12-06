import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.geometry import transform_points
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

DATASET_DIR_PATH = "../prediction-dataset/"
CONFIG_PATH = "./config.yaml"
SAVED_MODEL_PATH = "./models/"

def main():
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = DATASET_DIR_PATH
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(CONFIG_PATH)

    perturb_prob = cfg["train_data_loader"]["perturb_probability"]

    # rasterisation and perturbation
    rasterizer = build_rasterizer(cfg, dm)
    mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
    std = np.array([0.5, 1.5, np.pi / 6])
    perturbation = AckermanPerturbation(
            random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

    # ===== INIT DATASET
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_dataset = EgoDataset(cfg, train_zarr, rasterizer, perturbation)

    # ===== INIT MODEL
    model = RasterizedPlanningModel(
        model_arch="resnet50",
        num_input_channels=rasterizer.num_channels(),
        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
        weights_scaling= [1., 1., 1.],
        criterion=nn.MSELoss(reduction="none")
        )
    print(model)

    # ===== INIT DATALOADER
    train_cfg = cfg["train_data_loader"]
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(train_dataset)

    # ===== TRAIN LOOP
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []
    model.train()
    torch.set_grad_enabled(True)

    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        # Forward pass
        data = {k: v.float().to(device) for k, v in data.items()}
        result = model(data)
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
    to_save = torch.jit.script(model.cpu())
    path_to_save = str(Path(SAVED_MODEL_PATH, f"planning_model_{cfg['train_params']['name']}.pt"))
    to_save.save(path_to_save)
    print(f"MODEL STORED at {path_to_save}")

if __name__ == '__main__':
    main()