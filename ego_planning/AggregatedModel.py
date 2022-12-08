import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from l5kit.geometry import transform_points
from l5kit.dataset import AgentDataset


class AggregatedModel(nn.Module):
    """Raster-based model for planning with prediction data.
    """

    def __init__(
            self,
            model_arch: str,
            num_input_channels: int,
            num_targets: int,
            weights_scaling: List[float],
            criterion: nn.Module,
            pretrained: bool = True,
            agent_prediction_model = None
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.agent_prediction_model = agent_prediction_model

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        
        # [batch_size, num_steps * 2]
        outputs = self.model(image_batch)
        batch_size = len(data_batch["image"])

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_steps * 2]
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            predicted = outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict

    def forward_with_prediction(cfg, rasterizer, ego_data, agent_dataset : AgentDataset, ego_planning_model, agent_prediction_model, device, criterion):
        frame_number = ego_data['frame_index']
        agent_indices = agent_dataset.get_frame_indices(frame_number) 
        if not len(agent_indices):
            # skip this maybe idk
            pass

        # get AV point-of-view frame
        data_ego = ego_data
        im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
        center = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        
        predicted_positions = []
        target_positions = []

        for v_index in agent_indices:
            data_agent = agent_dataset[v_index]

            out_net = agent_prediction_model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))
            out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()
            # store absolute world coordinates
            predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))


        # convert coordinates to AV point-of-view so we can draw them
        predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])

        # draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
        # draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)