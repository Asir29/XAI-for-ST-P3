import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stp3.layers.convolutions import Bottleneck
from stp3.layers.temporal import SpatialGRU, Dual_GRU, BiGRU
from stp3.cost import Cost_Function

class Planning(nn.Module):
    def __init__(self, cfg, feature_channel, gru_input_size=6, gru_state_size=256):
        super(Planning, self).__init__()
        self.cost_function = Cost_Function(cfg)

        self.sample_num = cfg.PLANNING.SAMPLE_NUM
        self.commands = cfg.PLANNING.COMMAND
        assert self.sample_num % 3 == 0
        self.num = int(self.sample_num / 3)

        self.reduce_channel = nn.Sequential(
            Bottleneck(feature_channel, feature_channel, downsample=True),
            Bottleneck(feature_channel, int(feature_channel/2), downsample=True),
            Bottleneck(int(feature_channel/2), int(feature_channel/2), downsample=True),
            Bottleneck(int(feature_channel/2), int(feature_channel/8))
        )

        self.GRU = nn.GRUCell(gru_input_size, gru_state_size)
        self.decoder = nn.Sequential(
            nn.Linear(gru_state_size, gru_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(gru_state_size, 2)
        )


    def compute_L2(self, trajs, gt_traj):
        '''
        trajs: torch.Tensor (B, N, n_future, 3)
        gt_traj: torch.Tensor (B,1, n_future, 3)
        '''
        if trajs.ndim == 4 and gt_traj.ndim == 4:
            return ((trajs[:,:,:,:2] - gt_traj[:,:,:,:2]) ** 2).sum(dim=-1)
        if trajs.ndim == 3 and gt_traj.ndim == 3:
            return ((trajs[:, :, :2] - gt_traj[:, :, :2]) ** 2).sum(dim=-1)

        raise ValueError('trajs ndim != gt_traj ndim')

    def select(self, trajs, cost_volume, semantic_pred, lane_divider, drivable_area, target_points, k=1):
        '''
        trajs: torch.Tensor (B, N, n_future, 3)
        cost_volume: torch.Tensor (B, n_future, 200, 200)
        semantic_pred: torch.Tensor(B, n_future, 200, 200)
        lane_divider: torch.Tensor(B, 1/2, 200, 200)
        drivable_area: torch.Tensor(B, 1/2, 200, 200)
        target_points: torch.Tensor<float> (B, 2)
        '''
        sm_cost_fc, sm_cost_fo, all_costs, safetycost = self.cost_function(
            cost_volume, trajs[:, :, :, :2], semantic_pred, lane_divider, drivable_area, target_points
        )

        # Debug combined costs
        print(f"[select] sm_cost_fc shape: {sm_cost_fc.shape}, min/max/mean: {sm_cost_fc.min().item():.4f}/{sm_cost_fc.max().item():.4f}/{sm_cost_fc.mean().item():.4f}")
        print(f"[select] sm_cost_fo shape: {sm_cost_fo.shape}, min/max/mean: {sm_cost_fo.min().item():.4f}/{sm_cost_fo.max().item():.4f}/{sm_cost_fo.mean().item():.4f}")

        # Compute total cost per trajectory
        CS = sm_cost_fc + sm_cost_fo.sum(dim=-1)  # shape: (B, N)

        print(f"[select] Total cost CS shape: {CS.shape}, min/max/mean: {CS.min().item():.4f}/{CS.max().item():.4f}/{CS.mean().item():.4f}")

        CC, KK = torch.topk(CS, k, dim=-1, largest=False)  # KK shape: (B, k)

        batch_size = trajs.shape[0]
        batch_indices = torch.arange(batch_size, device=trajs.device)  # shape: (B,)
        selected_indices = KK.squeeze(-1)  # shape: (B,)

        print(f"[select] Selected trajectory indices: {selected_indices}")

        # Select trajectories
        select_traj = trajs[batch_indices[:, None], KK].squeeze(1)  # shape: (B, n_future, 3)

        # Extract costs for selected trajectories only
        selected_costs = {}
        for concept, cost_tensor in all_costs.items():
            #print(f"[select] Processing cost concept '{concept}' with shape {cost_tensor.shape}")
            if cost_tensor.dim() == 3:
                # Expected shape: (B, N, n_future)
                selected_costs[concept] = cost_tensor[batch_indices, selected_indices, :]  # shape: (B, n_future)
                #print(f"[select] Selected cost shape for '{concept}': {selected_costs[concept].shape}")
            elif cost_tensor.dim() == 2:
                # Expected shape: (B, N)
                selected_costs[concept] = cost_tensor[batch_indices, selected_indices]  # shape: (B,)
                #print(f"[select] Selected cost shape for '{concept}': {selected_costs[concept].shape}")
            else:
                raise ValueError(f"Unexpected cost tensor dimension {cost_tensor.shape}")

        # Aggregate costs over time dimension when applicable
        aggregated_costs = {}

        # Alternative safety cost aggregation for debugging
        safetycost_timeseries = safetycost[batch_indices, selected_indices, :]
        safetycost_scalar = safetycost_timeseries.max(dim=1).values
        aggregated_costs["safetycost_alternative"] = safetycost_scalar

        for concept, cost_val in selected_costs.items():
            if cost_val.dim() == 2:  # (B, n_future)
                aggregated_costs[concept] = cost_val.max(dim=1).values  # Extract values explicitly
            else:
                aggregated_costs[concept] = cost_val  # Already scalar

        # Debug print aggregated costs
        #for concept, val in aggregated_costs.items():
            #print(f"[select] Aggregated cost '{concept}': {val}")

        return select_traj, aggregated_costs





    def loss(self, trajs, gt_trajs, cost_volume, semantic_pred, lane_divider, drivable_area, target_points):
        '''
        trajs: torch.Tensor (B, N, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        cost_volume: torch.Tensor (B, n_future, 200, 200)
        semantic_pred: torch.Tensor(B, n_future, 200, 200)
        lane_divider: torch.Tensor(B, 1/2, 200, 200)
        drivable_area: torch.Tensor(B, 1/2, 200, 200)
        target_points: torch.Tensor<float> (B, 2)
        '''
        sm_cost_fc, sm_cost_fo = self.cost_function(cost_volume, trajs[:, :, :, :2], semantic_pred, lane_divider, drivable_area, target_points)

        if gt_trajs.ndim == 3:
            gt_trajs = gt_trajs[:, None]

        gt_cost_fc, gt_cost_fo = self.cost_function(cost_volume, gt_trajs[:, :, :, :2], semantic_pred, lane_divider, drivable_area, target_points)

        L, _ = F.relu(
            F.relu(gt_cost_fo - sm_cost_fo).sum(-1) + (gt_cost_fc - sm_cost_fc) + self.compute_L2(trajs, gt_trajs).mean(
                dim=-1)).max(dim=-1)

        return torch.mean(L)

    def forward(self,cam_front, trajs, gt_trajs, cost_volume, semantic_pred, hd_map, commands, target_points):
        '''
        cam_front: torch.Tensor (B, 64, 60, 28)
        trajs: torch.Tensor (B, N, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        cost_volume: torch.Tensor (B, n_future, 200, 200)
        semantic_pred: torch.Tensor(B, n_future, 200, 200)
        hd_map: torch.Tensor (B, 2/4, 200, 200)
        commands: List (B)
        target_points: (B, 2)
        '''

        cur_trajs = []
        for i in range(len(commands)):
            command = commands[i]
            traj = trajs[i]
            if command == 'LEFT':
                cur_trajs.append(traj[:self.num].repeat(3, 1, 1))
            elif command == 'FORWARD':
                cur_trajs.append(traj[self.num:self.num * 2].repeat(3, 1, 1))
            elif command == 'RIGHT':
                cur_trajs.append(traj[self.num * 2:].repeat(3, 1, 1))
            else:
                cur_trajs.append(traj)
        cur_trajs = torch.stack(cur_trajs)

        if hd_map.shape[1] == 2:
            lane_divider = hd_map[:, 0:1]
            drivable_area = hd_map[:, 1:2]
        elif hd_map.shape[1] == 4:
            lane_divider = hd_map[:, 0:2]
            drivable_area = hd_map[:, 2:4]
        else:
            raise NotImplementedError

        if self.training:
            loss = self.loss(cur_trajs, gt_trajs, cost_volume, semantic_pred, lane_divider, drivable_area, target_points)
        else:
            loss = 0

        cam_front = self.reduce_channel(cam_front)
        h0 = cam_front.flatten(start_dim=1) # (B, 256/128)
        final_traj, aggregated_costs = self.select(cur_trajs, cost_volume, semantic_pred, lane_divider, drivable_area, target_points) # (B, n_future, 3)
        target_points = target_points.to(dtype=h0.dtype)
        b, s, _ = final_traj.shape
        x = torch.zeros((b, 2), device=h0.device)
        output_traj = []
        for i in range(s):
            x = torch.cat([x, final_traj[:,i,:2], target_points], dim=-1) # (B, 6)
            h0 = self.GRU(x, h0)
            x = self.decoder(h0) # (B, 2)
            output_traj.append(x)
        output_traj = torch.stack(output_traj, dim=1) # (B, 4, 2)

        output_traj = torch.cat(
            [output_traj, torch.zeros((*output_traj.shape[:-1],1), device=output_traj.device)], dim=-1
        )

        if self.training:
            loss = loss*0.5 + (F.smooth_l1_loss(output_traj[:,:,:2], gt_trajs[:,:,:2], reduction='none')*torch.tensor([10., 1.], device=loss.device)).mean()

        return loss, output_traj, aggregated_costs
