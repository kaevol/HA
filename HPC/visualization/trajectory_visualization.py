import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

num_historical_steps = 20


def trajectory_visualization(data: Batch, traj_output: torch.Tensor, is_test: bool = False) -> None:
    batch_size = len(data['scenario_name'])

    agent_batch = data['agent']['batch']
    agent_position = data['agent']['position'].detach()
    agent_position = unbatch(agent_position, agent_batch)
    num_modes = traj_output.size(2)
    traj_output = traj_output.detach()
    traj_output = unbatch(traj_output[:, -1], agent_batch)
    agent_index = data['agent']['agent_index']

    # Get lane data for visualization
    lane_batch = data['lane']['batch']
    lane_position = data['lane']['position'].detach()
    
    for i in range(batch_size):
        plt.figure(figsize=(10, 8))
        agent_position_i = agent_position[i][agent_index[i]].squeeze(0)
        agent_historical_position = agent_position_i[:num_historical_steps].cpu().numpy()
        agent_future_position = agent_position_i[num_historical_steps:].cpu().numpy()
        agent_prediction_position = traj_output[i][agent_index[i]].squeeze(0).cpu().numpy()

        if not is_test:
            x_min = min(np.min(agent_historical_position[:, 0]), np.min(agent_future_position[:, 0]), np.min(agent_prediction_position[:, :, 0]))
            x_max = max(np.max(agent_historical_position[:, 0]), np.max(agent_future_position[:, 0]), np.max(agent_prediction_position[:, :, 0]))
            y_min = min(np.min(agent_historical_position[:, 1]), np.min(agent_future_position[:, 1]), np.min(agent_prediction_position[:, :, 1]))
            y_max = max(np.max(agent_historical_position[:, 1]), np.max(agent_future_position[:, 1]), np.max(agent_prediction_position[:, :, 1]))
        else:
            x_min = min(np.min(agent_historical_position[:, 0]), np.min(agent_prediction_position[:, :, 0]))
            x_max = max(np.max(agent_historical_position[:, 0]), np.max(agent_prediction_position[:, :, 0]))
            y_min = min(np.min(agent_historical_position[:, 1]), np.min(agent_prediction_position[:, :, 1]))
            y_max = max(np.max(agent_historical_position[:, 1]), np.max(agent_prediction_position[:, :, 1]))
        
        plt.xlim(x_min - 10, x_max + 10)
        plt.ylim(y_min - 10, y_max + 10)

        # Plot lanes
        lane_mask = lane_batch == i
        lanes_i = lane_position[lane_mask].cpu().numpy()
        if len(lanes_i) > 0:
            plt.scatter(
                lanes_i[:, 0],
                lanes_i[:, 1],
                c='#E0E0E0',
                s=5,
                alpha=0.5,
                zorder=0,
            )

        # History
        plt.plot(
            agent_historical_position[:, 0],
            agent_historical_position[:, 1],
            "-",
            color="green",
            alpha=1,
            linewidth=2,
            label="Historical Trajectory",
            zorder=2
        )
        plt.scatter(
            agent_historical_position[-1, 0],
            agent_historical_position[-1, 1],
            color="green",
            alpha=1,
            s=30,
            zorder=2
        )

        # GT
        if not is_test:
            plt.plot(
                agent_future_position[:, 0],
                agent_future_position[:, 1],
                "-",
                color="red",
                alpha=1,
                linewidth=2,
                label="Future Trajectory",
                zorder=2
            )
            plt.scatter(
                agent_future_position[-1, 0],
                agent_future_position[-1, 1],
                color="red",
                alpha=1,
                s=30,
                zorder=2
            )

        # Predict
        for j in range(num_modes):
            plt.plot(
                agent_prediction_position[j, :, 0],
                agent_prediction_position[j, :, 1],
                "-",
                color="blue",
                alpha=0.5,
                linewidth=1,
                label="Predicted Trajectory" if j == 0 else None,
                zorder=1
            )
            plt.scatter(
                agent_prediction_position[j, -1, 0],
                agent_prediction_position[j, -1, 1],
                color="blue",
                alpha=0.5,
                s=20,
                zorder=1
            )
        
        plt.legend()
        plt.axis("equal")
        
        if is_test:
            os.makedirs('test_output/visualization', exist_ok=True)
            plt.savefig(f'test_output/visualization/{data["scenario_name"][i]}_{data["case_id"][i]}.png', dpi=150, bbox_inches='tight')
        else:
            os.makedirs('visualization/trajectory', exist_ok=True)
            plt.savefig(f'visualization/trajectory/{data["scenario_name"][i]}_{data["case_id"][i]}.png', dpi=150, bbox_inches='tight')
        plt.close()
