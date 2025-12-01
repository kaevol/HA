import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B
from utils import wrap_angle


class INTERACTIONDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30) -> None:
        self.root = root

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]
        self._processed_file_names = []
        
        for raw_path in tqdm(self.raw_paths, desc='Scanning raw files'):
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = os.path.splitext(raw_file_name)[0]
            df = pd.read_csv(raw_path)
            for case_id in df['case_id'].unique():
                self._processed_file_names.append(scenario_name + '_' + str(int(case_id)) + '.pt')
        
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps

        self._agent_type = ['agent', 'others']
        self._polyline_side = ['left', 'center', 'right']
        
        super(INTERACTIONDataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed_data')
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def parse_osm_xy(self, osm_path: str) -> Dict:
        """Parse custom OSM format with x,y coordinates instead of lat,lon."""
        tree = ET.parse(osm_path)
        root = tree.getroot()
        
        # Parse nodes
        nodes = {}
        for node in root.findall('node'):
            node_id = int(node.get('id'))
            x = float(node.get('x'))
            y = float(node.get('y'))
            nodes[node_id] = np.array([x, y])
        
        # Parse ways (lane boundaries)
        ways = {}
        for way in root.findall('way'):
            way_id = int(way.get('id'))
            node_refs = [int(nd.get('ref')) for nd in way.findall('nd')]
            tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            ways[way_id] = {
                'nodes': node_refs,
                'tags': tags,
                'points': np.array([nodes[ref] for ref in node_refs])
            }
        
        # Parse relations (lanelets)
        lanelets = {}
        for relation in root.findall('relation'):
            rel_id = int(relation.get('id'))
            members = {}
            for member in relation.findall('member'):
                role = member.get('role')
                ref = int(member.get('ref'))
                members[role] = ref
            tags = {tag.get('k'): tag.get('v') for tag in relation.findall('tag')}
            
            if tags.get('type') == 'lanelet':
                left_way_id = members.get('left')
                right_way_id = members.get('right')
                
                if left_way_id and right_way_id and left_way_id in ways and right_way_id in ways:
                    left_points = ways[left_way_id]['points']
                    right_points = ways[right_way_id]['points']
                    
                    # Compute centerline
                    min_len = min(len(left_points), len(right_points))
                    left_interp = np.array([np.interp(np.linspace(0, 1, min_len), 
                                                       np.linspace(0, 1, len(left_points)), 
                                                       left_points[:, i]) for i in range(2)]).T
                    right_interp = np.array([np.interp(np.linspace(0, 1, min_len), 
                                                        np.linspace(0, 1, len(right_points)), 
                                                        right_points[:, i]) for i in range(2)]).T
                    centerline = (left_interp + right_interp) / 2
                    
                    lanelets[rel_id] = {
                        'left_boundary': left_points,
                        'right_boundary': right_points,
                        'centerline': centerline,
                        'tags': tags
                    }
        
        return {'nodes': nodes, 'ways': ways, 'lanelets': lanelets}

    def process(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        
        for raw_path in tqdm(self.raw_paths, desc='Processing files'):
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = os.path.splitext(raw_file_name)[0]
            base_dir = os.path.dirname(raw_dir)
            map_path = os.path.join(base_dir, 'maps', scenario_name + '.osm')
            
            # Parse map
            map_data = self.parse_osm_xy(map_path)
            
            # Process trajectory data
            df = pd.read_csv(raw_path)
            for case_id, case_df in tqdm(df.groupby('case_id'), desc=f'Processing {scenario_name}', leave=False):
                data = dict()
                data['scenario_name'] = scenario_name
                data['case_id'] = int(case_id)
                data.update(self.get_features(case_df, map_data))
                torch.save(data, os.path.join(self.processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))

    def get_features(self, 
                     df: pd.DataFrame,
                     map_data: Dict) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'polyline': {},
            ('polyline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        
        # AGENT
        # Get unique frame_ids and filter to historical steps
        frame_ids = list(np.sort(df['frame_id'].unique()))
        historical_frames = frame_ids[:self.num_historical_steps]
        historical_df = df[df['frame_id'].isin(historical_frames)]
        agent_ids = list(historical_df['track_id'].unique())
        num_agents = len(agent_ids)
        df = df[df['track_id'].isin(agent_ids)]

        # Find the agent index (the one with agent_type == 'agent')
        agent_df = df[df['agent_type'] == 'agent']
        if len(agent_df) > 0:
            agent_track_id = agent_df['track_id'].values[0]
            agent_index = agent_ids.index(agent_track_id)
        else:
            agent_index = 0  # Default to first agent if no 'agent' type found

        # Initialization
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        agent_velocity = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_velocity_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_velocity_theta = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, dtype=torch.float)
        agent_width = torch.zeros(num_agents, dtype=torch.float)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = [frame_ids.index(frame_id) for frame_id in track_df['frame_id']]

            visible_mask[agent_idx, agent_steps] = True

            # Agent type: 0 for 'agent', 1 for 'others'
            agent_type_name = track_df['agent_type'].values[0]
            agent_type[agent_idx] = torch.tensor(self._agent_type.index(agent_type_name), dtype=torch.uint8)

            # Vehicle dimensions
            agent_length[agent_idx] = track_df['length'].values[0]
            agent_width[agent_idx] = track_df['width'].values[0]

            # Position
            agent_position[agent_idx, agent_steps] = torch.from_numpy(
                np.stack([track_df['x'].values, track_df['y'].values], axis=-1)
            ).float()

            # Heading from psi_rad
            agent_heading[agent_idx, agent_steps] = torch.from_numpy(track_df['psi_rad'].values).float()

            # Velocity
            agent_velocity[agent_idx, agent_steps] = torch.from_numpy(
                np.stack([track_df['vx'].values, track_df['vy'].values], axis=-1)
            ).float()
            
            # Compute velocity length and theta for historical steps
            velocity_length, velocity_theta = compute_angles_lengths_2D(agent_velocity[agent_idx])
            agent_velocity_length[agent_idx] = velocity_length[:self.num_historical_steps]
            
            # Compute relative angle between velocity direction and heading
            heading_hist = agent_heading[agent_idx, :self.num_historical_steps]
            agent_velocity_theta[agent_idx] = wrap_angle(velocity_theta[:self.num_historical_steps] - heading_hist)

        data['agent']['num_nodes'] = num_agents
        data['agent']['agent_index'] = agent_index
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['velocity_length'] = agent_velocity_length
        data['agent']['velocity_theta'] = agent_velocity_theta
        data['agent']['length'] = agent_length
        data['agent']['width'] = agent_width
        data['agent']['type'] = agent_type

        # MAP
        lanelets = map_data['lanelets']
        lane_ids = list(lanelets.keys())
        num_lanes = len(lane_ids)

        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)

        num_polylines = torch.zeros(num_lanes, dtype=torch.long)
        polyline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_length: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_side: List[Optional[torch.Tensor]] = [None] * num_lanes

        lane_left_neighbor_edge_index = []
        lane_right_neighbor_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []

        for lane_id in lane_ids:
            lane_idx = lane_ids.index(lane_id)
            lanelet = lanelets[lane_id]

            centerline = torch.from_numpy(lanelet['centerline']).float()
            left_boundary = torch.from_numpy(lanelet['left_boundary']).float()
            right_boundary = torch.from_numpy(lanelet['right_boundary']).float()

            # Lane center position and heading
            center_index = int((centerline.size(0) - 1) / 2)
            lane_position[lane_idx] = centerline[center_index, :2]
            if center_index + 1 < centerline.size(0):
                lane_heading[lane_idx] = torch.atan2(
                    centerline[center_index + 1, 1] - centerline[center_index, 1],
                    centerline[center_index + 1, 0] - centerline[center_index, 0]
                )
            lane_length[lane_idx] = torch.norm(centerline[1:] - centerline[:-1], p=2, dim=-1).sum()

            # Polylines (segments) for left, right boundaries and centerline
            left_vector = left_boundary[1:] - left_boundary[:-1]
            right_vector = right_boundary[1:] - right_boundary[:-1]
            centerline_vector = centerline[1:] - centerline[:-1]

            polyline_position[lane_idx] = torch.cat([
                (left_boundary[1:] + left_boundary[:-1]) / 2,
                (right_boundary[1:] + right_boundary[:-1]) / 2,
                (centerline[1:] + centerline[:-1]) / 2
            ], dim=0)

            polyline_length_temp, polyline_heading_temp = compute_angles_lengths_2D(
                torch.cat([left_vector, right_vector, centerline_vector], dim=0)
            )
            polyline_length[lane_idx] = polyline_length_temp
            polyline_heading[lane_idx] = polyline_heading_temp

            num_left_polyline = len(left_vector)
            num_right_polyline = len(right_vector)
            num_centerline_polyline = len(centerline_vector)

            polyline_side[lane_idx] = torch.cat([
                torch.full((num_left_polyline,), self._polyline_side.index('left'), dtype=torch.uint8),
                torch.full((num_right_polyline,), self._polyline_side.index('right'), dtype=torch.uint8),
                torch.full((num_centerline_polyline,), self._polyline_side.index('center'), dtype=torch.uint8)
            ], dim=0)

            num_polylines[lane_idx] = num_left_polyline + num_right_polyline + num_centerline_polyline

        # Store lane data
        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading

        # Store polyline data
        if num_lanes > 0:
            data['polyline']['num_nodes'] = num_polylines.sum().item()
            data['polyline']['position'] = torch.cat([p for p in polyline_position if p is not None], dim=0)
            data['polyline']['heading'] = torch.cat([p for p in polyline_heading if p is not None], dim=0)
            data['polyline']['length'] = torch.cat([p for p in polyline_length if p is not None], dim=0)
            data['polyline']['side'] = torch.cat([p for p in polyline_side if p is not None], dim=0)

            polyline_to_lane_edge_index = torch.stack([
                torch.arange(num_polylines.sum(), dtype=torch.long),
                torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_polylines)
            ], dim=0)
            data['polyline', 'lane']['polyline_to_lane_edge_index'] = polyline_to_lane_edge_index
        else:
            data['polyline']['num_nodes'] = 0
            data['polyline']['position'] = torch.zeros(0, 2, dtype=torch.float)
            data['polyline']['heading'] = torch.zeros(0, dtype=torch.float)
            data['polyline']['length'] = torch.zeros(0, dtype=torch.float)
            data['polyline']['side'] = torch.zeros(0, dtype=torch.uint8)
            data['polyline', 'lane']['polyline_to_lane_edge_index'] = torch.tensor([[], []], dtype=torch.long)

        # Lane-to-lane edges (empty for now - can be populated if topology is available)
        data['lane', 'lane']['left_neighbor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
        data['lane', 'lane']['right_neighbor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
        data['lane', 'lane']['predecessor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
        data['lane', 'lane']['successor_edge_index'] = torch.tensor([[], []], dtype=torch.long)

        return data

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        return HeteroData(torch.load(self.processed_paths[idx]))
