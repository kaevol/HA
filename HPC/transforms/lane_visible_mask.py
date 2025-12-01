import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class LaneVisibleMask(BaseTransform):
    """Transform that adds visible_mask to lanes (all visible, no occlusion)."""
    
    def __init__(self):
        super(LaneVisibleMask, self).__init__()

    def __call__(self, data: HeteroData) -> HeteroData:
        # Set all lanes as visible (no occlusion)
        visible_mask = torch.ones(data['lane']['num_nodes'], dtype=torch.bool)
        data['lane']['visible_mask'] = visible_mask
        return data
