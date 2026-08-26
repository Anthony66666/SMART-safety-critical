from typing import Dict, Optional
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from smart.modules.agent_decoder import SMARTAgentDecoder
from smart.modules.map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token: Dict,
                 token_data: Dict,
                 use_intention=False,
                 beam_size: int = 5,
                 token_size=512) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token
        )
        self.agent_encoder = SMARTAgentDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_size=token_size,
            token_data=token_data,
            beam_size=beam_size
        )
        self.map_enc = None

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference(self, data: HeteroData, ego_planner=None,
                  forced_tokens=None, temperature: float = 1.0,
                  tilt_beta=None, adversary_mask=None, victim_index=None,
                  tilt_topk=None) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder.inference(data, map_enc, ego_planner=ego_planner,
                                                 forced_tokens=forced_tokens,
                                                 temperature=temperature,
                                                 tilt_beta=tilt_beta,
                                                 adversary_mask=adversary_mask,
                                                 victim_index=victim_index,
                                                 tilt_topk=tilt_topk)
        return {**map_enc, **agent_enc}

    def inference_no_map(self, data: HeteroData, map_enc) -> Dict[str, torch.Tensor]:
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}
