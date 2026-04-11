"""
Multi-Zone Temperature Prediction with Physics-Constrained Graph Neural Network (PC-GNN)

This module implements Stage 1 of the multi-stage framework described in the associated paper.
It combines LSTM temporal encoding with a multi-head graph attention network.
Three types of physical constraints are incorporated: Structural Physics (SP), Physical Consistency Loss (PL), and Multi-step Correction (MC).

The code is intended for methodological review and is not executable without the dataset and environment configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class EnhancedResidualBlock(nn.Module):
    """
    Residual block with layer normalization, dual-branch design, and gating mechanism.
    Improves gradient flow and representation capacity for complex thermal dynamics.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        hidden_dim = min(hidden_dim, input_dim * 2)

        self.branch1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim // 2),
        )
        self.branch2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim // 2),
        )
        self.combine = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=-1)
        out = self.combine(out)
        gate = self.gate(x)
        out = gate * out + (1 - gate) * residual
        return self.layer_norm(out)


class EnhancedGNNLayer(nn.Module):
    """
    Multi-head graph convolution with dynamic edge weights and temperature-aware gating.
    Corresponds to the message-passing mechanism illustrated in Fig.5 of the paper.
    """
    def __init__(
        self,
        hidden_size: int,
        adjacency_matrix: torch.Tensor,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.register_buffer("adjacency", adjacency_matrix)

        num_zones = adjacency_matrix.shape[0]
        self.edge_weights = nn.Parameter(torch.ones(num_zones, num_zones))

        self.W_self = nn.ModuleList([
            nn.Linear(hidden_size, self.head_dim) for _ in range(num_heads)
        ])
        self.W_neigh = nn.ModuleList([
            nn.Linear(hidden_size, self.head_dim) for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Temperature-aware gate: controls message passing intensity based on thermal state
        self.temp_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, zone_temps: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_zones, _ = x.shape
        norm_x = self.layer_norm1(x)

        # Dynamic adjacency matrix (Eq. 2)
        dynamic_adj = self.adjacency * torch.sigmoid(self.edge_weights)
        eye = torch.eye(num_zones, device=dynamic_adj.device)
        dynamic_adj = dynamic_adj * (1 - eye) + eye
        row_sum = dynamic_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        norm_adj = dynamic_adj / row_sum
        batch_adj = norm_adj.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head aggregation
        head_outputs = []
        for h in range(self.num_heads):
            self_feat = self.W_self[h](norm_x)
            neigh_feat = self.W_neigh[h](norm_x)
            agg_neigh = torch.bmm(batch_adj, neigh_feat)
            head_outputs.append(self_feat + agg_neigh)

        multi_head = torch.cat(head_outputs, dim=-1)
        out = self.output_proj(multi_head)
        out = self.dropout(out)

        if zone_temps is not None:
            gate_input = torch.cat([
                x, zone_temps.unsqueeze(-1).expand(-1, -1, x.shape[-1] // 2)
            ], dim=-1)
            gate_val = self.temp_gate(gate_input)
            out = gate_val * out

        out = x + out
        out = self.layer_norm2(out)
        return self.activation(out)


class PCGNN(nn.Module):
    """
    Physics-Constrained Graph Neural Network for multi-zone temperature prediction.
    Architecture: Input -> LayerNorm -> Shared Feature Extractor -> LSTM -> Multi-Head GNN ->
                  Zone-Specific Heads -> Multi-step Correction (MC).
    Physical constraints (SP, PL, MC) are enforced during training.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_zones: int = 8,
        prediction_horizon: int = 24,
        adjacency_matrix: Optional[np.ndarray] = None,
        dropout: float = 0.25,
        use_attention: bool = True,
        zone_type_info: Optional[torch.Tensor] = None,
        low_temp_focus: bool = True,
    ):
        super().__init__()
        self.num_zones = num_zones
        self.prediction_horizon = prediction_horizon
        self.use_attention = use_attention
        self.low_temp_focus = low_temp_focus

        if adjacency_matrix is None:
            adjacency_matrix = np.eye(num_zones)
        self.register_buffer("adjacency", torch.tensor(adjacency_matrix, dtype=torch.float32))

        # zone_type_info: [num_zones, 2] with [interior_flag, exterior_flag]
        if zone_type_info is None:
            zone_type_info = torch.ones(num_zones, 2)
        self.register_buffer("zone_type_info", zone_type_info)

        self.input_norm = nn.LayerNorm(input_dim)

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            EnhancedResidualBlock(hidden_dim, hidden_dim // 2, dropout),
            EnhancedResidualBlock(hidden_dim, hidden_dim // 2, dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5]
        ])
        self.conv_norm = nn.LayerNorm(hidden_dim)

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
            )

        self.gnn_layers = nn.ModuleList([
            EnhancedGNNLayer(hidden_dim, self.adjacency, num_heads=4, dropout=dropout)
            for _ in range(2)
        ])

        self.zone_specific = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                EnhancedResidualBlock(hidden_dim, hidden_dim // 2, dropout),
            )
            for _ in range(num_zones)
        ])

        # Specialized layers for interior and exterior zones (Table 1)
        self.interior_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.exterior_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        if low_temp_focus:
            self.low_temp_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.temp_detector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1), nn.Sigmoid()
            )

        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, prediction_horizon),
            )
            for _ in range(num_zones)
        ])

        # Learnable parameters for multi-step correction (MC)
        self.time_decay = nn.Parameter(0.95 * torch.ones(prediction_horizon))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = self.input_norm(x)
        x_flat = x.view(-1, x.shape[-1])
        features = self.feature_extractor(x_flat)
        features = features.view(batch_size, seq_len, -1)

        # Multi-scale temporal convolutions
        conv_in = features.transpose(1, 2)
        conv_outs = [conv(conv_in) for conv in self.conv_layers]
        conv_feat = sum(conv_outs) / len(conv_outs)
        features = features + self.conv_norm(conv_feat.transpose(1, 2))

        # LSTM temporal encoding
        lstm_out, _ = self.lstm(features)

        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            gate = self.attn_gate(lstm_out)
            lstm_out = lstm_out + gate * self.attn_norm(attn_out)

        final_feat = lstm_out[:, -1, :]  # [batch, hidden]
        zone_feat = final_feat.unsqueeze(1).expand(-1, self.num_zones, -1)

        # GNN spatial coupling
        for gnn in self.gnn_layers:
            zone_feat = gnn(zone_feat)

        zone_outputs = []
        for i in range(self.num_zones):
            zf = zone_feat[:, i, :]
            zf = self.zone_specific[i](zf)

            # Blend interior/exterior specialized features based on zone type
            is_interior = self.zone_type_info[i, 0]
            is_exterior = self.zone_type_info[i, 1]
            inter_proc = self.interior_layer(zf)
            exter_proc = self.exterior_layer(zf)
            zf = is_interior * inter_proc + is_exterior * exter_proc

            if self.low_temp_focus:
                temp_level = self.temp_detector(zf)
                low_feat = self.low_temp_layer(zf)
                zf = (1 - temp_level) * zf + temp_level * low_feat

            pred = self.predictors[i](zf)
            zone_outputs.append(pred)

        preds = torch.stack(zone_outputs, dim=1)  # [batch, zones, horizon]
        preds = self.apply_multi_step_correction(preds)
        return preds

    def apply_multi_step_correction(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Multi-step Correction (MC) as described in Supplementary S2.
        Uses influence matrix, momentum term, and time decay to enforce thermal inertia.
        """
        corrected = preds.clone()
        num_zones = self.num_zones
        device = preds.device

        influence = self.adjacency.clone()
        influence = influence / (influence.sum(dim=-1, keepdim=True).clamp(min=1e-6))

        for t in range(1, self.prediction_horizon):
            prev = corrected[:, :, t - 1]
            direct = preds[:, :, t]

            neighbor_inf = torch.matmul(prev.unsqueeze(1), influence.unsqueeze(0)).squeeze(1)

            time_factor = 0.1 + 0.3 * (1 - torch.exp(torch.tensor(-0.1 * t, device=device)))
            mixed = (1 - time_factor) * direct + time_factor * neighbor_inf

            if t > 1:
                momentum = corrected[:, :, t - 1] - corrected[:, :, t - 2]
                mixed = mixed + 0.2 * torch.sigmoid(self.time_decay[t]) * momentum

            decay = torch.sigmoid(self.time_decay[t]) * 0.1 + 0.9
            corrected[:, :, t] = mixed * decay

        return corrected


class PhysicalConsistencyLoss(nn.Module):
    """
    Physical Consistency Loss (PL) that penalizes predictions violating the
    basic direction of heat transfer (temperature difference drives heat flow).
    """
    def __init__(self, adjacency: torch.Tensor, lambda_phy: float = 0.15):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.lambda_phy = lambda_phy
        self.base_loss = nn.SmoothL1Loss(beta=0.3)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, current_temps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_loss = self.base_loss(preds[:, :, 0], targets)

        pred_change = preds[:, :, 0] - current_temps
        physics_loss = 0.0
        batch_size, num_zones = preds.shape[:2]

        for i in range(num_zones):
            neighbors = (self.adjacency[i] > 0).nonzero(as_tuple=True)[0]
            if len(neighbors) == 0:
                continue
            neighbor_temps = current_temps[:, neighbors]
            avg_neighbor = neighbor_temps.mean(dim=1)
            temp_diff = current_temps[:, i] - avg_neighbor
            # Violation occurs when temp_diff and pred_change have the same sign
            violation = F.softplus(temp_diff * pred_change[:, i])
            physics_loss += violation.mean()

        physics_loss = physics_loss / num_zones
        total_loss = pred_loss + self.lambda_phy * physics_loss
        return total_loss, physics_loss


# Training loop skeleton (for illustration only)

def train_model_example():
    """
    Example training routine showing key steps. Actual implementation requires
    data loaders, scalers, and full configuration (see paper Appendix Table A5).
    """
    adjacency = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 1]
    ])  # Eq. (1) adjacency matrix
    model = PCGNN(input_dim=40, hidden_dim=96, num_zones=8, adjacency_matrix=adjacency)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-5)
    criterion = PhysicalConsistencyLoss(adjacency=torch.tensor(adjacency), lambda_phy=0.15)

    # Placeholder for training loop:
    # for epoch in range(150):
    #     for batch in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss, _ = criterion(outputs, targets, current_temps)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    pass
