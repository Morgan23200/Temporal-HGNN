from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SCATTER / POOL HELPERS
# ============================================================


def scatter_mean(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((dim_size, x.size(-1)))

    out = x.new_zeros((dim_size, x.size(-1)))
    count = x.new_zeros((dim_size, 1))
    out.index_add_(0, index, x)
    count.index_add_(0, index, x.new_ones((x.size(0), 1)))
    return out / count.clamp_min(1.0)



def scatter_max(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((dim_size, x.size(-1)))

    out = x.new_full((dim_size, x.size(-1)), float("-inf"))
    for group_id in range(dim_size):
        mask = index == group_id
        if mask.any():
            out[group_id] = x[mask].max(dim=0).values
    out[out == float("-inf")] = 0.0
    return out



def pool_nodes_to_videos(
    node_embeddings: torch.Tensor,
    node_video_idx: torch.Tensor,
    num_videos: int,
    pooling: str = "mean",
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    base_mask = node_video_idx >= 0
    if valid_mask is not None:
        base_mask = base_mask & valid_mask.bool()

    x = node_embeddings[base_mask]
    idx = node_video_idx[base_mask]

    if x.numel() == 0:
        out_dim = node_embeddings.size(-1)
        if pooling == "meanmax":
            out_dim *= 2
        return node_embeddings.new_zeros((num_videos, out_dim))

    if pooling == "mean":
        return scatter_mean(x, idx, num_videos)
    if pooling == "max":
        return scatter_max(x, idx, num_videos)
    if pooling == "meanmax":
        mean_part = scatter_mean(x, idx, num_videos)
        max_part = scatter_max(x, idx, num_videos)
        return torch.cat([mean_part, max_part], dim=-1)
    raise ValueError(f"Unsupported pooling={pooling}")


# ============================================================
# PURE-TORCH HYPERGRAPH CONV
# ============================================================


class HypergraphConvLite(nn.Module):
    """
    Lightweight hypergraph message passing with pure PyTorch.

    Input hyperedge_index has shape [2, num_incidence]:
      row 0 = node indices
      row 1 = hyperedge indices

    Update rule:
      node -> hyperedge: mean over incident node embeddings
      hyperedge -> node: mean over incident hyperedge embeddings
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return self.lin(x)
        if hyperedge_index.numel() == 0:
            out = self.lin(x)
            out = self.norm(out)
            return F.relu(out)

        node_idx = hyperedge_index[0].long()
        edge_idx = hyperedge_index[1].long()

        x_proj = self.lin(x)
        num_nodes = x_proj.size(0)
        num_edges = int(edge_idx.max().item()) + 1

        edge_sum = x_proj.new_zeros((num_edges, x_proj.size(1)))
        edge_cnt = x_proj.new_zeros((num_edges, 1))
        edge_sum.index_add_(0, edge_idx, x_proj[node_idx])
        edge_cnt.index_add_(0, edge_idx, x_proj.new_ones((node_idx.size(0), 1)))
        edge_emb = edge_sum / edge_cnt.clamp_min(1.0)

        node_sum = x_proj.new_zeros((num_nodes, x_proj.size(1)))
        node_cnt = x_proj.new_zeros((num_nodes, 1))
        node_sum.index_add_(0, node_idx, edge_emb[edge_idx])
        node_cnt.index_add_(0, node_idx, x_proj.new_ones((node_idx.size(0), 1)))
        out = node_sum / node_cnt.clamp_min(1.0)

        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# ============================================================
# INPUT EMBEDDING SELECTION
# ============================================================


class EmbeddingSelector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# TEMPORAL BLOCK
# ============================================================


class TemporalEdgeBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.src_lin = nn.Linear(channels, channels)
        self.dst_lin = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, temporal_edge_index: torch.Tensor) -> torch.Tensor:
        if temporal_edge_index.numel() == 0:
            return x

        src, dst = temporal_edge_index
        messages = self.src_lin(x[src])

        agg = x.new_zeros(x.size())
        agg.index_add_(0, dst, messages)

        count = x.new_zeros((x.size(0), 1))
        count.index_add_(0, dst, x.new_ones((dst.size(0), 1)))
        agg = agg / count.clamp_min(1.0)

        out = self.dst_lin(x) + agg
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# ============================================================
# FUSION BLOCK
# ============================================================


class RelationFusion(nn.Module):
    def __init__(self, channels: int, num_relations: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(channels * num_relations, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = dropout

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(tensors, dim=-1)
        x = self.proj(x)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ============================================================
# ENCODERS
# ============================================================


class MLPOnlyVideoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.selector = EmbeddingSelector(in_channels, hidden_channels, emb_dim, dropout)
        self.pooling = pooling
        self.output_dim = emb_dim if pooling in {"mean", "max"} else emb_dim * 2

    def forward(self, snapshot: dict) -> torch.Tensor:
        z = self.selector(snapshot["x_node"])
        return pool_nodes_to_videos(
            node_embeddings=z,
            node_video_idx=snapshot["node_video_idx_current"],
            num_videos=len(snapshot["video_ids"]),
            pooling=self.pooling,
        )


class StaticCategoryHypergraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.selector = EmbeddingSelector(in_channels, hidden_channels, emb_dim, dropout)
        self.conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.pooling = pooling
        self.output_dim = emb_dim if pooling in {"mean", "max"} else emb_dim * 2

    def forward(self, snapshot: dict) -> torch.Tensor:
        x = self.selector(snapshot["x_node"])
        edge_index = snapshot["hyperedge_index_category"]
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        return pool_nodes_to_videos(
            node_embeddings=x,
            node_video_idx=snapshot["node_video_idx_current"],
            num_videos=len(snapshot["video_ids"]),
            pooling=self.pooling,
        )


class StaticHeteroHypergraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.selector = EmbeddingSelector(in_channels, hidden_channels, emb_dim, dropout)

        self.video_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.country_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.category_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.fuse1 = RelationFusion(emb_dim, num_relations=3, dropout=dropout)

        self.video_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.country_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.category_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.fuse2 = RelationFusion(emb_dim, num_relations=3, dropout=dropout)

        self.pooling = pooling
        self.output_dim = emb_dim if pooling in {"mean", "max"} else emb_dim * 2

    def forward(self, snapshot: dict) -> torch.Tensor:
        z = self.selector(snapshot["x_node"])

        h_v = self.video_conv1(z, snapshot["hyperedge_index_video"])
        h_c = self.country_conv1(z, snapshot["hyperedge_index_country"])
        h_k = self.category_conv1(z, snapshot["hyperedge_index_category"])
        h = self.fuse1([h_v, h_c, h_k])

        h_v = self.video_conv2(h, snapshot["hyperedge_index_video"])
        h_c = self.country_conv2(h, snapshot["hyperedge_index_country"])
        h_k = self.category_conv2(h, snapshot["hyperedge_index_category"])
        h = self.fuse2([h_v, h_c, h_k])

        return pool_nodes_to_videos(
            node_embeddings=h,
            node_video_idx=snapshot["node_video_idx_current"],
            num_videos=len(snapshot["video_ids"]),
            pooling=self.pooling,
        )


class TemporalCategoryHypergraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.selector = EmbeddingSelector(in_channels, hidden_channels, emb_dim, dropout)
        self.conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.temporal_block = TemporalEdgeBlock(emb_dim, dropout=dropout)
        self.pooling = pooling
        self.output_dim = emb_dim if pooling in {"mean", "max"} else emb_dim * 2

    def forward(self, snapshot: dict) -> torch.Tensor:
        x = self.selector(snapshot["x_node"])
        edge_index = snapshot["hyperedge_index_category"]

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.temporal_block(x, snapshot["temporal_edge_index"])

        current_mask = snapshot["time_layer"] == 1
        return pool_nodes_to_videos(
            node_embeddings=x,
            node_video_idx=snapshot["node_video_idx_current"],
            num_videos=len(snapshot["video_ids"]),
            pooling=self.pooling,
            valid_mask=current_mask,
        )


class TemporalHeteroHypergraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.selector = EmbeddingSelector(in_channels, hidden_channels, emb_dim, dropout)

        self.video_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.country_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.category_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.fuse1 = RelationFusion(emb_dim, num_relations=3, dropout=dropout)

        self.video_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.country_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.category_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.fuse2 = RelationFusion(emb_dim, num_relations=3, dropout=dropout)

        self.temporal_block = TemporalEdgeBlock(emb_dim, dropout=dropout)
        self.pooling = pooling
        self.output_dim = emb_dim if pooling in {"mean", "max"} else emb_dim * 2

    def forward(self, snapshot: dict) -> torch.Tensor:
        z = self.selector(snapshot["x_node"])

        h_v = self.video_conv1(z, snapshot["hyperedge_index_video"])
        h_c = self.country_conv1(z, snapshot["hyperedge_index_country"])
        h_k = self.category_conv1(z, snapshot["hyperedge_index_category"])
        h = self.fuse1([h_v, h_c, h_k])

        h_v = self.video_conv2(h, snapshot["hyperedge_index_video"])
        h_c = self.country_conv2(h, snapshot["hyperedge_index_country"])
        h_k = self.category_conv2(h, snapshot["hyperedge_index_category"])
        h = self.fuse2([h_v, h_c, h_k])

        h = self.temporal_block(h, snapshot["temporal_edge_index"])

        current_mask = snapshot["time_layer"] == 1
        return pool_nodes_to_videos(
            node_embeddings=h,
            node_video_idx=snapshot["node_video_idx_current"],
            num_videos=len(snapshot["video_ids"]),
            pooling=self.pooling,
            valid_mask=current_mask,
        )


# ============================================================
# PREDICTOR + WRAPPER
# ============================================================


class CountryMLPPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        num_countries: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_countries),
        )

    def forward(self, video_emb: torch.Tensor) -> torch.Tensor:
        return self.net(video_emb)


class VideoCountryModel(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, snapshot: dict) -> torch.Tensor:
        video_emb = self.encoder(snapshot)
        return self.predictor(video_emb)


# ============================================================
# MODEL BUILDER
# ============================================================


def build_model(
    model_name: str,
    in_channels: int,
    num_countries: int,
    hidden_channels: int = 128,
    emb_dim: int = 128,
    dropout: float = 0.2,
    pooling: str = "mean",
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "mlp_only":
        encoder = MLPOnlyVideoEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_name == "static_category":
        encoder = StaticCategoryHypergraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_name == "static_hetero":
        encoder = StaticHeteroHypergraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_name == "temporal_category":
        encoder = TemporalCategoryHypergraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_name == "temporal_hetero":
        encoder = TemporalHeteroHypergraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            dropout=dropout,
            pooling=pooling,
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    predictor = CountryMLPPredictor(
        in_dim=encoder.output_dim,
        hidden_channels=hidden_channels,
        num_countries=num_countries,
        dropout=dropout,
    )
    return VideoCountryModel(encoder=encoder, predictor=predictor)
