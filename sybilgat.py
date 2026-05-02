import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class SybilGAT(nn.Module):
    """
    2-layer Graph Attention Network for Twitter bot detection.

    Architecture:
        Layer 1: GATConv(in_channels → hidden * heads, concat=True)
        BatchNorm + ELU + Dropout
        Layer 2: GATConv(hidden * heads → out_channels, concat=False)
        Skip connection: Linear(in_channels → out_channels)

    Args:
        in_channels:     Number of input features (217 for cresci-2017)
        hidden_channels: Hidden dim per attention head (default 64)
        out_channels:    Number of classes (2: human/bot)
        heads:           Number of attention heads (default 4)
        dropout:         Dropout probability (default 0.5)
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             dropout=dropout, concat=False)
        self.skip = nn.Linear(in_channels, out_channels)
        self.bn1  = nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, x, edge_index):
        x_in = F.dropout(x, p=self.dropout, training=self.training)
        x    = self.conv1(x_in, edge_index)
        x    = self.bn1(x)
        x    = F.elu(x)
        x    = F.dropout(x, p=self.dropout, training=self.training)
        x    = self.conv2(x, edge_index)
        x    = x + self.skip(x_in)
        return x


class BotGCN(nn.Module):
    """GCN baseline — same architecture as SybilGAT minus attention."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.skip  = nn.Linear(in_channels, out_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x_in = F.dropout(x, p=self.dropout, training=self.training)
        x    = self.conv1(x_in, edge_index)
        x    = self.bn1(x)
        x    = F.elu(x)
        x    = F.dropout(x, p=self.dropout, training=self.training)
        x    = self.conv2(x, edge_index)
        x    = x + self.skip(x_in)
        return x


class BotSAGE(nn.Module):
    """GraphSAGE baseline — inductive neighbor sampling."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.skip  = nn.Linear(in_channels, out_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x_in = F.dropout(x, p=self.dropout, training=self.training)
        x    = self.conv1(x_in, edge_index)
        x    = self.bn1(x)
        x    = F.elu(x)
        x    = F.dropout(x, p=self.dropout, training=self.training)
        x    = self.conv2(x, edge_index)
        x    = x + self.skip(x_in)
        return x
