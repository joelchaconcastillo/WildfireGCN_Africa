import torch.nn as nn
import torch
from SpatioTemporalGCN import SpatioTemporalGCN

class GCN_GRU_Cell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos):
        super(GCN_GRU_Cell, self).__init__()
        self.node_num = node_num
        self.dim_out = dim_out
        self.gate = SpatioTemporalGCN(dim_in+dim_out, 2*dim_out, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos)
        self.update = SpatioTemporalGCN(dim_in+dim_out, dim_out, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos)

    def forward(self, x, state, x_full, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        h_current = state.to(x.device)
        input_and_state = torch.cat((x, h_current), dim=-1) #x + state
        z_r = torch.sigmoid(self.gate(input_and_state, x_full, node_embeddings))
        z, r = torch.split(z_r, self.dim_out, dim=-1)
        candidate = torch.cat((x, r*h_current), dim=-1)
        n = torch.tanh(self.update(candidate, x_full, node_embeddings))
        h_next = (1.0-z)*n + z*h_current
        return h_next

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.dim_out)

