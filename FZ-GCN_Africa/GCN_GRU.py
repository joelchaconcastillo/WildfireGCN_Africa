import torch
import torch.nn as nn
from GCN_GRU_Cell import GCN_GRU_Cell

class GCN_GRU(nn.Module):
    def __init__(self, node_num, dim_in, hidden_dim, link_len, embed_dim, num_layers=1, window_len = 10, num_persis_diagrams=2, supports_adj=None, return_all_layers=False):
        super(GCN_GRU, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.return_all_layers = return_all_layers 
        self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_len = window_len
        self.cell_list = nn.ModuleList()
        for i in range(0, num_layers):
            cur_input_dim = dim_in if i == 0 else hidden_dim
            self.cell_list.append(GCN_GRU_Cell(node_num, cur_input_dim, hidden_dim, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, i))

    def forward(self, x, node_embeddings, hidden_state = None):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        (batch_size, seq_len, input_dim, n) = x.shape
        hidden_state = self.init_hidden(batch_size)
        layer_output_list = []
        last_state_list = []
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                state = self.cell_list[layer_idx](cur_layer_input[:, t, :, :], state, cur_layer_input, node_embeddings)
                output_inner.append(state)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([state])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden_state(batch_size))
        return init_states

