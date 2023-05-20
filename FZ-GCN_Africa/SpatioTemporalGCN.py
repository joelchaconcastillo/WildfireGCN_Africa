import torch.nn.functional as F
import torch
import torch.nn as nn

class SpatioTemporalGCN(nn.Module):
    def __init__(self, dim_in, dim_out, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos = 0):
        super(SpatioTemporalGCN, self).__init__()
        subhidden_num = 2  #number of sub-hidden layers (adaptive adj matrix, time matrix, spatial matrix)
        self.link_len = link_len
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/subhidden_num)))
        self.weights_pool_adj = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/subhidden_num)))
        if layer_pos == 0:
           self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / subhidden_num))) #broadcasting with all input features 
        else:
           self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in/2), int(dim_out / subhidden_num)))  #broadcasting with half of hidden units
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.T = nn.Parameter(torch.FloatTensor(window_len))
        self.supports_adj = supports_adj


    def forward(self, x, x_window, node_embeddings):
        '''
           x: B, N, F
           node_num :  N, E
        '''
        (batch_size, lag, node_num, dim) = x_window.shape
        #S1: Graph construction, a suggestion is to pre-process graph, however since wildfire requires ~1TB for pre-processing graph we create it from fly
        #S2: Laplacian construction

        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        #S3: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S4: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #N, link_len, dim_in, hidden_dim/2 : on E
        bias = torch.matmul(node_embeddings, self.bias_pool) #N, hidden_dim : on E
        x_g = torch.einsum("knm,bmc->bknc", supports, x) #B, link_len, N, dim_in : on N
        x_g = x_g.permute(0, 2, 1, 3) #B, N, link_len, dim_in  : on 
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) #B, N, hidden_dim/2
        x_gconv = F.normalize(x_gconv, dim=-1)


        weights_adj = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool_adj) #N, link_len, dim_in, hidden_dim/2 : on E
        x_a = torch.einsum("knn,bmc->bknc", self.supports_adj, x) #B, link_len, N, dim_in : on N
        x_a = x_a.permute(0, 2, 1, 3) #B, N, link_len, dim_in  : on 
        x_aconv = torch.einsum('bnki,nkio->bno', x_a, weights_adj) #B, N, hidden_dim/2
        x_aconv = F.normalize(x_aconv, dim=-1)


        #S5: temporal feature transformation
        weights_window = torch.einsum('nd,dio->nio', node_embeddings, self.weights_window)  #N, dim_in, hidden_dim/2 : on E
        x_w = torch.einsum('btni,nio->btno', x_window, weights_window)  #B, T, N, hidden_dim/2: on D
        x_w = x_w.permute(0, 2, 3, 1)  #B, N, hidden_dim/2, T
        x_wconv = torch.matmul(x_w, self.T)  #B, N, hidden_dim/2: on T
        x_wconv = F.normalize(x_wconv, dim=-1)

        x_tagconv = x_aconv + x_gconv 
        x_twconv =  x_wconv 

#        #S7: combination operation
        x_gwconv = torch.cat([x_tagconv, x_twconv], dim = -1) + bias #B, N, hidden_dim
        return x_gwconv

