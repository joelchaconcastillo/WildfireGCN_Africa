import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from ViT import VisionTransformer
from GCN_GRU import GCN_GRU

def Laplacian_link(supports, link_len):
        node_num, _ = supports.shape
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        return torch.stack(support_set, dim=0)

def Image2Graph(num_nodes, patch_height, patch_width, device, link_len):
      adj = np.identity((num_nodes))
      nextR = [0, 1, 1, 1, 0, -1, -1, -1]  #displacement by rows
      nextC = [-1, -1, 0, 1, 1, 1, 0, -1]  #displacement by cols
      ##note that we are considering self-loops!!
      ##create graph with 8 neightbours..
      middlePixelR = patch_height/2
      for i in range(patch_height):
#          if i < middlePixelR-1 or i > middlePixelR+1:
#              continue
          for j in range(patch_width):
#              if j < middlePixelC-1 or j > middlePixelC+1:
#                 continue
              id_node = i*patch_width + j
              for k in range(len(nextR)):
                  nr, nc = i+nextR[k], j+nextC[k]
                  if nr<0 or nr>=patch_height or nc <0 or nc>=patch_width:
                      continue
                  id_node_next = nr*patch_width + nc
                  adj[id_node, id_node_next] +=1
                  adj[id_node_next, id_node] +=1
      support = torch.from_numpy(adj).float().to(device)
      support /= torch.sum(support, dim=1) #normalize by rows..
      supports_adj = Laplacian_link(support, link_len)
      return supports_adj

class GCN(nn.Module):
   def __init__(self, args):
      super().__init__()
      input_dim = args.input_dim
      self.num_nodes = int(args.patch_width)*int(args.patch_height)
      self.patch_width = args.patch_width
      self.patch_height = args.patch_height
      self.hidden_dim = args.rnn_units
      self.input_dim = args.input_dim
      self.output_dim = args.output_dim
      self.num_layers = args.num_layers
      self.embed_dim = args.embed_dim
      self.window_len = args.window_len
      self.link_len = args.link_len
      self.horizon = args.horizon
      self.num_persis_diagrams = 2*len(args.scaleParameter)
      self.ln1 = torch.nn.LayerNorm(self.input_dim)

      self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        #predictor
        # fully-connected part
      self.fc2 = nn.Linear(self.num_nodes*self.hidden_dim, 2)
      supports_adj = Image2Graph(self.num_nodes, self.patch_height, self.patch_width, args.device, self.link_len)
      self.encoder = GCN_GRU(self.num_nodes, self.input_dim, self.hidden_dim, self.link_len, self.embed_dim, self.num_layers, self.window_len, self.num_persis_diagrams, supports_adj)
      
      patch_size = 16
      totalsize = 64*64
      embed_dim = patch_size*4
      inChannelZPI = 2
      num_heads= 8
      num_patches = int(totalsize/(patch_size**2))
      hidden_dim = 16
      self.ViT = VisionTransformer(embed_dim = embed_dim, hidden_dim = hidden_dim, num_heads = num_heads, num_layers = 1, patch_size = patch_size, num_channels = inChannelZPI, num_patches = num_patches, num_classes = 2, dropout = 0.0)


   def forward(self, x: torch.Tensor, ZPI: torch.Tensor):
      '''
         x :     batch, time, features, nodes (width x height pixels)
         graph:  batch, time, nodes, nodes
         target:  batch, prediction
      '''
      (B, T, W, H, D) = x.shape
      x = x.float().reshape(B, T, W*H, D) ## B, T, N, F
      ZPI = ZPI.float()
      #x = x.permute(0, 1, 3, 2) ## B, T, N, D
      x = self.ln1(x)
      (B,T,N,D) = x.shape
    
      x, _ = self.encoder(x, self.node_embeddings) #B, T, N, hidden_dim
      x = x[0][:, -1:, :, :] #B, 1, N, hidden_dim

      # fully-connected
      x = torch.flatten(x, 1) ##B, hidden*12*12 (4608)
      x = self.fc2(x)

      (_, num_pers, zpiW, zpiH) = ZPI.shape
      ZPI = ZPI.reshape(B, -1, zpiH, zpiH)
      x_ZPI = self.ViT(ZPI)
      x = x + x_ZPI 

      return torch.nn.functional.log_softmax(x, dim=1)
