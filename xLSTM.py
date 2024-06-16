import torch
from torch import nn
import xlstm
from xlstm.blocks.slstm.layer import sLSTMLayer
from xlstm.blocks.mlstm.layer import mLSTMLayer
from xlstm.components.feedforward import create_feedforward
import xlstm.components.ln as ln

class xLSTM(nn.Module):
  def __init__(self, layers, scfg=None, mcfg=None, fcfg=None):
    super().__init__()
    self.layers = layers
    embedding_dim = (mcfg.embedding_dim if mcfg is not None else scfg.embedding_dim)
    self.xlstm_norm = nn.ModuleList()
    self.xlstm_blocks = nn.ModuleList()
    self.ffn_norm = nn.ModuleList()
    self.ffn = nn.ModuleList()
    for i in range(len(layers)):
      self.xlstm_norm.append(ln.LayerNorm(ndim=embedding_dim, weight=True, bias=False))
      if layers[i] == 's':
        self.xlstm_blocks.append(sLSTMLayer(scfg))
      else:
        self.xlstm_blocks.append(mLSTMLayer(mcfg))
      self.ffn_norm.append(ln.LayerNorm(ndim=embedding_dim, weight=True, bias=False))
      self.ffn.append(create_feedforward(fcfg))
    self.post_blocks_norm = ln.LayerNorm(ndim=embedding_dim)
    if scfg is not None:
      scfg.__post_init__()
    if mcfg is not None:
      mcfg.__post_init__()
    self.reset_parameters()

  def forward(self, x, hidden):
    if hidden is None:
      hidden = {}
    for block_idx, block in enumerate(self.xlstm_blocks):
      if self.layers[block_idx] == 's':
        x, hidden[f'block_{block_idx}'] = block(self.xlstm_norm[block_idx](x), hidden.get(f'block_{block_idx}', None), return_last_state=True)
      else:
        x = block(self.xlstm_norm[block_idx](x))
      x = x + self.ffn[block_idx](self.ffn_norm[block_idx](x))
    x = self.post_blocks_norm(x)
    return x, hidden

  def reset_parameters(self):
    for i in range(len(self.layers)):
      self.xlstm_norm[i].reset_parameters()
      self.xlstm_blocks[i].reset_parameters()
      self.ffn_norm[i].reset_parameters()
      self.ffn[i].reset_parameters()
    self.post_blocks_norm.reset_parameters()
