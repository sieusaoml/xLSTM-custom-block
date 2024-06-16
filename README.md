import xlstm
from xlstm import (
    mLSTMLayerConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

scfg = sLSTMLayerConfig(
        embedding_dim=22,
        backend="vanilla",
        num_heads=11,
        conv1d_kernel_size=4,
        bias_init="powerlaw_blockdependent",
)
mcfg=mLSTMLayerConfig(
    proj_factor=1,
    round_proj_up_to_multiple_of=1,
    embedding_dim=22,
    context_length=1,
    conv1d_kernel_size=3, qkv_proj_blocksize=4, num_heads=4
)
fcfg = FeedForwardConfig(embedding_dim=22, act_fn="gelu")

model = xLSTM(layers=['s', 'm'], scfg=scfg, mcfg=mcfg, fcfg=fcfg)
