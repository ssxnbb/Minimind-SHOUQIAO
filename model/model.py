from transformers import PretrainedConfig



class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # 线性层使用的激活函数是silu
        hidden_act: str = "silu",
        hidden_size: int = 512,
        # 这个表示中间线性层FFN的维度
        intermediate_size: int = None,
        # 理论上支持的最大序列长度
        max_position_embeddings: int = 32768,
        # 8头注意力
        num_attention_heads: int = 8,
        # 表示堆叠8个transfomer的decoder层
        num_hidden_layers: int = 8,
        # kv只有两个头，相当于每4个Q头对应一个kv头
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        # rmsNorm当中的常数egexinuo，初始化为1e-05
        rms_norm_eps: float = 1e-05,
        # rope旋转编码设置斯塔的分母值
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
import torch
import torch.nn as nn
# 1.编写RMSnorm代码
class RMSNorm(nn.Module):
  # （1）初始化RSM_norm，我们需要知道输入token的维度，因为需要对其归一化，然后还要知道eps
  def __init__(self,dim:int,eps:float = 1e-5):
    super().__init__()
    self.dim=dim
    self.eps=eps
    #  缩放系数要对token的每一个维度都要进行缩放，初始化为1，之后进行学习
    self.weight=nn.Parameter(torch.ones(dim)) 
  # （2）返回归一化系数，只有分母那一坨
  def norm(self,x):
    return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
  # （3）前向传播返回归一化的最终值
  def forward(self,x):
    return x*self.norm(x)*self.weight
if __name__ == "__main__":
    x = torch.tensor([
        [
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ]
    ])

    rmsnorm = RMSNorm(dim=4)

    y = rmsnorm(x)

    print("输入 shape:", x.shape)
    print("norm shape:", rmsnorm.norm(x).shape)
    print("weight shape:", rmsnorm.weight.shape)
    print("输出 shape:", y.shape)

    print("norm:\n", rmsnorm.norm(x))
    print("output:\n", y)