from sympy import im
from transformers import PretrainedConfig
from typing import Optional,Tuple
import math
import torch.nn.functional as F
from transformers.activations import ACT2FN
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
  #rsqrt的作用就是开方求倒数，mean函数的作用是求平均值，对x的平方相加再求均值
  def norm(self,x):
    return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
  # （3）前向传播返回归一化的最终值
  def forward(self,x):
    return x*self.norm(x)*self.weight
# 2.编写Rope位置编码
def precompute_freqs( 
    # 这个通常是attention_head这个维度
    dim: int,
    # 这个表示预计算多少位置，位置编码的最大长度
    end: int = int(32 * 1024),
    # 这个表示公式当中的base基值就是哪个10000
    rope_base: float = 1e6,
    # 如果不为 None，就启用 YaRN 这种长上下文缩放
    rope_scaling: Optional[dict] = None,):
    #(1)首先去计算最基本的频率
    freq=1/(rope_base**(torch.arange(0,dim,2)[: (dim // 2)].float()/dim))
    attention_temperature=1.0
    #(2)看是否启用YarN即看是否end超出origin_length
    if rope_scaling is not None:
        origin_len,factor,fast_r,low_r,attention_temperature=(
           rope_scaling.get('original_max_position_embeddings',2048),
           rope_scaling.get('factor',16),
           rope_scaling.get('beta_fast',32),
           rope_scaling.get('beta_slow',1),
           rope_scaling.get("attention_factor", 1.0)
        )
        if end>origin_len:
            # (3)设置函数可以将对应圈数转化为频率当中的坐标
            inv_dim=lambda r:math.log(origin_len/(math.pi*2*r))*dim/(2*math.log(rope_base))
            # (4)找到快慢圈对应的频率坐标,缩放频率
            low,high=max(math.floor(inv_dim(fast_r)),0),min(math.ceil(inv_dim(low_r)),dim//2-1)
            # (5)找到缩放系数
            ramp=torch.clamp(
            #    第一个必须加括号,因为除法优先级高于剑法
                (torch.arange(0,dim//2,device=freq.device).float()-low)/max(high - low, 0.001),
                0,
                1
            )
            freq=freq*(1-ramp+ramp/factor)
    # 构造到end的位置,用位置乘频率就是位置编码然后再cos,sin
    t=torch.arange(0,end,device=freq.device)
    pos_code=torch.outer(t,freq).float()
    # 因为一个位置需要两个cos和两个sin,所以需要扩展
    pos_cos=torch.cat([torch.cos(pos_code),torch.cos(pos_code)],dim=-1)*attention_temperature
    pos_sin=torch.cat([torch.sin(pos_code),torch.sin(pos_code)],dim=-1)*attention_temperature
    return pos_cos,pos_sin
# 3，将位置编码作用于Q和K
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 假设原来的维度是[x,y]，那么现在去构造[-y,x]，它这种方法是对前一半维度第一个和后一半维度第一个，拼成一个向量进行旋转
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        # 对位置编码在1维度加一个维度，是对注意力头进行广播
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
# 4.因为使用GQA需要对k和v进行复制
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        # 这个做法相当于unsqueeze直接在第三个维度增加一个维度，并且这个维度大小是1
        x[:, :, :, None, :]
        # expand是在逻辑上扩大维度,底层引用的还是原来的向量
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        # reshape是直接修改张量的维度
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )
# 5.attention机制
class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        # 设置kv的头,如果之前没有设置kv头那么就让kv的头等于原来的attention头的个数
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        # assert断言，如果判断条件为False就终止进行，如果条件为True就继续往下跑下去
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # 用总注意力头的数量除kv头的数量等于重复的次数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 总隐藏层维度除头的数量，得到每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        # 在于对算出分数之后对分数进行dropout，让模型在训练的过程中不再固定的依赖某几个token
        self.attn_dropout = nn.Dropout(args.dropout)
        # 这个表示在残差连接之前对神经网络的输出进行dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 第一个参数是检查配置是否支持，hasattr当中的两个参数表示，检查A里面是不是有属性B
        # 第二个参数表示用户是否要开启高效attention机制，只要两个都成立才会开启高效attention机制
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )
    def forward(
    self,
    x: torch.Tensor,
    # sin和cos的位置参数
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    # 推理所用的kv_cache
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache=False,
    attention_mask: Optional[torch.Tensor] = None,
    ):
        # 首先获取x的形状参数
        bsz, seq_len, _ = x.shape
        # 这三个代表带有权重w的参数矩阵，可以随反向传播进行更新。它们输入维度都是hid_dim，然后把x输出得到结果
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 改变张量形状开始进行分头(view只能用于空间连续张量的重塑)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 只要cos和sin的维度是head_dim这个加位置编码的操作就可以实现
        # 相当于对所有头都添加位置向量
        # 这个函数的作用就是真正把cos和sin变成旋转矩阵作用与xq和xk
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            # transpose的作用是对张量维度进行交换，交换第一维和第二维的维度
            xq.transpose(1, 2),
            # 对kv进行复制并交换1，2的维度,形状变成[B,H,S,D],复制之后Q,K,V的头都是一样的
            # 为什么要交换1,2维度,是因为这样可以让最后两个维度是S和D这样方便Qk相乘
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        # 如果满足高效attention机制就让高效attention机制自动计算
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 这个@矩阵乘法只对最后两维进行相乘，所以要对最后两个维度进行转置,score形状是[B,H,S,S]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 进行加减矩阵的时候自动进行维度对齐，从最后一维从后往前进行对齐，所以会把掩码矩阵从二维扩展到四维
            scores[:, :, :, -seq_len:] += torch.triu(
                # torch.full创建一个指定形状的张量这个张量的形状是(seq_len,seq_len)，值全是float('-inf')
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                # 外层包裹triu函数并将diagonal这个参数设置为1,这样对角线以下包括对角线元素都是0,其余元素都是负无穷
                diagonal=1,
            )
            # 这个attention_mask与上面的掩码矩阵不一样，它是为了防止看到padding，前面的是为了防止看到未来
            if attention_mask is not None:
                # 这个本质上也是对score进行处理,将其形状扩展为[B,1,1,S],然后与Score相加的时候会自动广播将1,1广播为H和S,这样就可以对pad不去计算它的注意力分数
                # attention_mask掩码一半1表示可以看见这个token,0表示对这个token进行pad
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 这样原本为0的部分就会变成负无穷,之后再与scores相加
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            # 经过casual掩码和attention掩码的处理之后再经过softmax函数
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 对输出的attention分数进行dropout
            scores = self.attn_dropout(scores)
            output = scores @ xv
        # 原本维度是[B,H,S,D],必须先更换轴,将H和S互换,然后再reshape之后变成[B,S,H*D],这个叫做合并注意力头
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 在残差连接之前对融合的信息进行dropout之后在block的时候再进行残差连接
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
# 6.FFN前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 因为传统的FFN是采用两个线性层，d到4d和4d到d，总参数是8d的平方，这个是三个线性层，两个升维，一个降维，d到md，md到d
            # 3md的平方，如果想让它们接近，m的值必须接近8/3
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 这个是让维度对齐到64的整数倍并且向上取整，之所以这么做，是因为64的倍数GPU更好计算，例如原来维度是80，向上取64的整数倍直接变成128
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # 在残差连接之前还需要进行dropout防止过拟合
        return self.dropout(self.down_proj(gated))
   
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