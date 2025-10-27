from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
from modules import *
from utils import extract



class Diffusion():
    def __init__(self, args):
        self.timesteps = args.timesteps
        # Beta值：控制每个时间步添加的噪声量
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.args = args

        # beta的调度策略
        if self.args.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.args.beta_sche == 'exp':
            self.betas = self.exp_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='cosine':
            self.betas = self.cosine_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='sqrt':
            self.betas = torch.tensor(self.betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas（timesteps, ）
        # alphas-保持原有信息的比例
        self.alphas = 1. - self.betas
        #alphas的类累计乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # 前一个时间步的累计乘积
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # 用于反向采样步骤
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        beta_start = beta_start
        beta_end = beta_end
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        x = torch.linspace(1, 2 * timesteps + 1, timesteps)
        betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
        return betas

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


class InDiRec(nn.Module):

    def __init__(self, device, args, num_heads=1):
        super(InDiRec, self).__init__()
        self.state_size = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.item_num = args.item_size
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.diffuser_type = args.diffuser_type
        self.device = device
        self.w = args.w

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.seq_model = SASRecModel(self.item_embeddings, args=args)

        self.diff = Diffusion(args)


        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)


        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )
    #有条件的扩散，输出值是去噪后的结果，这里的x是x-noisy
    def forward(self, x, s, step):
        #  x:(batch, dim) , h:(batch, dim), t:(batch, )
        # 将离散的时间步转换成向量
        t = self.step_mlp(step) # t:(batch, dim)


        if self.diffuser_type == 'mlp1':
            # 连接三个向量，输入给MLP（扩散模型预测）
            res = self.diffuser(torch.cat((x, s, t), dim=1)) # (batch, 3 x dim) -->diffuser-->(batch, dim)
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, s, t), dim=1))
        return res # (batch, dim)
    # 无条件的扩散
    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res
    #将物品ID转换为嵌入向量
    def calculate_x0(self, x0):
        return self.item_embeddings(x0)

    def calculate_s(self, input_seq, p):
        
        seq_out = self.seq_model(input_seq)
        s = seq_out[:,-1,:] # B x Dim 

        B, D = s.shape[0], s.shape[1] # B = batch_size，D = embedding_dim
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2 # (B,) 
        maske1d = mask1d.view(B, 1) # (B, 1) 。
        mask = torch.cat([maske1d] * D, dim=1) 
        mask = mask.to(self.device)

        s = s * mask + self.none_embedding(torch.tensor([0], device=self.device)) * (1 - mask)

        return s
    # 前向过程：加噪
    def forward_process(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # 根据t取出参数
        sqrt_alphas_cumprod_t = extract(self.diff.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.diff.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # 反向生成与条件s类似的序列
    @torch.no_grad()
    def sample_from_reverse_process(self, s):
        # 初始化x为标准高斯噪声
        x = torch.randn_like(s)
        for n in reversed(range(0, self.diff.timesteps)):
            t = torch.full((s.shape[0], ), n, device=self.device, dtype=torch.long)
            # x:(batch, dim), s:(batch, dim), t:(batch, ), n:1~timesteps
            x = self.p_sample_with_guidance(self.forward, self.forward_uncon, x, s, t, n)
        
        return x
    
    @torch.no_grad()
    def p_sample_with_guidance(self, model_forward, model_forward_uncon, x, s, t, t_index):

        # guidance, (batch, dim)
        # classifier-free的设计，有条件和无条件加权
        x_start = (1 + self.w) * model_forward(x, s, t) - self.w * model_forward_uncon(x, t)
        x_t = x

        # model_mean = (batch,1) x (batch, dim) + (batch,1) x (batch, dim) = (batch, dim)
        # 计算高斯分别的均值
        model_mean = (
            extract(self.diff.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.diff.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.diff.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        # 每个质心需要的最多和最少数据点数
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        # 是否是第一批数据
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        # 聚类中心点
        self.centroids = []
        
        # 序列到簇的映射
        self.seq2cluster = None
        # 簇到序列的映射
        self.cluster2sequences = defaultdict(list)  


    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        # print(" cluster train iterations:", niter)
        # faiss向量检索工具，这里创建一个K means对象
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        # 是否输出详细日志
        clus.verbose = verbose
        # 聚类迭代次数
        clus.niter = niter
        # 随机重启次数（避免局部最优）
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        # 创建标准的gpu资源管理器
        res = faiss.StandardGpuResources()
        # 禁用临时内存
        res.noTempMemory()
        # 配置gpu索引参数
        cfg = faiss.GpuIndexFlatConfig()
        # 不适用float16半精度而是使用全精度
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        # 返回举例最小的聚类中心
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        # 返回聚类器和索引对象
        return clus, index

    def train(self, x, sequence_ids):
        # train to get centroids
        # 只有输入数据点大于聚类数量才会训练
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        # 聚类中心->向量转数组->一维数组到二维数组（例如10个簇，每个质心是128维向量）
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        # 归一化质心向量
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

        # 搜索最近的簇，返回距离质心的距离和所属的簇
        # x是要查询的点，1代表最近邻搜最近的1个
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # 保留序列到簇的映射
        self.seq2cluster = seq2cluster  

        for seq_id, cluster_id in zip(sequence_ids, seq2cluster):
            self.cluster2sequences[cluster_id].append(seq_id)

    def query(self, x):
        # 输入x为n_sample*hidden size大小
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        # 如果 I = [[2], [0], [1], [2], [0]]
        # 那么 seq2cluster = [2, 0, 1, 2, 0]
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        # 返回簇的index和质心向量
        return seq2cluster, self.centroids[seq2cluster]

# 输入的是用户的交互序列，输出是包含物品语义信息，位置信息，上下文信息的嵌入向量
class SASRecModel(nn.Module):
    def __init__(self, item_embeddings, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = item_embeddings
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def item_embedding(self, sequence):
        return self.item_embeddings(sequence)
    
    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda == 0:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
