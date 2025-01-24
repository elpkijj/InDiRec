import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) # 取根据索引t从a的最后一个维度取值，gather(dim, index) 会在指定的维度 dim 上提取 index 所指定的元素
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device) # reshape将原来的向量转化为想要的形状，总元素不变
    #len()返回变量的维度，这里是2，*的作用是将（1，）作为一个整体传入reshape，不然会报错，因为reshape要两个int，而不是tuple

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # 使用 scipy.sparse.csr_matrix 构建一个稀疏矩阵。矩阵的形状为 (num_users, num_items)，
    # 即每行对应一个用户，每列对应一个物品，矩阵中的非零元素表示用户与物品有过交互（评分为 1）。
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_padded_sequences_tensor(batch_size, padded_seq_len, max_item, input_seq_len):
    """
    生成一个大小为 batch_size x seq_len 的物品序列，序列使用0填充到 seq_len，并转换为 PyTorch Tensor。
    
    Args:
        batch_size (int): 批量大小。
        seq_len (int): 固定的序列长度（包括填充）。
        max_item (int): item_id 的最大值。
    
    Returns:
        torch.Tensor: 形状为 (batch_size, seq_len) 的物品序列张量。
    """

    sequences = []
    
    for index in range(batch_size):
        # 随机生成序列的实际长度，范围为 [10, 20]
        # actual_length = np.random.randint(5, 21)
        actual_length = input_seq_len[index]
        # 随机生成物品 ID，范围为 [1, max_item]
        item_ids = np.random.randint(1, max_item - 2, size=actual_length)
        # 生成左侧填充0的序列
        padded_sequence = np.zeros(padded_seq_len, dtype=int)
        padded_sequence[-actual_length:] = item_ids
        sequences.append(padded_sequence)
    
    # 转换为 numpy 数组后，再转为 PyTorch Tensor
    sequences_array = np.array(sequences)
    sequences_tensor = torch.tensor(sequences_array, dtype=torch.long)  # 使用 long 类型存储整数
    
    return sequences_tensor


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            设置训练停止trigger
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f"Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score