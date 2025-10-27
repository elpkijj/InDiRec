import copy
import random
import torch

from torch.utils.data import Dataset
from utils import generate_rating_matrix_valid, generate_rating_matrix_test

# 生成子序列，不知道为什么这么做
def D(i_file,o_file,max_len):

    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    # training, validation, and testing
    max_save_len=max_len+3
    # save
    max_keep_len=max_len+2
    for d_ in data:
        # 用户和物品列表
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        # 取出最后一个item，eval("3+8")=>11
        item[-1]=str(eval(item[-1]))
        # 确保u_i是列表
        aug_d.setdefault(u_i, [])
        start=0
        j=3
        # 当item列表的长度比max_len+3更大时
        if len(item)>max_save_len:
            # training, validation, and testing
            while start<len(item)-max_keep_len:
                j=start+4
                while j<len(item):
                    # 加入0到4
                    if start<1 and j-start<max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j+=1
                    else:
                        # 加入start到start到max_save_len
                        aug_d[u_i].append(item[start:start+max_save_len])
                        break
                start+=1
        else:
            while j<len(item):
                # 当列表内item的数量比j更大的时候，直接在aug_d[u_i]加入其序列中的item
                # 第一次是0到4，第二次是0到5
                aug_d[u_i].append(item[start:j+1])
                j+=1
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")

def D_random(i_file, o_file, max_len):

    with open(i_file, "r+") as fr:
        data = fr.readlines()
    
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 3

    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))  # Ensure the last item is a valid number
        aug_d.setdefault(u_i, [])
        
        start = 0
        j = 3
        # 上面和D函数的操作基本一致
        # ---------------------
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - (max_len + 2):
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        # 与D函数不同的地方：这里是对子序列进行不放回采样，相当于洗牌——数据增强
                        random_sample = random.sample(item[start:j], len(item[start:j]))
                        aug_d[u_i].append(random_sample)
                        j += 1
                    else:
                        random_sample = random.sample(item[start:start + max_save_len], len(item[start:start + max_save_len]))
                        aug_d[u_i].append(random_sample)
                        break
                start += 1
        else:
            while j < len(item):
                random_sample = random.sample(item[start:j + 1], len(item[start:j + 1]))
                aug_d[u_i].append(random_sample)
                j += 1

    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")

def get_seqs_and_matrixes(type, data_file):
    user_seq = []
    user_id = []
    item_set = set()
    
    with open(data_file, "r") as fr:
        for line in fr:
            parts = line.strip().split()
            user = int(parts[0])
            items = list(map(int, parts[1:]))
            # 用户，用户序列，item集合
            user_id.append(user)
            user_seq.append(items)
            item_set.update(items)
    
    max_item = max(item_set)
    num_users = len(user_id)
    num_items = max_item + 2
    
    if type == "training":
        return user_id, user_seq
    elif type == "rating":
        # 调用utils，生成评分矩阵（交互矩阵，有交互为1，没有交互为0）
        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
        return user_id, user_seq, max_item, valid_rating_matrix, test_rating_matrix
    else:
        raise NotImplementedError


class DatasetForInDiRec(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        # 这里的args是由parser传参进去的
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length


    def __getitem__(self, index):
        user_id = index
        item_seq = self.user_seq[user_id]

        if self.data_type == "train":
            # 看后面这些分别是怎么用的
            # 输入序列：从开始到倒数第三个（不包括倒数第三个）
            input_seq = item_seq[ : -3]
            # 从第一个到倒数第二个
            target_pos = item_seq[1 : -2]
            # 倒数第三个
            answer = item_seq[-3]

        elif self.data_type == "valid":
            input_seq = item_seq[:-2]
            target_pos = item_seq[1:-1]
            answer = [item_seq[-2]]
        
        else:
            #  Robustness to Noisy Data.
            # 数据增强方式：插入序列（其实可以优化，插入和item类似的序列）
            item_seq_with_noise = self._add_noise_for_robustness(item_seq)
            input_seq = item_seq_with_noise[:-1]
            target_pos = item_seq_with_noise[1:]
            answer = [item_seq_with_noise[-1]]

        # 返回pad后的sequence
        cur_rec_tensors = self._data_construction(user_id, input_seq, target_pos, answer)

        return cur_rec_tensors


    # padding and to tensor
    def _data_construction(self, user_id, input_seq, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        # 拷贝输入序列
        copied_input_seq = copy.deepcopy(input_seq)
        input_seq_len = len(copied_input_seq)
        # pad的长度为最大长度减去序列长度[1,2]=>[0,0,1,2],[1]=>[0,0,0,1]
        pad_len = self.max_len - input_seq_len
        copied_input_seq =[0] * pad_len+copied_input_seq
        copied_input_seq=copied_input_seq[-self.max_len:]

        # padding
        # 填充
        target_pos =  [0] * pad_len+target_pos
        # 截断
        target_pos = target_pos[-self.max_len:]

        assert len(target_pos) == self.max_len
        assert len(copied_input_seq) == self.max_len

        # to tensor
        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_seq, dtype=torch.long),
            torch.tensor(input_seq_len, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_for_robustness(self, items):
        if self.args.noise_ratio == 0:
            return items
        # 拷贝一份数据
        copied_sequence = copy.deepcopy(items)
        # 插入噪声的位置数量
        insert_nums = int(self.args.noise_ratio * len(copied_sequence))

        # 随机选择插入噪声的idx
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                # 如果index是要被扰动位置的index，那么item_id由原来的变成是1到size-2范围内的随机数
                item_id = random.randint(1, self.args.item_size - 2)
                # while保证随机生成的新的item_id不能是原序列中的
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __len__(self):

        return len(self.user_seq)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader, SequentialSampler

    def get_args():
        parser = argparse.ArgumentParser()
        # system args
        parser.add_argument("--data_dir", default="./datasets/", type=str)
        parser.add_argument("--output_dir", default="output", type=str)
        parser.add_argument("--data_name", default="Beauty", type=str)


        parser.add_argument("--max_seq_length", type=int, default=50, 
                            help="max sequence length")

        parser.add_argument("--batch_size", type=int, default=4, 
                            help="number of batch_size")
        parser.add_argument("--epochs", type=int, default=300, 
                            help="number of epochs")
        parser.add_argument("--log_freq", type=int, default=1, 
                            help="per epoch print res")
        parser.add_argument("--seed", default=2022, type=int)

        return parser.parse_args()

    args = get_args()

    D("./datasets/Beauty.txt","./datasets/Beauty_s.txt",10)

    args.segmented_file = args.data_dir + args.data_name + "_s.txt"

    _,train_seq = get_seqs_and_matrixes("training", args.segmented_file)

    # 返回数据集：pad后的seq，用于训练
    cluster_dataset = DatasetForInDiRec(args, train_seq, data_type="train")
    # 按照数据集原始顺序生成索引
    cluster_sampler = SequentialSampler(cluster_dataset)
    # 按照批次采样数据
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    for batch_idx, data in enumerate(cluster_dataloader):
        print(f"Batch {batch_idx + 1}: {data}")
