import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from utils.convert import change_to_classify
from transformers import XLNetTokenizer

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import MOSI, MOSEI,UR_FUNNY,PAD, UNK


class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()

        self.data, self.word2id, _ = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.word2id = self.word2id
        # config.pretrained_emb = self.pretrained_emb

    @property
    def tva_dim(self):
        t_dim = 768
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True, configprint=False):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print(config.mode)

    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    # 只用初始化一遍即可！
    # 根据 hp.text_encoder 选择 tokenizer
    if not configprint:
        logging.set_verbosity_error()

    if getattr(hp, 'text_encoder', 'roberta') == 'bert':
        bert_tokenizer = AutoTokenizer.from_pretrained("LLMs/bert-base-uncased", local_files_only=True)
        selected_tokenizer = bert_tokenizer
    else:
        roberta_tokenizer = AutoTokenizer.from_pretrained("LLMs/twitter-roberta-base-sentiment", local_files_only=True)
        selected_tokenizer = roberta_tokenizer

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            #print(sample)
            if len(sample[0]) > 4:  # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:  # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            #print(type(sample), type(sample[1]))
            # labels.append(torch.from_numpy(sample[1]))
            # 确保 sample[1] 是 numpy.ndarray 类型
            if isinstance(sample[1], np.ndarray):
                labels.append(torch.from_numpy(sample[1]))
            else:
                labels.append(torch.from_numpy(np.array([sample[1]])))  # 转换为 numpy 数组并包装成 (1,)
            ids.append(sample[2])


        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)

        #print(labels.shape)
        if labels.dim() == 1:
            labels = labels[:, None]
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]
        #print(labels.shape)
        # Rewrite this
        def pad_to_target(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        v_masks = pad_to_target([torch.zeros(torch.FloatTensor(sample[0][1]).size(0)) for sample in batch], target_len=vlens.max().item(),padding_value=1)
        a_masks = pad_to_target([torch.zeros(torch.FloatTensor(sample[0][2]).size(0)) for sample in batch], target_len=alens.max().item(),padding_value=1)
        #print(a_masks)


        sentences = pad_to_target([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        #print(sentences)
        visual = pad_to_target([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_to_target([torch.FloatTensor(sample[0][2]) for sample in batch], target_len=alens.max().item())

        ## BERT-based features input prep

        # SENT_LEN = min(sentences.size(0),50)
        SENT_LEN = 50
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_sent = selected_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')

            if 'token_type_ids' not in encoded_sent:
                encoded_sent['token_type_ids'] = [0] * len(encoded_sent['input_ids'])

            # encoded_xlnet_sent = xlnet_tokenizer.encode_plus(
            #     text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')

            # print(sample[0][3])
            # print(encoded_bert_sent)
            # print(encoded_roberta_sent)
            # print(encoded_xlnet_sent)

            bert_details.append(encoded_sent)


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        #print(bert_sentences)
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        #print(bert_sentence_types)
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
        #print(bert_sentence_att_mask)

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        #print(lengths)
        if (vlens <= 0).sum() > 0:
            #vlens[np.where(vlens == 0)] = 1
            vlens[vlens <= 0] = 1


        # labels_classify = change_to_classify(y=labels, output_size=2)
        # labels_classify = change_to_classify(y=labels, output_size=7)   如果不做端到端优化，根本不需要！ 做端到端优化，取消注释
        # 并返回labels_classify即可！


        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids, v_masks, a_masks

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        persistent_workers=0)

    return data_loader
