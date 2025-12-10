import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from typing import Dict

from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file, if you want to use it.
word_emb_path = '/your path/glove.840B.300d.txt'
assert (word_emb_path is not None)

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent

sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')

data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath('MOSEI')}

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {"elu": nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,"leakyrelu": nn.LeakyReLU,
                   "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU, "tanh": nn.Tanh}

output_dim_dict: dict[str, int] = {'mosi': 1, 'mosei_senti': 1}

criterion_dict = {'mosi': 'L1Loss', 'mosei':'L1Loss'}


def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('--f', default='', type=str)

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    parser.add_argument('iid_setting',action='store_false',help="test using iid dataset")
    parser.add_argument('ood_setting',action='store_false',help="test using ood dataset")

    parser.add_argument('seven_class',action='store_false',help="seven classification")
    parser.add_argument('--npy_path',type=str,default='npy_folder',help='path for storing the Kmeans center')
    parser.add_argument('--npy_selection',choices=['bert_0','bert_mean'],default="bert_mean",help="text npy file")
    # Dropouts
    parser.add_argument('--dropout_a',type=float,default=0.14684748135792808,help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v',type=float,default=0.31967333598642866,help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj',type=float,default=0.14420097606804064,help='dropout of projection layer')

    parser.add_argument('--text_encoder',choices=['bert','roberta'],default="roberta",help="text_encoder") ###

    # Kmeans size
    parser.add_argument('--use_kmean',action='store_false',help='use Kmean initialization')
    parser.add_argument('--kmean_size',type=int,default=200,help='size of confounder dictionary')
    parser.add_argument('--audio_kmean_size',type=int,default=50,help='size of confounder dictionary')
    parser.add_argument('--text_kmean_size',type=int,default=200,help='size of confounder dictionary')
    parser.add_argument('--vision_kmean_size',type=int,default=50,help='size of confounder dictionary')

    # Debias模块
    parser.add_argument('--whether_debias_unimodal',action='store_false',help='whether to use debias module in unimodal')
    parser.add_argument('--whether_debias_audio',action='store_false',help='whether to debias audio')
    parser.add_argument('--whether_debias_text',action='store_false',help='whether to debias text')
    parser.add_argument('--whether_debias_vision',action='store_false',help='whether to debias vision')
    parser.add_argument('--audio_debias_layers',type=int,default=4,help="layers in debias self attention")
    parser.add_argument('--vision_debias_layers',type=int,default=4,help="layers in debias self attention")
    parser.add_argument('--text_debias_layers',type=int,default=3,help="layers in debias self attention, mosi for 3, mosei for 5")
    parser.add_argument('--attn_dropout_debias', type=float, default=0.17307476073796302, help="dropout in single modal self attention")

    #单模态Mlp模块
    parser.add_argument('--audio_mlp_hidden_size',type=int,default=256,help="the hidden size of mlp in audio projection, mosi for 256, mosei for 32")
    parser.add_argument('--vision_mlp_hidden_size',type=int,default=32,help="the hidden size of mlp in vision projection, mosi for 32, mosei for 256")   
    parser.add_argument('--text_mlp_hidden_size',type=int,default=128,help="the hidden size of mlp in text projection, mosi for 128, mosei for 512")   

    # CounterFactual模块
    parser.add_argument('--whether_use_counterfactual',action='store_true',help="whether to use counterfactual")
    parser.add_argument('--whether_use_counterfactual_ta',action='store_true',help="whether to use counterfactual ta")
    parser.add_argument('--whether_use_counterfactual_tv',action='store_true',help="whether to use counterfactual tv")
    parser.add_argument('--counterfactual_attention_type',choices=['random', 'uniform', 'reversed','shuffle'],default="reversed",help="the type of counterfactual attention")                                                                                                    
    parser.add_argument('--num_layers_counterfactual_attention',type=int,default=3,help="layers in counterfactual attention, mosi for 3, mosei for 6")

    #自注意力部分
    parser.add_argument('--model_dim_self',type=int,default=32,help="dim in single modal self attention")
    parser.add_argument('--num_heads_self',type=int,default=2,help="heads in single modal self attention")
    parser.add_argument('--num_layers_self', type=int,default=1,help="layers in self attention, mosi for 3, mosei for 2")
    parser.add_argument('--attn_dropout_self', type=float,default=0.34370241471389507,help="dropout in single modal self attention")

    #跨模态注意力部分
    parser.add_argument('--model_dim_cross',type=int,default=96, help="dim in single modal cross attention, like cross_tv/ta")
    parser.add_argument('--num_heads_cross',type=int,default=6, help="heads in single modal cross attention")
    parser.add_argument('--num_layers_cross',type=int,default=6, help="layers in cross attention, mosi for 3, mosei for 2")
    parser.add_argument('--attn_dropout_cross',type=float,default=0.3097204834149764 , help="dropout in single modal cross attention")

    # APF
    # ===== APF 开关与维度 =====
    parser.add_argument('--apf_enable', action='store_true', default=True, help='是否启用 APF 模块')
    parser.add_argument('--apf_d_model', type=int, default=-1, help='APF 融合输出维；-1 则用 model_dim_cross')
    parser.add_argument('--apf_z_dim', type=int, default=32, help='共享潜空间维度(KL 计算维度)，常用 16/32/64')

    # ===== VarHead 稳定性 =====
    parser.add_argument('--apf_use_LN_before_varhead', type=bool, default=True, help='VarHeadZ 前是否接 LayerNorm')
    parser.add_argument('--apf_varhead_hidden_ratio', type=float, default=1.0, help='VarHeadZ 隐层宽度=ratio*D_in, 1.0 表示等宽')

    # ===== KL 数值稳定 =====
    parser.add_argument('--apf_clamp_min', type=float, default=-8.0, help='logvar 下界（防止方差过小）')
    parser.add_argument('--apf_clamp_max', type=float, default=2.0, help='logvar 上界（防止方差过大）')
    parser.add_argument('--apf_eps', type=float, default=1e-6, help='数值稳定用 epsilon')

    # ===== EMA & 温度 =====
    parser.add_argument('--apf_beta', type=float, default=0.9907063660842929, help='EMA 动量（归一化分母）；抖动大可调到 0.995')
    parser.add_argument('--apf_tau_start', type=float, default=0.694334692053831, help='温度 τ 起始值（训练早期更软）')
    parser.add_argument('--apf_tau_end', type=float, default=0.449860079116019, help='温度 τ 结束值（训练后期更硬）')
    parser.add_argument('--apf_tau_schedule_frac', type=float, default=0.49987667807793296, help='τ 线性调度占比（训练进度的前多少）')
    parser.add_argument('--apf_separate_tau', action='store_true', default=True, help='是否为视觉/语音分别设置 τ')
    parser.add_argument('--apf_tau_v', type=float, default=0.7646510968194998, help='视觉门控 τ（仅在 separate_tau=True 时生效）')
    parser.add_argument('--apf_tau_a', type=float, default=0.7534636427015075, help='语音门控 τ（仅在 separate_tau=True 时生效）')

    # ===== 文本 768→d_model 投影 =====
    parser.add_argument('--apf_use_proj_LN_in', type=bool, default=True, help='文本投影前是否做 LayerNorm')
    parser.add_argument('--apf_use_proj_LN_out', type=bool, default=True, help='文本投影后是否做 LayerNorm')
    parser.add_argument('--apf_proj_init', type=str, choices=['orthogonal','xavier'], default='orthogonal', help='文本降维线性层初始化方式')
    parser.add_argument('--apf_proj_freeze_steps', type=int, default=0, help='训练早期冻结文本降维层的步数（0 表示不冻结）')

    # ===== 可选正则 =====
    parser.add_argument('--apf_align_loss_weight', type=float, default=0.09473876632532414, help='潜空间一致性正则权重（建议 0.05~0.1；0 关闭）')
    parser.add_argument('--apf_prior_kl_weight', type=float, default=0.059850237331625075, help='对 N(0,I) 的先验 KL 正则权重（默认 0 关闭）')

    # ===== 日志/调试 =====
    parser.add_argument('--apf_log_intermediate', action='store_true', default=False, help='记录 w/K 的直方图/热力图用于调参')


    # infonce
    # parser.add_argument('--embed_dropout_infonce', type=float, default=0.1, help="infonce emb dropout")
    # parser.add_argument('--embed_dropout_infonce_cross', type=float, default=0.1, help="infonce emb dropout after cross attention")
    #
    # fusion method
    # parser.add_argument('--fusion',type=str,default='sum',help='the fusion method used in final fusion')#[sum,fusion]

    # 最后的全模态自注意力
    # parser.add_argument('--model_dim_final',type=int,default=30,help="model dim in final self attention")
    # parser.add_argument('--attn_dropout_final',type=float,default=0.1,help="dropout in final self attention")
    # parser.add_argument('--num_layers_final',type=int,default=6,help="layers in final self attention")
    # parser.add_argument('--num_heads_final',type=int,default=6,help="heads in final self attention")


    # transformer dropout
    # parser.add_argument('--attn_dropout', type=float, default=0.1,help='attention dropout')
    # parser.add_argument('--attn_dropout_a', type=float, default=0.0,help='attention dropout (for audio)')
    # parser.add_argument('--attn_dropout_v', type=float, default=0.0,help='attention dropout (for visual)')

    parser.add_argument('--relu_dropout', type=float, default=0.17020559392165868,help='relu dropout')
    parser.add_argument('--res_dropout', type=float, default=0.19486421193535425,help='residual block dropout')
    parser.add_argument('--attn_mask', action='store_false', help='use attention mask for Transformer (default: true)')
    # parser.add_argument('--out_dropout', type=float, default=0.0,help='output layer dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.17940229327406354,help='embedding dropout')


    parser.add_argument('--vonly',action='store_true',help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly',action='store_true',help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly',action='store_true',help='use the crossmodal fusion into l (default: False)')

    # parser.add_argument('--layers', type=int, default=5,help='number of layers in the network (default: 5)')
    # parser.add_argument('--num_heads', type=int, default=5,help='number of heads for the transformer network (default: 5)')

    # Architecture
    # parser.add_argument('--n_tv',type=int,default=0,help='number of V-T transformer  (default: 0)')
    # parser.add_argument('--n_ta',type=int,default=1,help='number of A-T transformer (default: 1)')

    parser.add_argument('--multiseed',action='store_true',help='training using multiple seed')
    parser.add_argument('--contrast',action='store_false',help='using contrast learning')
    parser.add_argument('--add_va',action='store_false',help='if add va MMILB module')  # 是否采用VA互信息最大化
    parser.add_argument('--n_layer',type=int,default=1,help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers',type=int,default=2,help='number of layers in CPC NCE estimator (default: 1)')
    parser.add_argument('--d_vh',type=int,default=32,help='hidden size in visual rnn, mosi for 128, mosei for 32')
    parser.add_argument('--d_ah',type=int,default=64,help='hidden size in acoustic rnn, mosi for 64, mosei for 32')
    parser.add_argument('--d_vout',type=int,default=32,help='output size in visual rnn')
    parser.add_argument('--d_aout',type=int,default=32,help='output size in acoustic rnn')
    parser.add_argument('--bidirectional',action='store_true',help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=256,help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,help='dimension of pretrained model output')
    parser.add_argument('--mem_size',type=int,default=6, help='Memory size, mosi for 6, mosei for 3, you can try 256')

    # Activations
    parser.add_argument('--mmilb_mid_activation',type=str,default='Tanh',help='Activation layer type in the middle of all MMILB modules')
    parser.add_argument('--mmilb_last_activation',type=str,default='Tanh',help='Activation layer type at the end of all MMILB modules')
    parser.add_argument('--cpc_activation',type=str,default='Tanh',help='Activation layer type in all CPC modules')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 16)')
    parser.add_argument('--clip', type=float, default=0.8538198404179991, help='gradient clip value, mosi for 4.0, mosei for 1.0')
    parser.add_argument('--lr_main', type=float, default=4.5248184332682105e-05, help='initial learning rate for main model parameters (default: 1e-4)')
    parser.add_argument('--lr_bert', type=float, default=2.8920529520056032e-05, help='initial learning rate for bert parameters (default: 5e-6)')
    parser.add_argument('--lr_mmilb', type=float, default=3.6029962802800814e-05,  help='initial learning rate for mmilb parameters (default: 5e-5)')

    parser.add_argument('--alpha', type=float, default=0.05, help='weight for CPC NCE estimation item, mosi for 0.05, mosei for 0.4')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for lld item, mosi for 0.1, mosei for 0.25')
    parser.add_argument('--eta',type=float,default=0.3,help='weight for counterfactual, mosi for 0.3, mosei for 0.15')
    
    parser.add_argument('--weight_decay_main',type=float,default=8.952409738732232e-06,help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert',type=float,default=7.884963390699172e-08,help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club',type=float,default=0.0000709191983951567,help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim',type=str,default='Adam',help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs',type=int,default=40,help='number of epochs (default: 40)')
    parser.add_argument('--when',type=int,default=16,help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience',type=int,default=5,help='when to stop training if best never change')
    parser.add_argument('--update_batch',type=int,default=1,help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval',type=int,default=1,help='frequency of result logging (default: 100)')
    parser.add_argument('--seed',type=int,default=1111,help='random seed')
    args = parser.parse_args()

    valid_partial_mode = args.lonly + args.vonly + args.aonly
    # 默认全为False的话 valid_partial_mode==0

    if valid_partial_mode == 0:
        args.lonly = args.vonly = args.aonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")

    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train',args=None):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        self.args=args
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32,args=None):
    config = Config(data=dataset, mode=mode,args=args)

    config.dataset = dataset
    config.batch_size = batch_size

    return config
