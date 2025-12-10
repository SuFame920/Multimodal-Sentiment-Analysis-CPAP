# ====================== 放在所有 import 之前（必须置顶） ======================
import os

# 强制 PyTorch 的 SDPA 注意力使用 “math” 内核（纯 matmul+softmax 路径），
# 禁用 Flash / memory-efficient 内核，避免 Windows/CUDA 某些组合下的底层崩溃。
# ✅ 数值等价，几乎不影响精度；可能略慢，但更稳定。
os.environ["PYTORCH_FORCE_SDP_KERNEL"] = "math"

# 让 CUDA 调用改为“同步”模式，错误能在出错的那一行立刻抛出，便于定位。
# ❗会降低吞吐：调试期建议开，稳定后可注释掉以恢复异步加速。
os.environ["CUDA_LAUNCH_BLOCKING"]     = "1"

# 一旦发生异常，附带 C++ 栈，便于定位底层 native 调用源头。
# 只在异常路径生效，对性能影响可以忽略。
os.environ["TORCH_SHOW_CPP_STACKTRACES"]= "1"

# 彻底关闭 FX 栈格式化（有两个变量名历史别名，都设为 0 以防万一）。
# 这能显著减少频繁堆栈抓取/格式化的开销（训练会更快），
# 代价是错误回溯信息没那么“花哨”。不影响数值和精度。
os.environ["PYTORCH_SHOW_FX_TRACEBACK"]= "0"
os.environ["TORCH_SHOW_FX_TRACEBACK"]  = "0"

# 关闭 HuggingFace tokenizer 的多线程并行提示与线程争用（更干净、稳定）。
os.environ["TOKENIZERS_PARALLELISM"]   = "false"

# ---------------------------------------------------------------------------

import faulthandler
faulthandler.enable()  # 启用 Python 原生崩溃时的 C 栈打印（对性能无影响，定位硬崩溃很有用）。

import torch
# 关闭 autograd 的异常检测（detect_anomaly 会为每个算子记录追踪栈，严重拖慢训练）。
# ✅ 不影响数值与精度；调试爆 NaN/Inf 时才需要临时开启。
torch.autograd.set_detect_anomaly(False)

# ---- 强力禁用 PyTorch FX 的回溯格式化：将其变成 no-op（提高训练速度） ----
try:
    import torch.fx.traceback as _fx_tb
    _fx_tb.format_stack     = lambda *a, **k: []  # 不再收集/格式化 Python 栈
    _fx_tb.format_exception = lambda *a, **k: []  # 不再格式化异常
    if hasattr(_fx_tb, "extract_stack"):
        _fx_tb.extract_stack = lambda *a, **k: [] # 不再抽取调用栈
except Exception:
    # 某些版本/环境下 torch.fx.traceback 可能不存在；失败就静默略过即可。
    pass
# ---------------------------------------------------------------------------



import random
import numpy as np
import torch

import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


if __name__ == '__main__':
    args = get_args()
    print("1. Current device:", args.device)
    dataset = str.lower(args.dataset.strip())
    set_seed(args.seed)
    print("\n2. Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size,args=args)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size,args=args)
    test_config = get_config(dataset, mode='test', batch_size=args.batch_size,args=args)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, shuffle=True, configprint=True)
    print('\n3. Training data loaded!')
    valid_loader = get_loader(args, valid_config, shuffle=False, configprint=False)
    print('\n4. Validation data loaded!')
    test_loader = get_loader(args, test_config, shuffle=False, configprint=False)
    print('\n5. Test data loaded!')
    print('\n6. Finish loading the data....')

    #torch.autograd.set_detect_anomaly(True) ？？？ why？？？

    # addintional appending
    args.word2id = train_config.word2id


    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when

    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MAELoss')

    import time
    result = {
        'mae': [],
        'cor': [],
        'acc7': [],
        'acc2_1': [],
        'acc2_2': [],
        'f1_1': [],
        'f1_2': [],
        'weight_name': [],
        'time': [],
        'beta':[],
        'alpha': [],
        'args':[]
    }
    for i in range(1):
        beta = args.beta
        alpha = args.alpha
        print("\n7. start", i)
        print("\n8. args:", args)
        start_time = time.time()
        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=test_loader, is_train=True)
        to_exl, wight_name = solver.train_and_eval()

        cost_time = time.time() - start_time
        result['mae'].append(to_exl[0])
        result['cor'].append(to_exl[1])
        result['acc7'].append(to_exl[2])
        result['acc2_1'].append(to_exl[3])
        result['acc2_2'].append(to_exl[4])
        result['f1_1'].append(to_exl[5])
        result['f1_2'].append(to_exl[6])
        result['weight_name'].append(wight_name)
        result['time'].append(cost_time)
        result['beta'].append(beta)
        result['alpha'].append(alpha)
        result['args'].append(args)
        print("\n9. result:", result)
        print('*'*30)

    data_frame = pd.DataFrame(
        data={'mae': result['mae'], 'cor': result['cor'], 
              'acc7': result['acc7'], 
              'acc2_1': result['acc2_1'], 'acc2_2': result['acc2_2'], 
              'f1_1': result['f1_1'], 'f1_2': result['f1_2'],
              'time': result['time'], 'weight_name': result['weight_name'], 
              'beta': result['beta'], 'alpha': result['alpha'],
              'args':result['args']},
        index=range(1)
    )

    now_time = time.strftime("_%m%d_%H%M", time.localtime())
    if "mosi" in args.dataset:
        path = "pre_trained_best_models_mosi/" + args.dataset + now_time + "_result.csv"
    else:
        path = "pre_trained_best_models_mosei/" + args.dataset + now_time + "_result.csv"
    print("\n11. path:", path)
    data_frame.to_csv(path)

