import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from thop import profile, clever_format
from utils.eval_metrics import *
from utils.tools import *
from utils import gen_Kmeans_center
from model import CPAP
import csv
from utils.gen_Kmeans_center import gen_npy
from modules.livePlotter import LivePlotter

#torch.autograd.set_detect_anomaly(True)

# >>> APF monitor: 画图
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.plotter = LivePlotter() # 训练时，绘制loss曲线

        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.eta = hp.eta
        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = CPAP(hp)
        
        self.device = self.hp.device
        self.model = model.to(self.device)   

        # criterion
        if self.hp.dataset == "mosei_ood" or self.hp.dataset == "mosi_ood":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  
            self.criterion = criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            main_param = []
            bert_param = []
            mmilb_param = []

            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else:
                        main_param.append(p)

                for p in (mmilb_param + main_param):
                    if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)

        self.optimizer_mmilb = getattr(torch.optim, self.hp.optim)(mmilb_param, lr=self.hp.lr_mmilb, weight_decay=hp.weight_decay_club)

        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(optimizer_main_group)

        # —— 禁用 foreach/fused（Windows 下最稳）——
        for opt in [self.optimizer_mmilb, self.optimizer_main]:
            for g in opt.param_groups:
                g["foreach"] = False
                if "fused" in g:  # 某些版本没有 fused 键
                    g["fused"] = False

        # self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5, verbose=True)
        # self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)
        self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5)

        # >>> APF monitor: 全局 step/总步，用于 τ 调度
        steps_per_epoch = max(1, self.hp.n_train // self.hp.batch_size)
        self.total_steps = int(self.hp.num_epochs * steps_per_epoch)
        self.global_step = 0

        # >>> APF monitor: 历史曲线（每 epoch 一个点）
        self.apf_hist = {
            'train': {'wv': [], 'wa': [], 'Kv': [], 'Ka': []},
            'dev':   {'wv': [], 'wa': [], 'Kv': [], 'Ka': []},
            'test':  {'wv': [], 'wa': [], 'Kv': [], 'Ka': []},
        }
        self.apf_plot_dir = getattr(self.hp, 'apf_plot_dir', 'apf_plots')
        os.makedirs(self.apf_plot_dir, exist_ok=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main

        scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = self.hp.mem_size
        tanh = torch.nn.Tanh()
        relu = torch.nn.ReLU()

        def _check_grads(model):
            bad = []
            devs, dtypes = set(), set()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                devs.add(p.grad.device)
                dtypes.add(p.grad.dtype)
                # 抓 NaN/Inf
                if not torch.isfinite(p.grad).all():
                    bad.append(name)
            if len(devs) > 1:
                print("[GRAD CHECK] multiple devices for grads:", devs)
            if len(dtypes) > 1:
                print("[GRAD CHECK] multiple dtypes for grads:", dtypes)
            if bad:
                print("[GRAD CHECK] non-finite grads in:", bad[:10], ("... (+more)" if len(bad)>10 else ""))
            return devs, dtypes, bad



        def train(model, optimizer, criterion, stage=1):
            epoch_loss = 0

            all_t_mean, all_v_mean, all_a_mean = [], [], []

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            eff_loss=0.0
            ba_loss = 0.0
            task_loss=0.0
            start_time = time.time()

            left_batch = self.update_batch

            mem_pos_tv, mem_neg_tv, mem_pos_ta, mem_neg_ta = [], [], [], []
            if self.hp.add_va:
                mem_pos_va, mem_neg_va = [], []

            _loss, _loss_a, _loss_v = 0, 0, 0

            # >>> APF monitor: 训练期的监控累计器（打印用的批均值）
            apf_keys = ["w_v_mean","w_a_mean","frac_wv_0","frac_wv_1","frac_wa_0","frac_wa_1","Kv_mean","Ka_mean","tau_now","emaKv","emaKa"]
            apf_sums = {k: 0.0 for k in apf_keys}
            apf_cnt = 0
            # >>> APF monitor: 严格全局有效 token 级加权（本 epoch）
            apf_epoch = {"n_valid":0, "sum_wv":0.0, "sum_wa":0.0, "sum_Kv":0.0, "sum_Ka":0.0}

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids, v_mask, a_mask = batch_data

                # lld, nce, preds, pn_dic, H, counterfactual_preds = model(
                # text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask, y, None, v_mask, a_mask)

                # for mosei we only use 50% dataset in stage 1, that is enough for training! But you can increase it steadily for better performance.
                if "mosei" in self.hp.dataset:
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break

                model.zero_grad()
                # with torch.cuda.device(0):
                #     text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask, v_mask, a_mask = \
                #         text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                #             bert_sent_type.cuda(), bert_sent_mask.cuda(), v_mask.cuda(), a_mask.cuda()
                text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask, v_mask, a_mask = \
                    text.to(self.device), visual.to(self.device), audio.to(self.device), y.to(self.device), l.to(self.device), bert_sent.to(self.device), \
                        bert_sent_type.to(self.device), bert_sent_mask.to(self.device), v_mask.to(self.device), a_mask.to(self.device)

                batch_size = y.size(0)

                if stage == 0:  # Neg-lld, 0 for prepare, 1 for training.
                    y = None
                    mem = None
                elif stage == 1 and i_batch >= mem_size:  # TASK+BA+CPC  memory
                    mem = {'tv': {'pos': mem_pos_tv, 'neg': mem_neg_tv},
                           'ta': {'pos': mem_pos_ta, 'neg': mem_neg_ta},
                           'va': {'pos': mem_pos_va, 'neg': mem_neg_va} if self.hp.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}
                    
                ''' Flops 浮点运算次数(G), Params 参数量(M)
                visual = torch.randn((500,8,20)).cuda()
                audio = torch.randn((375,8,5)).cuda()
                flops, params = profile(model, inputs=(text, visual, audio, vlens, alens,
                                bert_sent, bert_sent_type, bert_sent_mask, y, mem,v_mask,a_mask), verbose=True)
                flops, params = clever_format([flops, params], "%.3f")
                print(flops,params)
                '''

                # 训练模型
                lld, nce, preds, pn_dic, H, _, t_mean, v_mean, a_mean, apf_mon = model(
                    text, visual, audio, vlens, alens,
                    bert_sent, bert_sent_type, bert_sent_mask,
                    y, 
                    mem, 
                    v_mask, a_mask,
                    # >>> APF monitor: τ 调度（全局步传给模型→CPAP→APF）
                    step=self.global_step, total_steps=self.total_steps
                )

                
                # >>> APF monitor: 批均值累计（打印用）
                if apf_mon is not None:
                    for k in apf_keys:
                        if k in apf_mon:
                            apf_sums[k] += float(apf_mon[k])
                    apf_cnt += 1
                    # >>> APF monitor: 严格全局（本 epoch 的 token 级加和）
                    apf_epoch["n_valid"] += int(apf_mon.get("n_valid", 0))
                    apf_epoch["sum_wv"]  += float(apf_mon.get("sum_wv", 0.0))
                    apf_epoch["sum_wa"]  += float(apf_mon.get("sum_wa", 0.0))
                    apf_epoch["sum_Kv"]  += float(apf_mon.get("sum_Kv", 0.0))
                    apf_epoch["sum_Ka"]  += float(apf_mon.get("sum_Ka", 0.0))

                all_t_mean.append(t_mean)
                all_v_mean.append(v_mean)
                all_a_mean.append(a_mean)

                if stage == 1:  # TASK+BA+CPC
                    y_loss = criterion(preds, y)
                    
                    # update memory
                    if len(mem_pos_tv) < mem_size:
                        mem_pos_tv.append(pn_dic['tv']['pos'].detach())
                        mem_neg_tv.append(pn_dic['tv']['neg'].detach())
                        mem_pos_ta.append(pn_dic['ta']['pos'].detach())
                        mem_neg_ta.append(pn_dic['ta']['neg'].detach())
                        if self.hp.add_va:
                            mem_pos_va.append(pn_dic['va']['pos'].detach())
                            mem_neg_va.append(pn_dic['va']['neg'].detach())
                    else:  # memory is full! replace the oldest with the newest data
                        oldest = i_batch % mem_size
                        mem_pos_tv[oldest] = pn_dic['tv']['pos'].detach()
                        mem_neg_tv[oldest] = pn_dic['tv']['neg'].detach()
                        mem_pos_ta[oldest] = pn_dic['ta']['pos'].detach()
                        mem_neg_ta[oldest] = pn_dic['ta']['neg'].detach()
                        if self.hp.add_va:
                            mem_pos_va[oldest] = pn_dic['va']['pos'].detach()
                            mem_neg_va[oldest] = pn_dic['va']['neg'].detach()

                    if self.hp.contrast:
                        loss = y_loss + self.alpha * nce - self.beta * lld
                    else:
                        loss = y_loss

                    if i_batch > mem_size:
                        #loss -= self.beta * H
                        pass
                    t1 = time.time()
                    loss.backward()
                    t2 = time.time()
                    print(t2 - t1)
                    exit(0)

                elif stage == 0:
                    # maximize likelihood equals minimize neg-likelihood
                    loss = -lld
                    loss.backward()
                else:
                    raise ValueError('stage index can either be 0 or 1')

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch

                    ### test
                    torch.cuda.synchronize()
                    # ===== 在 clip_grad_norm_ 之前做梯度体检（仅打印/断言，不改任何逻辑）=====
                    devs, dtypes, bad = _check_grads(model)

                    # 1) 有参数的 data.device 与 grad.device 不一致？
                    mismatch = []
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if p.grad.device != p.device:
                            mismatch.append((name, str(p.device), str(p.grad.device)))
                    if mismatch:
                        print("[GRAD CHECK] param.device != grad.device (前 8 条):", mismatch[:8])
                        assert False, "[GRAD-DEVICE-MISMATCH] 发现参数与其梯度不在同一设备"

                    # 2) 是否存在稀疏梯度？（dense 版 clip_grad_norm_ 在 Win 上更易崩）
                    has_sparse = any((p.grad is not None) and p.grad.is_sparse for p in model.parameters())
                    if has_sparse:
                        print("[GRAD CHECK] 存在稀疏梯度参数（前 8 条）:",
                            [n for n,p in list(model.named_parameters())[:100] if (p.grad is not None and p.grad.is_sparse)][:8])
                        assert False, "[GRAD-SPARSE] 发现稀疏梯度，先定位来源再裁剪"

                    # 3) 多设备 / 多 dtype / 非有限梯度
                    assert len(devs) <= 1, f"[GRAD-DEVICES] 多设备梯度: {devs}"
                    assert len(dtypes) <= 1, f"[GRAD-DTYPES] 多 dtype 梯度: {dtypes}"
                    assert not bad, f"[GRAD-NONFINITE] 梯度含 NaN/Inf（前 8 个参数名）: {bad[:8]}"

                    # 4) （可选）检查 loss 是否非有限（更早暴露爆炸）
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"[LOSS NONFINITE] loss={loss.item()}")
                    # ============================================================


                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()

                # >>> APF monitor: 全局步 +1（用于 τ 调度）
                self.global_step += 1

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                nce_loss += nce.item() * batch_size  # CPC loss
                ba_loss += (-H - lld) * batch_size  # BA loss
                task_loss += y_loss.item() * batch_size if stage==1 else 0  # Task loss

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size  #
                    avg_ba = ba_loss / proc_size
                    print(
                        'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                            format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                                   'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
                                   avg_loss, avg_nce, avg_ba))
                    # >>> APF monitor: 打印区间均值
                    if apf_cnt > 0:
                        apf_avg = {k: apf_sums[k] / max(apf_cnt, 1) for k in apf_keys}
                        print('    [APF] wv {:.3f} wa {:.3f} | Kv {:.3f} Ka {:.3f} | τ {:.3f} | v<.05 {:.2f} v>.95 {:.2f} | emaKv {:.3f} emaKa {:.3f}'.format(
                            apf_avg["w_v_mean"], apf_avg["w_a_mean"], apf_avg["Kv_mean"], apf_avg["Ka_mean"],
                            apf_avg["tau_now"], apf_avg["frac_wv_0"], apf_avg["frac_wv_1"], apf_avg["emaKv"], apf_avg["emaKa"]
                        ))
                        apf_sums = {k: 0.0 for k in apf_keys}
                        apf_cnt = 0

                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
                    eff_loss=0.0
                    start_time = time.time()

            # print("更新全局字典中......")
            
            # modal_lists = {
            #     "text_bert_mean": all_t_mean,
            #     "visual": all_v_mean,
            #     "audio": all_a_mean,
            # }
            # thresholds = (50, 100, 150, 200, 250)

            # for name, tensors in modal_lists.items():
            #     x = torch.cat(tensors, dim=0).detach().cpu()
            #     for k in thresholds:
            #         gen_npy(x, "mosei", k, name)
            
            # print("全局字典更完毕......")
            # exit(0)

            # >>> APF monitor: 返回严格全局统计（本 epoch）
            return task_loss / self.hp.n_train, epoch_loss / self.hp.n_train, apf_epoch

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0

            results = []
            truths = []

            # >>> APF monitor: 验证/测试累计器（严格全局）
            apf_epoch = {"n_valid":0, "sum_wv":0.0, "sum_wa":0.0, "sum_Kv":0.0, "sum_Ka":0.0}

            # 验证模型
            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids, v_mask, a_mask = batch

                    # with torch.cuda.device(0):
                    #     text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                    #     v_mask, a_mask = v_mask.cuda(), a_mask.cuda()
                    #     lengths = lengths.cuda()
                    #     bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                    text, audio, vision, y = text.to(self.device), audio.to(self.device), vision.to(self.device), y.to(self.device)
                    v_mask, a_mask = v_mask.to(self.device), a_mask.to(self.device)
                    lengths = lengths.to(self.device)
                    bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(self.device), bert_sent_type.to(self.device), bert_sent_mask.to(self.device)

                    batch_size = lengths.size(0)

                    # we don't need lld and bound anymore
                    _, _, preds, _, _, _, _, _, _, apf_mon = model(
                        text, vision, audio, vlens, alens, 
                        bert_sent, bert_sent_type, bert_sent_mask,
                        v_mask=v_mask, a_mask=a_mask,
                        # >>> APF monitor: τ 调度也在评估时透传（不更新 EMA）
                        step=self.global_step, total_steps=self.total_steps
                    )

                    # >>> APF monitor: 严格全局累计
                    if apf_mon is not None:
                        apf_epoch["n_valid"] += int(apf_mon.get("n_valid", 0))
                        apf_epoch["sum_wv"]  += float(apf_mon.get("sum_wv", 0.0))
                        apf_epoch["sum_wa"]  += float(apf_mon.get("sum_wa", 0.0))
                        apf_epoch["sum_Kv"]  += float(apf_mon.get("sum_Kv", 0.0))
                        apf_epoch["sum_Ka"]  += float(apf_mon.get("sum_Ka", 0.0))

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()  # 任务误差，MAE
                    if self.hp.dataset in ['mosi_ood', 'mosei_ood'] and test:
                        criterion =nn.CrossEntropyLoss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)  
            truths = torch.cat(truths)  

            #test
            # print(truths)
            # McNemar_test = {
            #     'results': results.cpu().squeeze(1),
            #     'truths': truths.cpu().squeeze(1)
            # }
            # data_frame = pd.DataFrame(data=McNemar_test)
            # data_frame.to_csv('TeFNA_mosi_MTest.csv')

            # >>> APF monitor: 返回严格全局统计（本轮 DEV/TEST）
            return avg_loss, results, truths, apf_epoch

        # >>> APF monitor: 画图函数（每个参数一张图：train/dev/test 三条线）
        def plot_apf_curves(epoch_idx: int):
            xs = list(range(1, epoch_idx+1))
            for key, title in [("wv","APF w_v (mean over valid tokens)"),
                               ("wa","APF w_a (mean over valid tokens)"),
                               ("Kv","APF K_v (sym KL, mean over valid tokens)"),
                               ("Ka","APF K_a (sym KL, mean over valid tokens)")]:
                plt.figure(figsize=(6,4))
                for split, color in [("train", None), ("dev", None), ("test", None)]:
                    ys = self.apf_hist[split][key]
                    if len(ys) == len(xs):
                        plt.plot(xs, ys, label=split)
                plt.xlabel("Epoch")
                plt.ylabel(key)
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(self.apf_plot_dir, f"apf_{key}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()

        best_valid, best_mae, best_loss = 1e8, 1e8, 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs + 1):
            start = time.time()

            self.epoch = epoch

            # maximize likelihood
            if self.hp.contrast:
                #train_loss = train(model, optimizer_mmilb, criterion, 0)
                pass

            # minimize all losses left
            tk_loss, train_loss, apf_train = train(model, optimizer_main, criterion, 1)

            val_loss, _, _, apf_dev = evaluate(model, criterion, test=False)
            test_loss, results, truths, apf_test = evaluate(model, criterion, test=True)

            # >>> APF monitor: 严格全局均值 = sum / n_valid
            def safe_mean(d, num_key, *sum_keys):
                n = max(1, int(d.get(num_key, 0)))
                return [float(d.get(k, 0.0)) / n for k in sum_keys]
            train_wv, train_wa, train_Kv, train_Ka = safe_mean(apf_train, "n_valid", "sum_wv","sum_wa","sum_Kv","sum_Ka")
            dev_wv,   dev_wa,   dev_Kv,   dev_Ka   = safe_mean(apf_dev,   "n_valid", "sum_wv","sum_wa","sum_Kv","sum_Ka")
            test_wv,  test_wa,  test_Kv,  test_Ka  = safe_mean(apf_test,  "n_valid", "sum_wv","sum_wa","sum_Kv","sum_Ka")

            # >>> APF monitor: 追加到历史并画图
            self.apf_hist['train']['wv'].append(train_wv); self.apf_hist['train']['wa'].append(train_wa)
            self.apf_hist['train']['Kv'].append(train_Kv); self.apf_hist['train']['Ka'].append(train_Ka)
            self.apf_hist['dev']['wv'].append(dev_wv);     self.apf_hist['dev']['wa'].append(dev_wa)
            self.apf_hist['dev']['Kv'].append(dev_Kv);     self.apf_hist['dev']['Ka'].append(dev_Ka)
            self.apf_hist['test']['wv'].append(test_wv);   self.apf_hist['test']['wa'].append(test_wa)
            self.apf_hist['test']['Kv'].append(test_Kv);   self.apf_hist['test']['Ka'].append(test_Ka)
            plot_apf_curves(epoch)

            self.plotter.update_epoch(epoch, tk_loss, val_loss, test_loss)

            end = time.time()
            duration = end - start
            scheduler_main.step(val_loss)  # Decay learning rate by validation loss

            # validation F1
            print("-" * 50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-" * 50)

            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss

                if "ood" in self.hp.dataset and test_loss < best_loss and not self.hp.seven_class:
                    best_epoch = epoch
                    best_loss = test_loss
                    best_acc = eval_ood_2(results, truths, True)
                    now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
                    self.hp.modelname = self.hp.dataset + now_time + '_loss' + str(best_loss)
                    print(f"Saved model at pre_trained_best_models_mosi/" + self.hp.modelname + ".pt!")
                    save_model(self.hp, model, self.hp.modelname)
                    best_results = results
                    best_truths = truths
                elif "ood" in self.hp.dataset and test_loss < best_loss and self.hp.seven_class:
                    best_epoch = epoch
                    best_loss = test_loss
                    best_acc = eval_ood_7(results, truths, True)
                    now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
                    self.hp.modelname = self.hp.dataset + now_time + '_loss' + str(best_loss)
                    print(f"Saved model at pre_trained_best_models_mosi/" + self.hp.modelname + ".pt!")
                    save_model(self.hp, model, self.hp.modelname)
                    best_results = results
                    best_truths = truths
                elif not "ood" in self.hp.dataset and test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    # 验证模型
                    if "mosei" in self.hp.dataset:
                        res = eval_mosei_senti(results, truths, True)
                    elif "mosi" in self.hp.dataset:
                        res = eval_mosi(results, truths, True)

                    # print(res['to_exl'])

                    best_results = results
                    best_truths = truths
                    now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
                    mae = str(best_mae).split('.')[0] + '.' + str(best_mae).split('.')[1][:5]
                    self.hp.modelname = self.hp.dataset + now_time + '_eam' + mae
                    print(f"Saved model -> " + self.hp.modelname + ".pt!")
                    save_model(self.hp, model, self.hp.modelname)
            else:
                patience -= 1
                if patience == 0:
                    break

        print(f'Best epoch: {best_epoch}')
        # 验证模型：
        if "ood" in self.hp.dataset and not self.hp.seven_class:
            self.best_dict = eval_ood_2(best_results, best_truths, True)
            return self.best_dict, self.hp.modelname
        elif "ood" in self.hp.dataset and self.hp.seven_class:
            self.best_dict = eval_ood_7(best_results, best_truths, True)
            return self.best_dict, self.hp.modelname
        elif "mosei" in self.hp.dataset:
            self.best_dict = eval_mosei_senti(best_results, best_truths, True)
        elif 'mosi' in self.hp.dataset:
            self.best_dict = eval_mosi(best_results, best_truths, True)

        # -------------保存识别结果----------

        if not "ood" in self.hp.dataset:
            #print(best_results)
            #print(best_truths)
            McNemar_test = {
                'results': best_results.cpu().squeeze(1),
                'truths': best_truths.cpu().squeeze(1)
            }
            data_frame = pd.DataFrame(data=McNemar_test)
            data_frame.to_csv('TeFNA_mosi_MTest.csv')

            return self.best_dict['to_exl'], self.hp.modelname
            # self.to_exl['mae'].append()

            # to_exl = self.best_dict['to_exl']
            # data_frame = pd.DataFrame(
            #     data={'mae': to_exl[0], 'coor': to_exl[1], 'acc7':to_exl[2], 'acc2_1':to_exl[3],
            #           'acc2_2': to_exl[4], 'f1_1': to_exl[5], 'f1_2': to_exl[6]}, index=[1])
            # data_frame.to_csv('pre_trained_best_models/'+self.hp.modelname+'.csv')

            sys.stdout.flush()
