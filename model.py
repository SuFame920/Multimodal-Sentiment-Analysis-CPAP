import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, FusionTrans,Encoder,SelfAttention,CrossAttention,FinalFusionSelfAttention,SumFusion,ConcatFusion, VarHeadZ, APF
from modules.InfoNCE import InfoNCE
from modules.UnimodelCPE import CPE, CounterFactualAttention
from utils.gen_Kmeans_center import gen_npy
from transformers import BertModel, BertConfig


class CPAP(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.uni_text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder

        self.uni_visual_enc = RNNEncoder(  # 视频特征提取
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.uni_acoustic_enc = RNNEncoder(  # 音频特征提取
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # For MI maximization   互信息最大化
        # Modality Mutual Information Lower Bound（MMILB）
        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:  # 一般是tv和ta   若va也要MMILB
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size=hp.d_tout,     # to be predicted
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        if hp.whether_debias_unimodal:
            # modal_type in ['audio','text','vision']
            if hp.whether_debias_audio:
                self.audio_mediator = CPE(hp, modal_type="audio")
                self.audio_mlp = SubNet(in_size=hp.d_ain + hp.model_dim_cross,
                                        hidden_size=hp.audio_mlp_hidden_size,
                                        n_class=None, dropout=hp.dropout_prj,
                                        output_size=hp.d_ain)  # [bs,seq_len,d_ain]
            else:
                self.uni_audio_encoder = SelfAttention(hp, d_in=hp.d_ain,
                                                       d_model=hp.model_dim_self,
                                                       nhead=hp.num_heads_self,
                                                       dim_feedforward=4 * hp.model_dim_self,
                                                       dropout=hp.attn_dropout_self,
                                                       num_layers=hp.num_layers_self)
            if hp.whether_debias_text:
                self.text_mediator = CPE(hp, modal_type="text")
                self.text_mlp = SubNet(in_size=hp.d_tin + hp.model_dim_cross,
                                       hidden_size=hp.text_mlp_hidden_size,
                                       n_class=None, dropout=hp.dropout_prj,
                                       output_size=hp.d_tin)  # [bs,seq_len,d_tin]
            else:
                pass

            if hp.whether_debias_vision:
                self.vision_mediator = CPE(hp, modal_type="vision")
                self.vision_mlp = SubNet(in_size=hp.d_vin + hp.model_dim_cross,
                                         hidden_size=hp.vision_mlp_hidden_size,
                                         n_class=None, dropout=hp.dropout_prj,
                                         output_size=hp.d_vin)  # [bs,seq_len,d_vin]
            else:
                self.uni_vision_encoder = SelfAttention(hp, d_in=hp.d_vin,
                                                        d_model=hp.model_dim_self,
                                                        nhead=hp.num_heads_self,
                                                        dim_feedforward=4 * hp.model_dim_self,
                                                        dropout=hp.attn_dropout_self,
                                                        num_layers=hp.num_layers_self)
        else:
            self.uni_audio_encoder = SelfAttention(hp, d_in=hp.d_ain,
                                                   d_model=hp.model_dim_self,
                                                   nhead=hp.num_heads_self,
                                                   dim_feedforward=4 * hp.model_dim_self,
                                                   dropout=hp.attn_dropout_self,
                                                   num_layers=hp.num_layers_self)
            self.uni_vision_encoder = SelfAttention(hp, d_in=hp.d_vin,
                                                    d_model=hp.model_dim_self,
                                                    nhead=hp.num_heads_self,
                                                    dim_feedforward=4 * hp.model_dim_self,
                                                    dropout=hp.attn_dropout_self,
                                                    num_layers=hp.num_layers_self)

        # 用 MULT 融合：每个模块的输出都是 [L,B,model_dim_cross]
        self.ta_cross_attn = CrossAttention(hp, d_modal1=hp.d_tin, d_modal2=hp.d_ain,
                                            d_model=hp.model_dim_cross, nhead=hp.num_heads_cross,
                                            dim_feedforward=4 * hp.model_dim_cross,
                                            dropout=hp.attn_dropout_cross,
                                            num_layers=hp.num_layers_cross)
        self.tv_cross_attn = CrossAttention(hp, d_modal1=hp.d_tin, d_modal2=hp.d_vin,
                                            d_model=hp.model_dim_cross, nhead=hp.num_heads_cross,
                                            dim_feedforward=4 * hp.model_dim_cross,
                                            dropout=hp.attn_dropout_cross,
                                            num_layers=hp.num_layers_cross)

        # 反事实模块（可选）：每个模块的输出都是 [L,B,model_dim_cross]
        if hp.whether_use_counterfactual:
            self.ta_counterfactual_attn = CounterFactualAttention(
                hp, d_modal1=hp.d_tin, d_modal2=hp.d_ain,
                d_model=hp.model_dim_cross, nhead=hp.num_heads_cross,
                dim_feedforward=4 * hp.model_dim_cross,
                dropout=hp.attn_dropout_cross,
                num_layers=hp.num_layers_counterfactual_attention
            )
            self.tv_counterfactual_attn = CounterFactualAttention(
                hp, d_modal1=hp.d_tin, d_modal2=hp.d_vin,
                d_model=hp.model_dim_cross, nhead=hp.num_heads_cross,
                dim_feedforward=4 * hp.model_dim_cross,
                dropout=hp.attn_dropout_cross,
                num_layers=hp.num_layers_counterfactual_attention
            )
            self.fusion_mlp_for_counterfactual_regression = SubNet(
                in_size=hp.model_dim_cross * 2,
                hidden_size=hp.d_prjh,
                dropout=hp.dropout_prj,
                n_class=hp.n_class
            )

        # 原有（非 APF）融合回归头：输入是 cat(mean_ta, mean_tv) → 2*d_model
        self.fusion_mlp_for_regression = SubNet(
            in_size=hp.model_dim_cross * 2,
            hidden_size=hp.d_prjh,
            dropout=hp.dropout_prj,
            n_class=hp.n_class
        )

        # ---- APF 模块（可开关）----
        apf_enabled = getattr(hp, 'apf_enable', True)
        apf_d_model = getattr(hp, 'apf_d_model', -1)
        apf_d_model = hp.model_dim_cross if apf_d_model == -1 else apf_d_model

        if apf_enabled:
            self.apf = APF(
                D_tin=hp.d_tin,
                d_model=apf_d_model,
                z_dim=getattr(hp, 'apf_z_dim', 32),
                clamp_min=getattr(hp, 'apf_clamp_min', -8.0),
                clamp_max=getattr(hp, 'apf_clamp_max', 2.0),
                tau_start=getattr(hp, 'apf_tau_start', 0.7),
                tau_end=getattr(hp, 'apf_tau_end', 0.5),
                beta=getattr(hp, 'apf_beta', 0.99),
                ln_eps=1e-5,
                use_LN_before_varhead=getattr(hp, 'apf_use_LN_before_varhead', True),
                use_proj_LN_in=getattr(hp, 'apf_use_proj_LN_in', True),
                use_proj_LN_out=getattr(hp, 'apf_use_proj_LN_out', True),
                eps=getattr(hp, 'apf_eps', 1e-6),
            )
            # APF 专用回归头：输入是 APF 的 masked mean → d_model
            self.apf_fusion_mlp_for_regression = SubNet(
                in_size=apf_d_model,
                hidden_size=hp.d_prjh,
                dropout=hp.dropout_prj,
                n_class=hp.n_class
            )
        else:
            self.apf = None
            self.apf_fusion_mlp_for_regression = None

        #self.APF = BaseAPF()

    def gen_mask(self, a, length=None):
        if length is None:
            msk_tmp = torch.sum(a, dim=-1)  # d全为0时
            mask = (msk_tmp == 0)
            return mask
        else:
            b = a.shape[0]
            l = a.shape[1]
            msk = torch.ones((b, l))
            x = []
            y = []
            for i in range(b):
                for j in range(length[i], l):
                    x.append(i)
                    y.append(j)
            msk[x, y] = 0
            return (msk == 0)

    def masked_mean_LBD(self, x: torch.Tensor, pad_mask: torch.Tensor, only_keep_valid=True) -> torch.Tensor:
        """
        x:        (L, B, D)
        pad_mask: (B, L)  True=PAD(无效)
        返回:      (B_valid, D)  过滤掉全PAD的样本，避免对KMeans产生干扰，减少无效计算
        """
        valid = ~pad_mask                                 # (B,L)
        m = valid.t().unsqueeze(-1).to(dtype=x.dtype)     # (L,B,1)
        sums = (x * m).sum(dim=0)                         # (B,D)
        counts = valid.sum(dim=1)                         # (B,)
        means = sums / counts.clamp(min=1).unsqueeze(-1).to(x.dtype)  # (B,D)
        return means[counts != 0] if only_keep_valid else means

    def forward(self, sentences, visual, acoustic,
                      v_len, a_len,
                      bert_sent, bert_sent_type, bert_sent_mask,
                      y=None,
                      mem=None,
                      v_mask=None, a_mask=None,
                      step=None, total_steps=None):

        X_t = self.uni_text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask)  # [B,L,768]
        X_t = X_t.transpose(0, 1)  # [L,B,768]
        X_v = visual # [L,B,d_vin]
        X_a = acoustic # [L,B,d_ain]
        print(X_v.shape)

        with torch.no_grad():
            maskT = (bert_sent_mask == 0)  # [B,L]  True=PAD
            maskV = self.gen_mask(visual.transpose(0, 1), v_len).to(device=visual.device, dtype=torch.bool)
            maskA = self.gen_mask(acoustic.transpose(0, 1), a_len).to(device=acoustic.device, dtype=torch.bool)
            # 加断言（只打印，一旦触发你就能一眼看到是哪一轮/哪一形状）
            # assert maskT.dtype == torch.bool and maskV.dtype == torch.bool and maskA.dtype == torch.bool, \
            #     f"mask dtype wrong: T={maskT.dtype}, V={maskV.dtype}, A={maskA.dtype}"
            # assert maskT.device == bert_sent_mask.device and maskV.device == visual.device and maskA.device == acoustic.device, \
            #     f"mask device wrong: T={maskT.device}/{bert_sent_mask.device}, V={maskV.device}/{visual.device}, A={maskA.device}/{acoustic.device}"  

        # 第一个epoch，生成全局字典, 主干全搭好后再生成一次！
        # with torch.no_grad():
        #     t_mean = self.masked_mean_LBD(X_t, maskT)
        #     v_mean = self.masked_mean_LBD(X_v, maskV)
        #     a_mean = self.masked_mean_LBD(X_a, maskA) 

        # 1. 三个模态 分别去偏 / 或仅自注意力
        if self.hp.whether_debias_unimodal:
            if self.hp.whether_debias_audio:
                X_a = self.audio_mediator(X_a, maskA)
                F_a = self.audio_mlp(X_a)  # [L,B,d_ain]
            else:
                F_a = self.uni_audio_encoder(X_a)  # [L,B,d]
            if self.hp.whether_debias_text:
                X_t = self.text_mediator(X_t, maskT)
                F_t = self.text_mlp(X_t)  # [L,B,d_tin]
            else:
                F_t = X_t
            if self.hp.whether_debias_vision:
                X_v = self.vision_mediator(X_v, maskV)
                F_v = self.vision_mlp(X_v)  # [L,B,d_vin]
            else:
                F_v = self.uni_vision_encoder(X_v)  # [L,B,d]
        else:
            F_t = X_t
            F_a = self.uni_audio_encoder(X_a)  # [L,B,d]
            F_v = self.uni_vision_encoder(X_v)     # [L,B,d]

        # 2. 跨模态注意力部分（输出 [L,B,model_dim_cross]）
        F_tv = self.tv_cross_attn(F_t, F_v, Tmask=maskT, Vmask=maskV)
        F_ta = self.ta_cross_attn(F_t, F_a,  Tmask=maskT, Amask=maskA)

        # 3. APF
        use_apf = (self.apf is not None)
        apf_mon = None
        if use_apf:
            # APF 输出 [L,B,d_model]；随后用 maskT 做 masked mean → [B,d_model]
            F_apf, apf_mon = self.apf(
                Ft=F_t,                      # [L,B,768]
                Fvt=F_tv,                       # [L,B,d_model]
                Fat=F_ta,                       # [L,B,d_model]
                maskT=maskT.transpose(0, 1),        # [L,B] (1=pad)
                train_step=self.training,
                step=step, total_steps=total_steps
            )
            apf_pool = self.masked_mean_LBD(F_apf, maskT, only_keep_valid=False)  # [B,d_model]
            fusion_core, preds_out = self.apf_fusion_mlp_for_regression(apf_pool)  # fusion_core:[B, d_prjh]
        else:
            # 原路径：mean over L，然后 concat 再回归
            # mean_tv = cross_tv.mean(dim=0)  # [B,d_model]
            # mean_ta = cross_ta.mean(dim=0)  # [B,d_model]
            mean_tv = self.masked_mean_LBD(F_tv, maskT, only_keep_valid=False)
            mean_ta = self.masked_mean_LBD(F_ta, maskT, only_keep_valid=False)
            fusion_core, preds_out = self.fusion_mlp_for_regression(torch.cat([mean_ta, mean_tv], dim=1))  # [B,2*d_model]→fusion:[B,d_prjh]

        # MI / CPC（融合表示用当前选定的 fusion_core：APF 优先）
        if self.training:
            text = F_t[0, :, :]            # [B,768]
            acoustic_pool = self.uni_acoustic_enc(F_a, a_len)  # [B,d_aout]
            visual_pool = self.uni_visual_enc(F_v, v_len)    # [B,d_vout]

            if y is not None:
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual_pool,   labels=y, mem=mem['tv'])
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic_pool, labels=y, mem=mem['ta'])
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual_pool, y=acoustic_pool, labels=y, mem=mem['va'])
            else:
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual_pool)
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic_pool)
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual_pool, y=acoustic_pool)

            # CPC：用 fusion_core
            nce_t = self.cpc_zt(text, fusion_core)
            nce_v = self.cpc_zv(visual_pool, fusion_core)
            nce_a = self.cpc_za(acoustic_pool, fusion_core)
            nce = nce_t + nce_v + nce_a

            pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
            lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
            H = H_tv  + H_ta  + (H_va  if self.add_va else 0.0)

        # 返回
        if self.training:
            return lld, nce, preds_out, pn_dic, H, None, None, None, None, apf_mon
            # return lld, nce, preds_out, pn_dic, H, None, t_mean, v_mean, a_mean, apf_mon
        else:
            return None, None, preds_out, None, None, None, None, None, None, apf_mon
