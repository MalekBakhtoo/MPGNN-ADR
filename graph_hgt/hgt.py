"""
Heterogeneous Graph Transformer (HGT) for ADR prediction on the MFAHGN / GRDTI dataset.

Usage (from project root, where ./newdata/ exists):
    python hgt_adr_prediction.py

این اسکریپت:
  * ماتریس‌های رابطه‌ای را از ./newdata/ می‌خواند
  * گراف ناهمگن DGL را می‌سازد
  * یک مدل HGT روی کل گراف ناهمگن آموزش می‌دهد
  * با بازسازی روابط (DDI, Drug–Disease, Drug–Protein, PPI, Protein–Disease, Drug–SideEffect)
    و نمونه‌برداری منفی، مدل را آموزش می‌دهد
  * معیارهای AUC / AUPR / Precision / Recall / F1 / MCC را روی تست گزارش می‌کند
  * از early stopping بر اساس AUPR روی validation استفاده می‌کند
  * بهترین مدل هر فولد و اطلاعات مهم آن را در پوشه saved_models ذخیره می‌کند
"""

import os
import time
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl import NTYPE, ETYPE
from dgl.nn.pytorch.conv import HGTConv

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split


# =============================================================
# 1) Utilities
# =============================================================


def l2_norm(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Row-wise L2 normalization with numerical safety."""
    t = t.float()
    norm = torch.norm(t, p=2, dim=dim, keepdim=True) + 1e-12
    out = t / norm
    out[torch.isnan(out) | torch.isinf(out)] = 0.0
    return out


@dataclass
class Args:
    # training
    epochs: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    reg_lambda: float = 1.0
    patience: int = 20          # early stopping patience

    # embedding / HGT
    dim_embedding: int = 256
    num_heads: int = 4
    hgt_dropout: float = 0.2

    # optional extra propagation after HGT (like APPNP)
    k: int = 3
    alpha: float = 0.8
    edge_drop: float = 0.3

    # evaluation / sampling
    n_fold: int = 3  # چند بار آموزش از صفر با همان split (برای میانگین‌گیری)


# =============================================================
# 2) Data loading (مثل نسخه‌های قبلی)
# =============================================================


def load_drug_sideeffect_mapping(excel_path: str = "./data/takeToDict_data.xlsx"):
    """Optional: نگاشت اسم دارو/عوارض برای تفسیر نتایج (در آموزش استفاده نمی‌شود)."""
    if not os.path.exists(excel_path):
        return {}, {}

    df = pd.read_excel(excel_path)

    drug_dict = {}
    drug_index = 0
    for drug_name in df["drugName"]:
        if pd.isnull(drug_name) or drug_index >= 680:
            break
        drug_dict[drug_name] = int(drug_index)
        drug_index += 1

    side_effect_dict = {}
    side_effect_index = 0
    for se_id, group in df.groupby("sideeffectId"):
        if side_effect_index >= 2600:
            break
        names = group["sideeffectName"].tolist()
        side_effect_dict[int(side_effect_index)] = names[0] if names else f"SE_{side_effect_index}"
        side_effect_index += 1

    return drug_dict, side_effect_dict


def load_relation_matrices(network_path: str = "./newdata/"):
    """Load all relation matrices from text files (همان فرمت GRDTI اصلی)."""
    drug_drug = np.loadtxt(os.path.join(network_path, "mat_drug_drug.txt"))
    drug_disease = np.loadtxt(os.path.join(network_path, "mat_drug_disease.txt"))
    drug_sideeffect = np.loadtxt(os.path.join(network_path, "mat_drug_se.txt"))
    protein_protein = np.loadtxt(os.path.join(network_path, "mat_protein_protein.txt"))
    protein_disease = np.loadtxt(os.path.join(network_path, "mat_protein_disease.txt"))
    drug_protein = np.loadtxt(os.path.join(network_path, "mat_drug_protein.txt"))

    print("[Data] Loaded relation matrices from", network_path)
    return (
        drug_drug,
        drug_disease,
        drug_protein,
        protein_protein,
        protein_disease,
        drug_sideeffect,
    )


def construct_hetero_graph(
    drug_drug,
    drug_disease,
    drug_protein,
    protein_protein,
    protein_disease,
    drug_sideeffect,
):
    """Construct DGL heterogeneous graph با همان schema مقاله."""
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = drug_disease.shape[1]
    num_sideeffect = drug_sideeffect.shape[1]

    drug_nodes = [(i, i) for i in range(num_drug)]
    protein_nodes = [(i, i) for i in range(num_protein)]
    disease_nodes = [(i, i) for i in range(num_disease)]
    sideeffect_nodes = [(i, i) for i in range(num_sideeffect)]

    ddi_edges = [
        (i, j)
        for i in range(num_drug)
        for j in range(num_drug)
        if drug_drug[i, j] > 0
    ]

    drug_disease_edges = []
    disease_drug_edges = []
    for i in range(num_drug):
        for j in range(num_disease):
            if drug_disease[i, j] > 0:
                drug_disease_edges.append((i, j))
                disease_drug_edges.append((j, i))

    drug_protein_edges = []
    protein_drug_edges = []
    for i in range(num_drug):
        for j in range(num_protein):
            if drug_protein[i, j] > 0:
                drug_protein_edges.append((i, j))
                protein_drug_edges.append((j, i))

    ppi_edges = [
        (i, j)
        for i in range(num_protein)
        for j in range(num_protein)
        if protein_protein[i, j] > 0
    ]

    protein_disease_edges = []
    disease_protein_edges = []
    for i in range(num_protein):
        for j in range(num_disease):
            if protein_disease[i, j] > 0:
                protein_disease_edges.append((i, j))
                disease_protein_edges.append((j, i))

    drug_sideeffect_edges = []
    sideeffect_drug_edges = []
    for i in range(num_drug):
        for j in range(num_sideeffect):
            if drug_sideeffect[i, j] > 0:
                drug_sideeffect_edges.append((i, j))
                sideeffect_drug_edges.append((j, i))

    g_hetero = dgl.heterograph(
        {
            ("disease", "disease_disease virtual", "disease"): disease_nodes,
            ("drug", "drug_drug virtual", "drug"): drug_nodes,
            ("protein", "protein_protein virtual", "protein"): protein_nodes,
            ("sideeffect", "sideeffect_sideeffect virtual", "sideeffect"): sideeffect_nodes,
            ("drug", "drug_drug interaction", "drug"): ddi_edges,
            ("protein", "protein_protein interaction", "protein"): ppi_edges,
            ("drug", "drug_protein interaction", "protein"): drug_protein_edges,
            ("protein", "protein_drug interaction", "drug"): protein_drug_edges,
            ("drug", "drug_sideeffect association", "sideeffect"): drug_sideeffect_edges,
            ("sideeffect", "sideeffect_drug association", "drug"): sideeffect_drug_edges,
            ("drug", "drug_disease association", "disease"): drug_disease_edges,
            ("disease", "disease_drug association", "drug"): disease_drug_edges,
            ("protein", "protein_disease association", "disease"): protein_disease_edges,
            ("disease", "disease_protein association", "protein"): disease_protein_edges,
        }
    )

    g = g_hetero.edge_type_subgraph(
        [
            "drug_drug interaction",
            "protein_protein interaction",
            "drug_protein interaction",
            "protein_drug interaction",
            "drug_sideeffect association",
            "sideeffect_drug association",
            "drug_disease association",
            "disease_drug association",
            "protein_disease association",
            "disease_protein association",
        ]
    )

    return g


# =============================================================
# 3) APPNP-style propagation (اختیاری بعد از HGT)
# =============================================================


class Propagation(nn.Module):
    """Simple personalized propagation over homogeneous graph (APPNP-style)."""

    def __init__(self, k: int, alpha: float, edge_drop: float = 0.0):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        g = graph.local_var().to(feat.device)
        deg = g.in_degrees().float().clamp(min=1e-12)
        norm = torch.pow(deg, -0.5).view(-1, 1)

        h0 = feat
        h = feat
        for _ in range(self.k):
            h = h * norm
            g.ndata["h"] = h
            g.edata["w"] = self.edge_drop(
                torch.ones(g.number_of_edges(), 1, device=feat.device)
            )
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
            h = g.ndata.pop("h")
            h = h * norm
            h = (1.0 - self.alpha) * h + self.alpha * h0
        return h


# =============================================================
# 4) HGT-based ADR model
# =============================================================


class HGTADR(nn.Module):
    """Heterogeneous Graph Transformer for ADR prediction with reconstruction loss."""

    def __init__(
        self,
        g: dgl.DGLGraph,
        n_disease: int,
        n_drug: int,
        n_protein: int,
        n_sideeffect: int,
        args: Args,
    ):
        super().__init__()
        self.g_hetero = g
        self.device = torch.device(args.device)
        self.dim_embedding = args.dim_embedding
        self.reg_lambda = args.reg_lambda

        self.num_disease = n_disease
        self.num_drug = n_drug
        self.num_protein = n_protein
        self.num_sideeffect = n_sideeffect

        # Trainable initial embeddings per node type
        self.disease_feat = nn.Parameter(
            torch.randn(n_disease, self.dim_embedding) * 0.1
        )
        self.drug_feat = nn.Parameter(
            torch.randn(n_drug, self.dim_embedding) * 0.1
        )
        self.protein_feat = nn.Parameter(
            torch.randn(n_protein, self.dim_embedding) * 0.1
        )
        self.sideeffect_feat = nn.Parameter(
            torch.randn(n_sideeffect, self.dim_embedding) * 0.1
        )

        # Convert hetero graph to homogeneous graph for HGTConv
        self.g_homo, self.ntype_count, self.etype_count = dgl.to_homogeneous(
            g, ndata=None, edata=None, store_type=True, return_count=True
        )
        self.g_homo = self.g_homo.to(self.device)
        self.ntype_ids = self.g_homo.ndata[NTYPE].to(self.device)
        self.etype_ids = self.g_homo.edata[ETYPE].to(self.device)

        self.ntypes = g.ntypes               # ['disease','drug','protein','sideeffect']
        self.num_ntypes = len(self.ntypes)
        self.num_etypes = len(g.etypes)

        # HGT layers
        self.num_heads = args.num_heads
        head_size = self.dim_embedding // self.num_heads

        self.hgt1 = HGTConv(
            in_size=self.dim_embedding,
            head_size=head_size,
            num_heads=self.num_heads,
            num_ntypes=self.num_ntypes,
            num_etypes=self.num_etypes,
            dropout=args.hgt_dropout,
            use_norm=True,
        )
        self.hgt2 = HGTConv(
            in_size=head_size * self.num_heads,
            head_size=head_size,
            num_heads=self.num_heads,
            num_ntypes=self.num_ntypes,
            num_etypes=self.num_etypes,
            dropout=args.hgt_dropout,
            use_norm=True,
        )

        # project HGT output back to dim_embedding
        self.proj = nn.Linear(head_size * self.num_heads, self.dim_embedding)

        # Optional propagation after HGT
        self.propagation = Propagation(args.k, args.alpha, args.edge_drop)

        # Diagonal reconstruction parameters (مثل DistMult)
        tmp = torch.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_P = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Side = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_PPI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_P_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))

    def forward(
        self,
        drug_sideeffect_mask: torch.Tensor,
        drug_drug: torch.Tensor,
        drug_disease: torch.Tensor,
        drug_protein: torch.Tensor,
        protein_protein: torch.Tensor,
        protein_disease: torch.Tensor,
        drug_sideeffect: torch.Tensor,
    ):
        # 1) initial node features (same order as to_homogeneous)
        type_feat_dict = {
            "disease": self.disease_feat,
            "drug": self.drug_feat,
            "protein": self.protein_feat,
            "sideeffect": self.sideeffect_feat,
        }

        feat_list = []
        for ntype in self.ntypes:
            feat_list.append(type_feat_dict[ntype])
        x0 = torch.cat(feat_list, dim=0).to(self.device)

        # 2) HGTConv on homogeneous graph
        h = self.hgt1(self.g_homo, x0, self.ntype_ids, self.etype_ids, presorted=True)
        h = F.relu(h)
        h = self.hgt2(self.g_homo, h, self.ntype_ids, self.etype_ids, presorted=True)
        h = self.proj(h)  # (N_total, dim_embedding)

        # 3) APPNP-style propagation
        h = self.propagation(self.g_homo, h)

        # 4) split embeddings by node type
        emb_by_type = {}
        start = 0
        for ntype, count in zip(self.ntypes, self.ntype_count):
            emb_by_type[ntype] = h[start: start + count]
            start += count

        disease_embedding = emb_by_type["disease"]
        drug_embedding = emb_by_type["drug"]
        protein_embedding = emb_by_type["protein"]
        sideeffect_embedding = emb_by_type["sideeffect"]

        disease_vec = l2_norm(disease_embedding)
        drug_vec = l2_norm(drug_embedding)
        protein_vec = l2_norm(protein_embedding)
        sideeffect_vec = l2_norm(sideeffect_embedding)

        # 5) relation reconstruction
        drug_drug_rec = drug_vec @ self.re_DDI @ drug_vec.t()
        loss_ddi = torch.sum((drug_drug_rec - drug_drug.float()) ** 2)

        drug_disease_rec = drug_vec @ self.re_D_Di @ disease_vec.t()
        loss_dise = torch.sum((drug_disease_rec - drug_disease.float()) ** 2)

        drug_protein_rec = drug_vec @ self.re_D_P @ protein_vec.t()
        loss_dp = torch.sum((drug_protein_rec - drug_protein.float()) ** 2)

        protein_protein_rec = protein_vec @ self.re_PPI @ protein_vec.t()
        loss_ppi = torch.sum((protein_protein_rec - protein_protein.float()) ** 2)

        protein_disease_rec = protein_vec @ self.re_P_Di @ disease_vec.t()
        loss_pd = torch.sum((protein_disease_rec - protein_disease.float()) ** 2)

        drug_sideeffect_rec = drug_vec @ self.re_D_Side @ sideeffect_vec.t()
        tmp = drug_sideeffect_mask.float() * (
            drug_sideeffect_rec - drug_sideeffect.float()
        )
        dse_potential = drug_sideeffect_rec - drug_sideeffect.float()
        loss_dse = torch.sum(tmp**2)

        other_loss = loss_ddi + loss_dise + loss_dp + loss_ppi + loss_pd

        # L2 regularization
        l2 = torch.tensor(0.0, device=self.device)
        for name, p in self.named_parameters():
            if "bias" not in name:
                l2 = l2 + torch.sum(p.pow(2))
        l2 = 0.5 * l2

        total_loss = loss_dse + 1.0 * other_loss + self.reg_lambda * l2

        return total_loss, loss_dse, l2, drug_sideeffect_rec, dse_potential


# =============================================================
# 5) Training & evaluation (با ذخیره بهترین مدل و info)
# =============================================================


def train_and_evaluate(
    DSE_train,
    DSE_valid,
    DSE_test,
    num_sideeffect,
    args: Args,
    drug_drug,
    drug_disease,
    drug_protein,
    protein_protein,
    protein_disease,
    fold_index: int,
    save_dir: str,
):
    device = torch.device(args.device)

    num_disease = drug_disease.shape[1]
    num_drug = drug_drug.shape[0]
    num_protein = protein_protein.shape[0]

    # ماتریس کامل drug-sideeffect برای بازسازی
    drug_sideeffect = torch.zeros((num_drug, num_sideeffect), dtype=torch.float32)
    mask = torch.zeros_like(drug_sideeffect, device=device)
    for d, s, label in DSE_train:
        drug_sideeffect[d, s] = float(label)
        mask[d, s] = 1.0

    # گراف ناهمگن
    g = construct_hetero_graph(
        drug_drug,
        drug_disease,
        drug_protein,
        protein_protein,
        protein_disease,
        drug_sideeffect.numpy(),
    )

    # تبدیل ماتریس‌ها به Tensor روی device
    drug_drug_t = torch.tensor(drug_drug, dtype=torch.float32, device=device)
    drug_disease_t = torch.tensor(drug_disease, dtype=torch.float32, device=device)
    drug_protein_t = torch.tensor(drug_protein, dtype=torch.float32, device=device)
    protein_protein_t = torch.tensor(protein_protein, dtype=torch.float32, device=device)
    protein_disease_t = torch.tensor(protein_disease, dtype=torch.float32, device=device)
    drug_sideeffect_t = drug_sideeffect.to(device)

    model = HGTADR(
        g,
        n_disease=num_disease,
        n_drug=num_drug,
        n_protein=num_protein,
        n_sideeffect=num_sideeffect,
        args=args,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    best_valid_aupr = 0.0
    best_test_auc = 0.0
    best_test_aupr = 0.0
    patience = 0

    precision_hist, recall_hist, f1_hist, mcc_hist = [], [], [], []

    # برای ذخیره بهترین مدل همین فولد
    best_model_path = os.path.join(save_dir, f"hgt_best_fold{fold_index+1}.pt")
    best_info_path = os.path.join(save_dir, f"hgt_best_fold{fold_index+1}_info.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, dse_loss, l2_loss, dse_rec, dse_pot = model(
            mask.to(device),
            drug_drug_t,
            drug_disease_t,
            drug_protein_t,
            protein_protein_t,
            protein_disease_t,
            drug_sideeffect_t,
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---- Evaluation ------
        model.eval()
        with torch.no_grad():
            scores = dse_rec.detach().cpu().numpy()

            # validation
            val_scores = [scores[d, s] for d, s, _ in DSE_valid]
            val_labels = [int(lbl) for _, _, lbl in DSE_valid]

            valid_auc = roc_auc_score(val_labels, val_scores)
            valid_aupr = average_precision_score(val_labels, val_scores)

            # threshold = 0.5 برای محاسبه P/R/F1/MCC
            val_pred_labels = (np.array(val_scores) >= 0.5).astype(int)
            precision = precision_score(val_labels, val_pred_labels, zero_division=0)
            recall = recall_score(val_labels, val_pred_labels, zero_division=0)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            mcc = matthews_corrcoef(val_labels, val_pred_labels)

            precision_hist.append(precision)
            recall_hist.append(recall)
            f1_hist.append(f1)
            mcc_hist.append(mcc)

            # فقط وقتی اعتبارسنجی بهتر شد، تست را آپدیت کن و مدل را ذخیره کن (early stopping)
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0

                test_scores = [scores[d, s] for d, s, _ in DSE_test]
                test_labels = [int(lbl) for _, _, lbl in DSE_test]

                best_test_auc = roc_auc_score(test_labels, test_scores)
                best_test_aupr = average_precision_score(test_labels, test_scores)

                # ---------- ذخیره بهترین مدل و اطلاعات مهم ----------
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "valid_auc": float(valid_auc),
                        "valid_aupr": float(valid_aupr),
                        "test_auc": float(best_test_auc),
                        "test_aupr": float(best_test_aupr),
                        "args": vars(args),
                    },
                    best_model_path,
                )

                best_info = {
                    "fold": int(fold_index + 1),
                    "epoch": int(epoch),
                    "valid_auc": float(valid_auc),
                    "valid_aupr": float(valid_aupr),
                    "test_auc": float(best_test_auc),
                    "test_aupr": float(best_test_aupr),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "mcc": float(mcc),
                    "args": vars(args),
                }
                with open(best_info_path, "w", encoding="utf-8") as f:
                    json.dump(best_info, f, indent=2, ensure_ascii=False)
            else:
                patience += 1

        print(
            f"Epoch {epoch:03d} | Loss {total_loss.item():.4f} | "
            f"DSE {dse_loss.item():.4f} | L2 {l2_loss.item():.4f} | "
            f"Val AUC {valid_auc:.4f} | Val AUPR {valid_aupr:.4f} | "
            f"Prec {precision:.4f} | Rec {recall:.4f} | F1 {f1:.4f} | MCC {mcc:.4f}"
        )

        if patience >= args.patience:
            print("[Early stopping] No validation improvement.")
            break

    return (
        best_test_auc,
        best_test_aupr,
        float(np.mean(precision_hist)),
        float(np.mean(recall_hist)),
        float(np.mean(f1_hist)),
        float(np.mean(mcc_hist)),
    )


# =============================================================
# 6) Main
# =============================================================


def main():
    args = Args()
    print("[Config]", args)

    # پوشه ذخیره مدل‌ها و نتایج
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    (
        drug_drug,
        drug_disease,
        drug_protein,
        protein_protein,
        protein_disease,
        drug_sideeffect,
    ) = load_relation_matrices("./newdata/")

    num_sideeffect = drug_sideeffect.shape[1]

    # ساخت positive / negative pairs (نسبت 1:10)
    pos_idx = []
    neg_idx = []
    for i in range(drug_sideeffect.shape[0]):
        for j in range(drug_sideeffect.shape[1]):
            if int(drug_sideeffect[i, j]) == 1:
                pos_idx.append((i, j))
            else:
                neg_idx.append((i, j))

    rng = np.random.default_rng(seed=42)
    neg_sample_idx = rng.choice(
        np.arange(len(neg_idx)), size=10 * len(pos_idx), replace=False
    )

    dataset = np.zeros((len(pos_idx) + len(neg_sample_idx), 3), dtype=int)
    count = 0
    for i, j in pos_idx:
        dataset[count] = [i, j, 1]
        count += 1
    for k in neg_sample_idx:
        i, j = neg_idx[k]
        dataset[count] = [i, j, 0]
        count += 1

    # یک بار split به train/valid/test (نسبت 0.81 / 0.09 / 0.10)
    train, DSE_test = train_test_split(dataset, test_size=0.1, random_state=42)
    DSE_train, DSE_valid = train_test_split(train, test_size=0.1, random_state=42)

    DSE_train = np.array(DSE_train, dtype=int)
    DSE_valid = np.array(DSE_valid, dtype=int)
    DSE_test = np.array(DSE_test, dtype=int)

    results = []
    start = time.time()
    for fold in range(args.n_fold):
        print("\n==============================")
        print(f"[Fold {fold + 1}/{args.n_fold}]")
        print("==============================")
        (
            auc,
            aupr,
            p_mean,
            r_mean,
            f1_mean,
            mcc_mean,
        ) = train_and_evaluate(
            DSE_train,
            DSE_valid,
            DSE_test,
            num_sideeffect,
            args,
            drug_drug,
            drug_disease,
            drug_protein,
            protein_protein,
            protein_disease,
            fold_index=fold,
            save_dir=save_dir,
        )
        results.append((auc, aupr, p_mean, r_mean, f1_mean, mcc_mean))

    end = time.time()

    res_arr = np.array(results)
    mean = res_arr.mean(axis=0)
    std = res_arr.std(axis=0)

    print("\n================ FINAL RESULTS (HGT) ================")
    print(
        f"AUC    : {mean[0]:.4f} ± {std[0]:.4f}\n"
        f"AUPR   : {mean[1]:.4f} ± {std[1]:.4f}\n"
        f"Prec   : {mean[2]:.4f} ± {std[2]:.4f}\n"
        f"Recall : {mean[3]:.4f} ± {std[3]:.4f}\n"
        f"F1     : {mean[4]:.4f} ± {std[4]:.4f}\n"
        f"MCC    : {mean[5]:.4f} ± {std[5]:.4f}"
    )
    print(f"Total time: {(end - start) / 60:.1f} minutes")

    # ذخیره خلاصهٔ نتایج تمام فولدها در CSV
    summary_path = os.path.join(save_dir, "hgt_results_summary.csv")
    np.savetxt(
        summary_path,
        res_arr,
        delimiter=",",
        fmt="%.6f",
        header="AUC,AUPR,Precision,Recall,F1,MCC",
        comments="",
    )
    print(f"[Saved] Fold-wise results summary -> {summary_path}")


if __name__ == "__main__":
    main()


