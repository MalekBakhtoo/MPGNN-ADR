"""GraphGPS-style HGNN for ADR prediction on the MFAHGN / GRDTI dataset.

این نسخه:
  * کاملاً PyTorch است (بدون DGL)
  * گراف‌ها فقط به‌صورت ماتریس مجاورت (adjacency matrix) استفاده می‌شوند.
  * local: متاپث‌های کوتاه با LocalMetaPathEncoder
  * global: یک TransformerEncoder (الهام‌گرفته از GraphGPS)

Usage (from project root, where ./newdata/ exists):
    python graph_gps_adr_prediction.py

The script will:
  * load all relation matrices from ./newdata/
  * build meta-path adjacency matrices
  * train the GraphGPS-HGNN model (dense adjacency, no DGL)
  * print AUC / AUPR / Precision / Recall / F1 / MCC

All hyperparameters are defined in the Args class below.
"""

import os
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split


# =============================================================
# 1) Utilities: normalization and config
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
    patience: int = 20

    # embeddings & model
    dim_embedding: int = 256
    transformer_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2

    # evaluation / sampling
    n_fold: int = 3  # number of times to repeat training with the same split


# =============================================================
# 2) Data loading (same format as original HAN/GRDTI code)
# =============================================================


def load_drug_sideeffect_mapping(excel_path: str = "./data/takeToDict_data.xlsx"):
    """Load mapping: drug name -> index, side-effect index -> name.

    This is only needed if you want to inspect predicted side-effects
    per drug. It is optional for core training.
    """
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
    """Load all relation matrices from text files.

    Expected files (same as original GRDTI / HAN implementation):
      - mat_drug_drug.txt
      - mat_drug_disease.txt
      - mat_drug_se.txt
      - mat_protein_protein.txt
      - mat_protein_disease.txt
      - mat_drug_protein.txt
    """
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


# =============================================================
# 3) Local Meta-path Encoder (GraphGPS local part)
# =============================================================


class LocalMetaPathEncoder(nn.Module):
    """Local encoder based on short meta-paths.

    Given:
      * node features x (N, d)
      * list of adjacency matrices [A1, A2, ..., AM]
    It aggregates neighbors along each meta-path and fuses them.
    This plays the role of the local message passing in GraphGPS.
    """

    def __init__(self, dim_in: int, num_metapaths: int, dim_hidden: int):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.linears = nn.ModuleList(
            [nn.Linear(dim_in, dim_hidden) for _ in range(num_metapaths)]
        )
        self.out = nn.Linear(dim_hidden, dim_in)

    def forward(self, x: torch.Tensor, adj_list):
        # x: (N, d)
        outs = []
        for k, adj in enumerate(adj_list):
            # normalize adjacency row-wise
            rowsum = adj.sum(-1, keepdim=True) + 1e-12
            adj_norm = adj / rowsum
            neigh = adj_norm @ x  # (N, d)
            h = self.linears[k](neigh)
            h = torch.relu(h)
            outs.append(h)
        if len(outs) == 0:
            h_sum = x
        else:
            h_sum = torch.stack(outs, dim=0).mean(dim=0)
        out = self.out(h_sum)
        return out


# =============================================================
# 4) GraphGPS-HGNN model (ADR predictor, بدون DGL)
# =============================================================


class GraphGPSADR(nn.Module):
    """GraphGPS-style heterogeneous GNN for ADR prediction.

    ساختار کلی:
      * برای هر نوع نود (Drug/Disease/Protein/SideEffect) یک LocalMetaPathEncoder داریم
        که متاپث‌های کوتاه مربوط به همان نوع نود را تجمیع می‌کند.
      * همه نودها را پشت سر هم می‌چینیم و یک نوع embedding (type embedding)
        به آن‌ها اضافه می‌کنیم.
      * سپس یک TransformerEncoder روی کل نودها اعمال می‌کنیم (global attention).
      * در پایان embedding نهایی را به چهار بخش جدا می‌کنیم و مثل MFAHGN
        همه ماتریس‌های روابط را بازسازی کرده و loss را حساب می‌کنیم.
    """

    def __init__(
        self,
        n_disease: int,
        n_drug: int,
        n_protein: int,
        n_sideeffect: int,
        args: Args,
    ):
        super().__init__()
        self.device = torch.device(args.device)
        self.dim_embedding = args.dim_embedding
        self.reg_lambda = args.reg_lambda

        self.num_disease = n_disease
        self.num_drug = n_drug
        self.num_protein = n_protein
        self.num_sideeffect = n_sideeffect

        # trainable initial embeddings for each node type
        self.disease_feat = nn.Parameter(
            torch.randn(n_disease, self.dim_embedding) * 0.1
        )
        self.drug_feat = nn.Parameter(torch.randn(n_drug, self.dim_embedding) * 0.1)
        self.protein_feat = nn.Parameter(
            torch.randn(n_protein, self.dim_embedding) * 0.1
        )
        self.sideeffect_feat = nn.Parameter(
            torch.randn(n_sideeffect, self.dim_embedding) * 0.1
        )

        # Local meta-path encoders per node type
        # disease: D-D (via drugs), D-D (via proteins)
        self.local_disease = LocalMetaPathEncoder(
            self.dim_embedding, num_metapaths=2, dim_hidden=self.dim_embedding
        )
        # drug: via drug-drug, drug-disease-drug, drug-sideeffect-drug, drug-protein-drug
        self.local_drug = LocalMetaPathEncoder(
            self.dim_embedding, num_metapaths=4, dim_hidden=self.dim_embedding
        )
        # protein: via PPI, protein-disease-protein, protein-drug-protein
        self.local_protein = LocalMetaPathEncoder(
            self.dim_embedding, num_metapaths=3, dim_hidden=self.dim_embedding
        )
        # side-effect: via sideeffect-drug-sideeffect
        self.local_sideeffect = LocalMetaPathEncoder(
            self.dim_embedding, num_metapaths=1, dim_hidden=self.dim_embedding
        )

        # type embeddings for global transformer
        self.type_emb = nn.Embedding(4, self.dim_embedding)  # 0: disease, 1: drug, 2: protein, 3: sideeffect

        # TransformerEncoder as global GraphGPS-style block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_embedding,
            nhead=args.n_heads,
            dim_feedforward=self.dim_embedding * 4,
            dropout=args.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_layers)

        # diagonal reconstruction parameters (like MFAHGN)
        tmp = torch.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_P = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Side = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_PPI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_P_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))

    def forward(
        self,
        drug_drug,
        drug_disease_drug,
        drug_protein_drug,
        drug_sideeffect_drug,
        protein_protein,
        protein_drug_protein,
        protein_disease_protein,
        disease_drug_disease,
        disease_protein_disease,
        sideeffect_drug_sideeffect,
        drug_sideeffect_mask,
        drug_disease,
        drug_protein,
        protein_disease,
        drug_sideeffect,
    ):
        # ----- Local meta-path encoding per node type -----
        disease_local = self.local_disease(
            self.disease_feat,
            [disease_drug_disease, disease_protein_disease],
        )
        drug_local = self.local_drug(
            self.drug_feat,
            [drug_drug, drug_disease_drug, drug_sideeffect_drug, drug_protein_drug],
        )
        protein_local = self.local_protein(
            self.protein_feat,
            [protein_protein, protein_disease_protein, protein_drug_protein],
        )
        sideeffect_local = self.local_sideeffect(
            self.sideeffect_feat, [sideeffect_drug_sideeffect]
        )

        # concat all node embeddings
        node_local = torch.cat(
            [disease_local, drug_local, protein_local, sideeffect_local], dim=0
        )  # (N_total, d)

        # build type index
        nDi, nDr, nPr, nSe = (
            self.num_disease,
            self.num_drug,
            self.num_protein,
            self.num_sideeffect,
        )
        type_idx = torch.tensor(
            [0] * nDi + [1] * nDr + [2] * nPr + [3] * nSe,
            dtype=torch.long,
            device=node_local.device,
        )
        type_vec = self.type_emb(type_idx)

        # GraphGPS global transformer: local + type embedding as input
        x_in = node_local + type_vec
        x_in = x_in.unsqueeze(0)  # (1, N_total, d)
        x_tr = self.transformer(x_in).squeeze(0)  # (N_total, d)

        # residual connection from local part
        node_final = x_tr + node_local

        # split back per type
        disease_embedding = node_final[:nDi]
        drug_embedding = node_final[nDi : nDi + nDr]
        protein_embedding = node_final[nDi + nDr : nDi + nDr + nPr]
        sideeffect_embedding = node_final[-nSe:]

        disease_vec = l2_norm(disease_embedding)
        drug_vec = l2_norm(drug_embedding)
        protein_vec = l2_norm(protein_embedding)
        sideeffect_vec = l2_norm(sideeffect_embedding)

        # ----- reconstruct relations -----
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

        # L2 regularization over all parameters (except biases)
        l2 = torch.tensor(0.0, device=self.device)
        for name, p in self.named_parameters():
            if "bias" not in name:
                l2 = l2 + torch.sum(p.pow(2))
        l2 = 0.5 * l2

        total_loss = loss_dse + 1.0 * other_loss + self.reg_lambda * l2

        return total_loss, loss_dse, l2, drug_sideeffect_rec, dse_potential


# =============================================================
# 5) Training & evaluation (pure PyTorch)
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
):
    device = torch.device(args.device)

    num_disease = drug_disease.shape[1]
    num_drug = drug_drug.shape[0]
    num_protein = protein_protein.shape[0]

    # build full drug-sideeffect matrix for reconstruction
    drug_sideeffect = torch.zeros((num_drug, num_sideeffect), dtype=torch.float32)
    mask = torch.zeros_like(drug_sideeffect, device=device)
    for d, s, label in DSE_train:
        drug_sideeffect[d, s] = float(label)
        mask[d, s] = 1.0

    # convert all adjacency-like matrices to tensors on device
    drug_drug_t = torch.tensor(drug_drug, dtype=torch.float32, device=device)
    drug_disease_t = torch.tensor(drug_disease, dtype=torch.float32, device=device)
    drug_protein_t = torch.tensor(drug_protein, dtype=torch.float32, device=device)
    protein_protein_t = torch.tensor(
        protein_protein, dtype=torch.float32, device=device
    )
    protein_disease_t = torch.tensor(
        protein_disease, dtype=torch.float32, device=device
    )
    drug_sideeffect_t = drug_sideeffect.to(device)

    # meta-path adjacency (dense) ---------------------------------
    drug_disease_drug = (drug_disease_t @ drug_disease_t.t()).fill_diagonal_(0)
    drug_protein_drug = (drug_protein_t @ drug_protein_t.t()).fill_diagonal_(0)
    drug_sideeffect_drug = (drug_sideeffect_t @ drug_sideeffect_t.t()).fill_diagonal_(0)
    protein_drug_protein = (drug_protein_t.t() @ drug_protein_t).fill_diagonal_(0)
    protein_disease_protein = (protein_disease_t @ protein_disease_t.t()).fill_diagonal_(0)
    disease_drug_disease = (drug_disease_t.t() @ drug_disease_t).fill_diagonal_(0)
    disease_protein_disease = (protein_disease_t.t() @ protein_disease_t).fill_diagonal_(0)
    sideeffect_drug_sideeffect = (
        drug_sideeffect_t.t() @ drug_sideeffect_t
    ).fill_diagonal_(0)

    model = GraphGPSADR(
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, dse_loss, l2_loss, dse_rec, dse_pot = model(
            drug_drug_t,
            drug_disease_drug,
            drug_protein_drug,
            drug_sideeffect_drug,
            protein_protein_t,
            protein_drug_protein,
            protein_disease_protein,
            disease_drug_disease,
            disease_protein_disease,
            sideeffect_drug_sideeffect,
            mask.to(device),
            drug_disease_t,
            drug_protein_t,
            protein_disease_t,
            drug_sideeffect_t,
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # evaluation --------------------------------------------------
        model.eval()
        with torch.no_grad():
            scores = dse_rec.detach().cpu().numpy()

            # validation
            val_scores = [scores[d, s] for d, s, _ in DSE_valid]
            val_labels = [int(lbl) for _, _, lbl in DSE_valid]

            valid_auc = roc_auc_score(val_labels, val_scores)
            valid_aupr = average_precision_score(val_labels, val_scores)

            # binarize at 0.5 for P/R/F1/MCC
            val_pred_labels = (np.array(val_scores) >= 0.5).astype(int)
            precision = precision_score(val_labels, val_pred_labels, zero_division=0)
            recall = recall_score(val_labels, val_pred_labels, zero_division=0)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            try:
                mcc = matthews_corrcoef(val_labels, val_pred_labels)
            except Exception:
                mcc = 0.0

            precision_hist.append(precision)
            recall_hist.append(recall)
            f1_hist.append(f1)
            mcc_hist.append(mcc)

            # test evaluation only when validation improves
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0

                test_scores = [scores[d, s] for d, s, _ in DSE_test]
                test_labels = [int(lbl) for _, _, lbl in DSE_test]

                # AUC & AUPR
                best_test_auc = roc_auc_score(test_labels, test_scores)
                best_test_aupr = average_precision_score(test_labels, test_scores)
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
# 6) Main entry
# =============================================================


def main():
    args = Args()
    print("[Config]", args)
    (
        drug_drug,
        drug_disease,
        drug_protein,
        protein_protein,
        protein_disease,
        drug_sideeffect,
    ) = load_relation_matrices("./newdata/")

    num_sideeffect = drug_sideeffect.shape[1]

    # build positive / negative pairs (pos:neg = 1:10)
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

    # split once into train/valid/test
    train, DSE_test = train_test_split(dataset, test_size=0.1, random_state=42)
    DSE_train, DSE_valid = train_test_split(train, test_size=0.1, random_state=42)

    # to numpy int64 triplets
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
        )
        results.append((auc, aupr, p_mean, r_mean, f1_mean, mcc_mean))

    end = time.time()

    res_arr = np.array(results)
    mean = res_arr.mean(axis=0)
    std = res_arr.std(axis=0)

    print("\n================ FINAL RESULTS ================")
    print(
        f"AUC    : {mean[0]:.4f} ± {std[0]:.4f}\n"
        f"AUPR   : {mean[1]:.4f} ± {std[1]:.4f}\n"
        f"Prec   : {mean[2]:.4f} ± {std[2]:.4f}\n"
        f"Recall : {mean[3]:.4f} ± {std[3]:.4f}\n"
        f"F1     : {mean[4]:.4f} ± {std[4]:.4f}\n"
        f"MCC    : {mean[5]:.4f} ± {std[5]:.4f}"
    )
    print(f"Total time: {(end - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
