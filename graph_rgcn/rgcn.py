"""
R-GCN-based HGNN for ADR prediction on the MFAHGN / GRDTI dataset.

This script is self-contained and follows the same data pipeline as the
GraphMamba/HAN/HGT-based implementations, but replaces the encoder with an
R-GCN-style heterogeneous GNN over the DGL heterograph.

Usage (from project root, where ./newdata/ exists):
    python rgcn_adr_prediction.py

The script will:
  * load all relation matrices from ./newdata/
  * build the heterogeneous DGL graph
  * train the R-GCN-based model
  * use early stopping based on validation AUPR
  * for each fold, save the best model and metadata
  * save a CSV summary of all folds
  * print AUC / AUPR / Precision / Recall / F1 / MCC

All hyperparameters are defined in the Args dataclass below.
"""

import os
import time
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn

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
    patience: int = 20  # early stopping patience

    # embeddings (R-GCN hidden dim)
    dim_embedding: int = 256

    # evaluation / sampling
    n_fold: int = 3  # number of times to repeat training with same split


# =============================================================
# 2) Data loading (same format as original code)
# =============================================================


def load_relation_matrices(network_path: str = "./newdata/"):
    """Load all relation matrices from text files.

    Expected files (same as original GRDTI implementation):
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


def construct_hetero_graph(
    drug_drug,
    drug_disease,
    drug_protein,
    protein_protein,
    protein_disease,
    drug_sideeffect,
):
    """Construct DGL heterogeneous graph with the same schema as original code."""
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
# 3) R-GCN layers & model
# =============================================================


class HeteroRGCNLayer(nn.Module):
    """One R-GCN layer over a DGL heterograph."""

    def __init__(self, in_dim: int, out_dim: int, canonical_etypes, ntypes):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # relation-specific transforms
        self.weight = nn.ModuleDict()
        rel_names = set([rel for (_, rel, _) in canonical_etypes])
        for rel in rel_names:
            self.weight[rel] = nn.Linear(in_dim, out_dim, bias=False)

        # self-loop per node type
        self.self_loop = nn.ModuleDict()
        for ntype in ntypes:
            self.self_loop[ntype] = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, g: dgl.DGLHeteroGraph, h_dict):
        with g.local_scope():
            out_dict = {}
            # self-loop term
            for ntype, h in h_dict.items():
                out_dict[ntype] = self.self_loop[ntype](h)

            # message passing per relation
            for srctype, etype, dsttype in g.canonical_etypes:
                if etype not in self.weight:
                    continue
                sub_g = g[srctype, etype, dsttype]
                if sub_g.num_edges() == 0:
                    continue

                Wh = self.weight[etype](h_dict[srctype])
                sub_g.srcdata["h"] = Wh
                sub_g.update_all(
                    fn.copy_u("h", "m"),
                    fn.sum("m", "h_neigh"),
                )
                h_neigh = sub_g.dstdata["h_neigh"]

                if dsttype in out_dict:
                    out_dict[dsttype] = out_dict[dsttype] + h_neigh
                else:
                    out_dict[dsttype] = h_neigh

        return out_dict


class RGCNADR(nn.Module):
    """R-GCN-based heterogeneous GNN for ADR prediction.

    Uses the same reconstruction-based loss structure as GRDTI:
      - reconstruct DDI, drug-disease, drug-protein,
        PPI, protein-disease, drug-sideeffect
    """

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
        self.g = g
        self.device = torch.device(args.device)
        self.dim_embedding = args.dim_embedding
        self.reg_lambda = args.reg_lambda

        self.num_disease = n_disease
        self.num_drug = n_drug
        self.num_protein = n_protein
        self.num_sideeffect = n_sideeffect

        # trainable initial embeddings
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

        # two-layer R-GCN over the heterograph
        self.layers = nn.ModuleList(
            [
                HeteroRGCNLayer(
                    self.dim_embedding,
                    self.dim_embedding,
                    g.canonical_etypes,
                    g.ntypes,
                ),
                HeteroRGCNLayer(
                    self.dim_embedding,
                    self.dim_embedding,
                    g.canonical_etypes,
                    g.ntypes,
                ),
            ]
        )

        # diagonal reconstruction parameters
        tmp = torch.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_P = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Side = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_PPI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_P_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(
        self,
        drug_drug: torch.Tensor,
        drug_disease: torch.Tensor,
        drug_protein: torch.Tensor,
        protein_protein: torch.Tensor,
        protein_disease: torch.Tensor,
        drug_sideeffect: torch.Tensor,
        drug_sideeffect_mask: torch.Tensor,
    ):
        # initial node features
        h_dict = {
            "disease": self.disease_feat,
            "drug": self.drug_feat,
            "protein": self.protein_feat,
            "sideeffect": self.sideeffect_feat,
        }

        # R-GCN message passing
        for layer in self.layers:
            h_dict = layer(self.g, h_dict)
            for ntype in h_dict:
                h_dict[ntype] = F.relu(h_dict[ntype])

        disease_embedding = h_dict["disease"]
        drug_embedding = h_dict["drug"]
        protein_embedding = h_dict["protein"]
        sideeffect_embedding = h_dict["sideeffect"]

        disease_vec = l2_norm(disease_embedding)
        drug_vec = l2_norm(drug_embedding)
        protein_vec = l2_norm(protein_embedding)
        sideeffect_vec = l2_norm(sideeffect_embedding)

        # reconstruct relations
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
# 4) Training & evaluation (with best-model saving)
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

    # full drug-sideeffect matrix and mask
    drug_sideeffect = torch.zeros((num_drug, num_sideeffect), dtype=torch.float32)
    mask = torch.zeros_like(drug_sideeffect, device=device)
    for d, s, label in DSE_train:
        drug_sideeffect[d, s] = float(label)
        mask[d, s] = 1.0

    # hetero graph
    g = construct_hetero_graph(
        drug_drug,
        drug_disease,
        drug_protein,
        protein_protein,
        protein_disease,
        drug_sideeffect.numpy(),
    ).to(device)

    # tensors on device
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

    model = RGCNADR(
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

    # paths for best model & info of this fold
    best_model_path = os.path.join(save_dir, f"rgcn_best_fold{fold_index+1}.pt")
    best_info_path = os.path.join(save_dir, f"rgcn_best_fold{fold_index+1}_info.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, dse_loss, l2_loss, dse_rec, dse_pot = model(
            drug_drug_t,
            drug_disease_t,
            drug_protein_t,
            protein_protein_t,
            protein_disease_t,
            drug_sideeffect_t,
            mask.to(device),
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            scores = dse_rec.detach().cpu().numpy()

            # validation
            val_scores = [scores[d, s] for d, s, _ in DSE_valid]
            val_labels = [int(lbl) for _, _, lbl in DSE_valid]

            valid_auc = roc_auc_score(val_labels, val_scores)
            valid_aupr = average_precision_score(val_labels, val_scores)

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

            # early stopping + best-model saving based on validation AUPR
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0

                test_scores = [scores[d, s] for d, s, _ in DSE_test]
                test_labels = [int(lbl) for _, _, lbl in DSE_test]

                best_test_auc = roc_auc_score(test_labels, test_scores)
                best_test_aupr = average_precision_score(test_labels, test_scores)

                # save best model state & metadata
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
# 5) Main entry
# =============================================================


def main():
    args = Args()
    print("[Config]", args)

    # directory for saving models and results
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

    print("\n================ FINAL RESULTS (R-GCN) ================")
    print(
        f"AUC    : {mean[0]:.4f} ± {std[0]:.4f}\n"
        f"AUPR   : {mean[1]:.4f} ± {std[1]:.4f}\n"
        f"Prec   : {mean[2]:.4f} ± {std[2]:.4f}\n"
        f"Recall : {mean[3]:.4f} ± {std[3]:.4f}\n"
        f"F1     : {mean[4]:.4f} ± {std[4]:.4f}\n"
        f"MCC    : {mean[5]:.4f} ± {std[5]:.4f}"
    )
    print(f"Total time: {(end - start) / 60:.1f} minutes")

    # save fold-wise summary as CSV
    summary_path = os.path.join(save_dir, "rgcn_results_summary.csv")
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
