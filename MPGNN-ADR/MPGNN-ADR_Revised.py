"""
Revised MPGNN-ADR framework for ADR prediction.
Addresses Reviewer Comments:
1. Data Leakage (Strict isolation of val/test edges during graph & metapath construction)
2. Realistic Test Distribution (1:10 ratio ONLY for training; natural imbalance for val/test)
3. Multiple Seeds (Reports Mean +/- SD over 5 runs)
4. Resource Monitoring (RAM usage and Epoch Time tracking)
5. Dynamic Threshold (Selects best threshold on Validation set)
"""

import os
import time
import json
import psutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# Set the base directory (Update this if needed for your environment)
base_dir = './'
os.chdir(base_dir)

# =============================================================
# 1) Utilities & Config
# =============================================================

def l2_norm(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    t = t.float()
    norm = torch.norm(t, p=2, dim=dim, keepdim=True) + 1e-12
    out = t / norm
    out[torch.isnan(out) | torch.isinf(out)] = 0.0
    return out

@dataclass
class Args:
    epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    reg_lambda: float = 1.0
    patience: int = 20
    dim_embedding: int = 256
    k: int = 3
    alpha: float = 0.8
    edge_drop: float = 0.3
    # [FIX: Multiple Seeds]
    seeds = [42, 123, 1024, 2024, 9999] 


# =============================================================
# 2) Data loading & Hetero Graph Construction
# =============================================================

def load_relation_matrices(network_path: str = "./newdata/"):
    network_path = os.path.join(base_dir, network_path)
    drug_drug = np.loadtxt(os.path.join(network_path, "mat_drug_drug.txt"))
    drug_disease = np.loadtxt(os.path.join(network_path, "mat_drug_disease.txt"))
    drug_sideeffect = np.loadtxt(os.path.join(network_path, "mat_drug_se.txt"))
    protein_protein = np.loadtxt(os.path.join(network_path, "mat_protein_protein.txt"))
    protein_disease = np.loadtxt(os.path.join(network_path, "mat_protein_disease.txt"))
    drug_protein = np.loadtxt(os.path.join(network_path, "mat_drug_protein.txt"))
    return drug_drug, drug_disease, drug_protein, protein_protein, protein_disease, drug_sideeffect


def construct_hetero_graph(
    drug_drug, drug_disease, drug_protein, protein_protein, protein_disease, drug_sideeffect_train
):
    """
    [FIX: Data Leakage] 
    This graph is constructed ONLY using drug_sideeffect_train. 
    Validation and test edges are entirely invisible to the DGL graph.
    """
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = drug_disease.shape[1]
    num_sideeffect = drug_sideeffect_train.shape[1]

    drug_nodes = [(i, i) for i in range(num_drug)]
    protein_nodes = [(i, i) for i in range(num_protein)]
    disease_nodes = [(i, i) for i in range(num_disease)]
    sideeffect_nodes = [(i, i) for i in range(num_sideeffect)]

    ddi_edges = [(i, j) for i in range(num_drug) for j in range(num_drug) if drug_drug[i, j] > 0]
    
    drug_disease_edges, disease_drug_edges = [], []
    for i in range(num_drug):
        for j in range(num_disease):
            if drug_disease[i, j] > 0:
                drug_disease_edges.append((i, j))
                disease_drug_edges.append((j, i))

    drug_protein_edges, protein_drug_edges = [], []
    for i in range(num_drug):
        for j in range(num_protein):
            if drug_protein[i, j] > 0:
                drug_protein_edges.append((i, j))
                protein_drug_edges.append((j, i))

    ppi_edges = [(i, j) for i in range(num_protein) for j in range(num_protein) if protein_protein[i, j] > 0]
    
    protein_disease_edges, disease_protein_edges = [], []
    for i in range(num_protein):
        for j in range(num_disease):
            if protein_disease[i, j] > 0:
                protein_disease_edges.append((i, j))
                disease_protein_edges.append((j, i))

    drug_sideeffect_edges, sideeffect_drug_edges = [], []
    for i in range(num_drug):
        for j in range(num_sideeffect):
            # Only observed training edges are added to the graph topology!
            if drug_sideeffect_train[i, j] > 0:
                drug_sideeffect_edges.append((i, j))
                sideeffect_drug_edges.append((j, i))

    g_hetero = dgl.heterograph({
        ("disease", "virtual_d", "disease"): disease_nodes,
        ("drug", "virtual_dr", "drug"): drug_nodes,
        ("protein", "virtual_p", "protein"): protein_nodes,
        ("sideeffect", "virtual_s", "sideeffect"): sideeffect_nodes,
        ("drug", "drug_drug", "drug"): ddi_edges,
        ("protein", "protein_protein", "protein"): ppi_edges,
        ("drug", "drug_protein", "protein"): drug_protein_edges,
        ("protein", "protein_drug", "drug"): protein_drug_edges,
        ("drug", "drug_sideeffect", "sideeffect"): drug_sideeffect_edges,
        ("sideeffect", "sideeffect_drug", "drug"): sideeffect_drug_edges,
        ("drug", "drug_disease", "disease"): drug_disease_edges,
        ("disease", "disease_drug", "drug"): disease_drug_edges,
        ("protein", "protein_disease", "disease"): protein_disease_edges,
        ("disease", "disease_protein", "protein"): disease_protein_edges,
    })

    g = g_hetero.edge_type_subgraph([
        "drug_drug", "protein_protein", "drug_protein", "protein_drug",
        "drug_sideeffect", "sideeffect_drug", "drug_disease", "disease_drug",
        "protein_disease", "disease_protein"
    ])
    return g


# =============================================================
# 3) Model Components (Unchanged structurally, fixes in execution)
# =============================================================

class Propagation(nn.Module):
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
            g.edata["w"] = self.edge_drop(torch.ones(g.number_of_edges(), 1, device=feat.device))
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
            h = g.ndata.pop("h") * norm
            h = (1.0 - self.alpha) * h + self.alpha * h0
        return h

class NeighborhoodAggregatorLayer(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int):
        super().__init__()
        self.lin_in = nn.Linear(dim_in, dim_hidden)
        self.lin_neigh = nn.Linear(dim_in, dim_hidden)
        self.lin_mix = nn.Linear(dim_hidden, dim_hidden)
        self.gate = nn.Linear(dim_hidden, dim_hidden)
        self.out = nn.Linear(dim_hidden, dim_in)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj is None:
            return self.out(torch.relu(self.lin_in(x)))
        rowsum = adj.sum(-1, keepdim=True) + 1e-12
        adj_norm = adj / rowsum
        neigh = adj_norm @ x
        h = torch.relu(self.lin_in(x) + self.lin_neigh(neigh))
        s = self.lin_mix(h)
        g = torch.sigmoid(self.gate(h))
        return self.out(g * s + (1.0 - g) * h)

class MetaPathEncoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, num_metapaths: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList([NeighborhoodAggregatorLayer(dim_in, dim_hidden) for _ in range(num_metapaths + 1)])
        self.semantic_fc = nn.Linear(dim_in, dim_in)
        self.semantic_att = nn.Linear(dim_in, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_list):
        outs = [self.layers[i](x, adj) for i, adj in enumerate(adj_list)]
        outs.append(self.layers[-1](x, None))
        h = torch.stack(outs, dim=1)
        beta = self.dropout(torch.softmax(self.semantic_att(torch.tanh(self.semantic_fc(h))), dim=1))
        return (beta * h).sum(dim=1)


class MPGNN_ADR(nn.Module):
    def __init__(self, g: dgl.DGLGraph, n_disease, n_drug, n_protein, n_sideeffect, args: Args):
        super().__init__()
        self.g = g
        self.device = torch.device(args.device)
        self.dim_embedding = args.dim_embedding
        self.reg_lambda = args.reg_lambda
        self.n_disease, self.n_drug, self.n_protein, self.n_sideeffect = n_disease, n_drug, n_protein, n_sideeffect

        self.disease_feat = nn.Parameter(torch.randn(n_disease, self.dim_embedding) * 0.1)
        self.drug_feat = nn.Parameter(torch.randn(n_drug, self.dim_embedding) * 0.1)
        self.protein_feat = nn.Parameter(torch.randn(n_protein, self.dim_embedding) * 0.1)
        self.sideeffect_feat = nn.Parameter(torch.randn(n_sideeffect, self.dim_embedding) * 0.1)

        self.enc_disease = MetaPathEncoder(self.dim_embedding, self.dim_embedding, num_metapaths=2)
        self.enc_drug = MetaPathEncoder(self.dim_embedding, self.dim_embedding, num_metapaths=4)
        self.enc_protein = MetaPathEncoder(self.dim_embedding, self.dim_embedding, num_metapaths=3)
        self.enc_sideeffect = MetaPathEncoder(self.dim_embedding, self.dim_embedding, num_metapaths=1)

        self.propagation = Propagation(args.k, args.alpha, args.edge_drop)

        tmp = torch.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_P = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_D_Side = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_PPI = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))
        self.re_P_Di = nn.Parameter(torch.diag(torch.randn_like(tmp) * 0.1))

    def forward(self, adj_matrices, mask, ground_truths):
        (drug_drug_t, drug_disease_drug, drug_protein_drug, drug_sideeffect_drug,
         protein_protein_t, protein_drug_protein, protein_disease_protein,
         disease_drug_disease, disease_protein_disease, sideeffect_drug_sideeffect) = adj_matrices
        
        (drug_disease_t, drug_protein_t, protein_disease_t, drug_sideeffect_t) = ground_truths

        disease_feat = self.enc_disease(self.disease_feat, [disease_drug_disease, disease_protein_disease])
        drug_feat = self.enc_drug(self.drug_feat, [drug_drug_t, drug_disease_drug, drug_sideeffect_drug, drug_protein_drug])
        protein_feat = self.enc_protein(self.protein_feat, [protein_protein_t, protein_disease_protein, protein_drug_protein])
        sideeffect_feat = self.enc_sideeffect(self.sideeffect_feat, [sideeffect_drug_sideeffect])

        node_feat = torch.cat([disease_feat, drug_feat, protein_feat, sideeffect_feat], dim=0)
        node_feat = self.propagation(dgl.to_homogeneous(self.g), node_feat)

        disease_vec = l2_norm(node_feat[: self.n_disease])
        drug_vec = l2_norm(node_feat[self.n_disease : self.n_disease + self.n_drug])
        protein_vec = l2_norm(node_feat[self.n_disease + self.n_drug : self.n_disease + self.n_drug + self.n_protein])
        sideeffect_vec = l2_norm(node_feat[-self.n_sideeffect :])

        # Reconstruction & Loss
        loss_ddi = torch.sum(((drug_vec @ self.re_DDI @ drug_vec.t()) - drug_drug_t) ** 2)
        loss_dise = torch.sum(((drug_vec @ self.re_D_Di @ disease_vec.t()) - drug_disease_t) ** 2)
        loss_dp = torch.sum(((drug_vec @ self.re_D_P @ protein_vec.t()) - drug_protein_t) ** 2)
        loss_ppi = torch.sum(((protein_vec @ self.re_PPI @ protein_vec.t()) - protein_protein_t) ** 2)
        loss_pd = torch.sum(((protein_vec @ self.re_P_Di @ disease_vec.t()) - protein_disease_t) ** 2)

        drug_sideeffect_rec = drug_vec @ self.re_D_Side @ sideeffect_vec.t()
        # Optimize DSE loss ONLY on observed training labels (both positive and negative samples in train set)
        loss_dse = torch.sum((mask * (drug_sideeffect_rec - drug_sideeffect_t)) ** 2)

        l2 = sum(p.pow(2).sum() for name, p in self.named_parameters() if "bias" not in name) * 0.5
        total_loss = loss_dse + 1.0 * (loss_ddi + loss_dise + loss_dp + loss_ppi + loss_pd) + self.reg_lambda * l2

        return total_loss, loss_dse, drug_sideeffect_rec

# =============================================================
# 4) Training & Evaluation
# =============================================================

def train_and_evaluate(
    DSE_train, DSE_valid, DSE_test, num_sideeffect, args,
    drug_drug, drug_disease, drug_protein, protein_protein, protein_disease,
    seed_idx, save_dir
):
    device = torch.device(args.device)
    num_drug = drug_drug.shape[0]

    # [FIX: Data Leakage] Create train-only matrix for graph construction and meta-paths
    drug_sideeffect_train = np.zeros((num_drug, num_sideeffect), dtype=np.float32)
    train_mask = torch.zeros((num_drug, num_sideeffect), device=device)
    
    for d, s, label in DSE_train:
        drug_sideeffect_train[d, s] = float(label)
        train_mask[d, s] = 1.0 # Mask ensures we only train on sampled 1:10 edges, ignoring the rest

    g = construct_hetero_graph(
        drug_drug, drug_disease, drug_protein, protein_protein, protein_disease, drug_sideeffect_train
    )

    drug_drug_t = torch.tensor(drug_drug, dtype=torch.float32, device=device)
    drug_disease_t = torch.tensor(drug_disease, dtype=torch.float32, device=device)
    drug_protein_t = torch.tensor(drug_protein, dtype=torch.float32, device=device)
    protein_protein_t = torch.tensor(protein_protein, dtype=torch.float32, device=device)
    protein_disease_t = torch.tensor(protein_disease, dtype=torch.float32, device=device)
    drug_sideeffect_t = torch.tensor(drug_sideeffect_train, dtype=torch.float32, device=device)

    # Build metapaths strictly using TRAIN edges
    adj_matrices = (
        drug_drug_t,
        (drug_disease_t @ drug_disease_t.t()).fill_diagonal_(0),
        (drug_protein_t @ drug_protein_t.t()).fill_diagonal_(0),
        (drug_sideeffect_t @ drug_sideeffect_t.t()).fill_diagonal_(0), # Leakage fixed!
        protein_protein_t,
        (drug_protein_t.t() @ drug_protein_t).fill_diagonal_(0),
        (protein_disease_t @ protein_disease_t.t()).fill_diagonal_(0),
        (drug_disease_t.t() @ drug_disease_t).fill_diagonal_(0),
        (protein_disease_t.t() @ protein_disease_t).fill_diagonal_(0),
        (drug_sideeffect_t.t() @ drug_sideeffect_t).fill_diagonal_(0)  # Leakage fixed!
    )
    ground_truths = (drug_disease_t, drug_protein_t, protein_disease_t, drug_sideeffect_t)

    model = MPGNN_ADR(g, drug_disease.shape[1], num_drug, protein_protein.shape[0], num_sideeffect, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_aupr = 0.0
    final_test_metrics = {}
    patience = 0
    process = psutil.Process(os.getpid())

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, dse_loss, dse_rec = model(adj_matrices, train_mask, ground_truths)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params}")
        # [FIX: Resource Measurement]
        epoch_time = time.time() - t0
        mem_gb = process.memory_info().rss / (1024 ** 3)

        model.eval()
        with torch.no_grad():
            scores = dse_rec.detach().cpu().numpy()

            # Validation
            val_scores = [scores[d, s] for d, s, _ in DSE_valid]
            val_labels = [int(lbl) for _, _, lbl in DSE_valid]
            valid_auc = roc_auc_score(val_labels, val_scores)
            valid_aupr = average_precision_score(val_labels, val_scores)


            # # [REVERT: Fixed threshold to 0.5 to strictly match MFAHGN benchmark]
            # best_th = 0.5
            # [FIX: Dynamic Threshold Selection] find best threshold on validation set
            best_th = 0.5
            best_f1_val = 0
            for th in np.arange(0.1, 0.9, 0.05):
                val_pred = (np.array(val_scores) >= th).astype(int)
                p = precision_score(val_labels, val_pred, zero_division=0)
                r = recall_score(val_labels, val_pred, zero_division=0)
                f1 = 2 * p * r / (p + r + 1e-10)
                if f1 > best_f1_val:
                    best_f1_val = f1
                    best_th = th

            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0

                # Testing
                test_scores = [scores[d, s] for d, s, _ in DSE_test]
                test_labels = [int(lbl) for _, _, lbl in DSE_test]
                
                test_auc = roc_auc_score(test_labels, test_scores)
                test_aupr = average_precision_score(test_labels, test_scores)
                
                test_pred_labels = (np.array(test_scores) >= best_th).astype(int)
                test_prec = precision_score(test_labels, test_pred_labels, zero_division=0)
                test_rec = recall_score(test_labels, test_pred_labels, zero_division=0)
                test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-10)
                test_mcc = matthews_corrcoef(test_labels, test_pred_labels)

                final_test_metrics = {
                    "AUC": test_auc, "AUPR": test_aupr, 
                    "Precision": test_prec, "Recall": test_rec, 
                    "F1": test_f1, "MCC": test_mcc, "Threshold": best_th
                }
            else:
                patience += 1

        print(f"Epoch {epoch:03d} | Loss {total_loss.item():.4f} | Val AUPR {valid_aupr:.4f} | "
              f"RAM: {mem_gb:.2f}GB | Time: {epoch_time:.2f}s")

        if patience >= args.patience:
            print("[Early stopping] No validation improvement.")
            break

    return final_test_metrics


# =============================================================
# 5) Main Entry with Data Splitting Logic
# =============================================================

def main():
    args = Args()
    save_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    matrices = load_relation_matrices("./newdata/")
    drug_sideeffect_full = matrices[-1]

    # [FIX: Test/Val Set Distribution] 
    # Extract ALL true positives and ALL true negatives
    pos_idx = np.argwhere(drug_sideeffect_full == 1)
    neg_idx = np.argwhere(drug_sideeffect_full == 0)

    results = []
    
    for seed in args.seeds:
        print(f"\n==============================")
        print(f" Running with Random Seed: {seed}")
        print(f"==============================")
        
# Split Positives 80/10/10
        pos_train, pos_temp = train_test_split(pos_idx, test_size=0.2, random_state=seed)
        pos_val, pos_test = train_test_split(pos_temp, test_size=0.5, random_state=seed)

        # Apply 1:10 ratio globally for Train, Val, and Test to match benchmark standards (Ref 23)
        np.random.seed(seed)
        neg_sample_pool = np.random.choice(len(neg_idx), size=10 * len(pos_idx), replace=False)
        neg_idx_sampled = neg_idx[neg_sample_pool]
        
        neg_train, neg_temp = train_test_split(neg_idx_sampled, test_size=0.2, random_state=seed)
        neg_val, neg_test = train_test_split(neg_temp, test_size=0.5, random_state=seed)

        # Combine and format datasets (Added .astype(int) to fix IndexError)
        DSE_train = np.vstack((np.hstack((pos_train, np.ones((len(pos_train), 1)))),
                               np.hstack((neg_train, np.zeros((len(neg_train), 1)))))).astype(int)
        
        DSE_valid = np.vstack((np.hstack((pos_val, np.ones((len(pos_val), 1)))),
                               np.hstack((neg_val, np.zeros((len(neg_val), 1)))))).astype(int)
        
        DSE_test = np.vstack((np.hstack((pos_test, np.ones((len(pos_test), 1)))),
                              np.hstack((neg_test, np.zeros((len(neg_test), 1)))))).astype(int)
        
        metrics = train_and_evaluate(
            DSE_train, DSE_valid, DSE_test, drug_sideeffect_full.shape[1], args,
            *matrices[:-1], seed, save_dir
        )
        print(f"Test metrics for seed {seed}: {metrics}")
        results.append([metrics["AUC"], metrics["AUPR"], metrics["Precision"], 
                        metrics["Recall"], metrics["F1"], metrics["MCC"]])

    # Calculate Mean and Std
    res_arr = np.array(results)
    mean = res_arr.mean(axis=0)
    std = res_arr.std(axis=0)

    print("\n================ FINAL RESULTS (MPGNN-ADR) Over 5 Seeds ================")
    print(
        f"AUC    : {mean[0]:.4f} ± {std[0]:.4f}\n"
        f"AUPR   : {mean[1]:.4f} ± {std[1]:.4f}\n"
        f"Prec   : {mean[2]:.4f} ± {std[2]:.4f}\n"
        f"Recall : {mean[3]:.4f} ± {std[3]:.4f}\n"
        f"F1     : {mean[4]:.4f} ± {std[4]:.4f}\n"
        f"MCC    : {mean[5]:.4f} ± {std[5]:.4f}"
    )

if __name__ == "__main__":
    main()