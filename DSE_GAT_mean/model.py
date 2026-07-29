import dgl
import torch.nn.functional as F
import torch as th
import torch.nn as nn
from utils import *
from propagation import Propagation
from GAT import GraphAttentionLayer, GAT


class GRDTI(nn.Module):
    def __init__(self, g, n_disease, n_drug, n_protein, n_sideeffect, args):
        super(GRDTI, self).__init__()
        self.g = g
        self.device = th.device(args.device)
        self.dim_embedding = args.dim_embedding

        #self.activation = F.elu
        self.reg_lambda = args.reg_lambda

        self.num_disease = n_disease
        self.num_drug = n_drug
        self.num_protein = n_protein
        self.num_sideeffect = n_sideeffect

        self.drug_feat = nn.Parameter(th.FloatTensor(self.num_drug, self.dim_embedding))
        nn.init.normal_(self.drug_feat, mean=0, std=0.1)
        self.protein_feat = nn.Parameter(th.FloatTensor(self.num_protein, self.dim_embedding))
        nn.init.normal_(self.protein_feat, mean=0, std=0.1)
        self.disease_feat = nn.Parameter(th.FloatTensor(self.num_disease, self.dim_embedding))
        nn.init.normal_(self.disease_feat, mean=0, std=0.1)
        self.sideeffect_feat = nn.Parameter(th.FloatTensor(self.num_sideeffect, self.dim_embedding))
        nn.init.normal_(self.sideeffect_feat, mean=0, std=0.1)


        self.gat_Di_D = GAT(self.dim_embedding, self.dim_embedding)
        self.gat_Di_P = GAT(self.dim_embedding, self.dim_embedding)
        self.gat_DDI = GAT(self.dim_embedding, self.dim_embedding, is_hetero=False)
        self.gat_D_Di = GAT(self.dim_embedding, self.dim_embedding)
        self.gat_D_Side = GAT(self.dim_embedding, self.dim_embedding)
        self.gat_D_P = GAT(self.dim_embedding, self.dim_embedding)

        self.gat_PPI = GAT(self.dim_embedding, self.dim_embedding, is_hetero=False)

        self.gat_P_Di = GAT(self.dim_embedding, self.dim_embedding)
        self.gat_P_D = GAT(self.dim_embedding, self.dim_embedding)

        self.gat_Side_D = GAT(self.dim_embedding, self.dim_embedding)



        self.propagation = Propagation(args.k, args.alpha, args.edge_drop)

        # Linear transformation for reconstruction
        tmp = th.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_ch = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Side = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_P = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_PPI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_seq = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))

        self.reset_parameters()

    def reset_parameters(self):
        for m in GRDTI.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_disease, drug_protein, protein_protein,
                protein_disease, drug_sideeffect, drug_sideeffect_mask):

        disease_feat = th.mean(th.stack((self.gat_Di_D(drug_disease.T, self.disease_feat, self.drug_feat),
                                         self.gat_Di_P(protein_disease.T, self.disease_feat, self.protein_feat),
                                         self.disease_feat), dim=1), dim=1)

        drug_feat = th.mean(th.stack((self.gat_DDI(drug_drug, self.drug_feat),
                                      self.gat_D_Di(drug_disease, self.drug_feat, self.disease_feat),
                                      self.gat_D_Side(drug_sideeffect, self.drug_feat, self.sideeffect_feat),
                                      self.gat_D_P(drug_protein, self.drug_feat, self.protein_feat),
                                      self.drug_feat), dim=1), dim=1)

        protein_feat = th.mean(th.stack((self.gat_PPI(protein_protein, self.protein_feat),
                                         self.gat_P_Di(protein_disease, self.protein_feat, self.disease_feat),
                                         self.gat_P_D(drug_protein.T, self.protein_feat, self.drug_feat),
                                         self.protein_feat), dim=1), dim=1)


        sideeffect_feat = th.mean(th.stack((self.gat_Side_D(drug_sideeffect.T, self.sideeffect_feat, self.drug_feat),
                                            self.sideeffect_feat), dim=1), dim=1)

        node_feat = th.cat((disease_feat, drug_feat, protein_feat, sideeffect_feat), dim=0)

        node_feat = self.propagation(dgl.to_homogeneous(self.g), node_feat)

        disease_embedding = node_feat[:self.num_disease].to(self.device)
        drug_embedding = node_feat[self.num_disease:self.num_disease + self.num_drug].to(self.device)
        protein_embedding = node_feat[self.num_disease + self.num_drug:self.num_disease + self.num_drug +
                                                                       self.num_protein].to(self.device)
        sideeffect_embedding = node_feat[-self.num_sideeffect:].to(self.device)

        disease_vector = l2_norm(disease_embedding)
        drug_vector = l2_norm(drug_embedding)
        protein_vector = l2_norm(protein_embedding)
        sideeffect_vector = l2_norm(sideeffect_embedding)

        drug_drug_reconstruct = th.mm(th.mm(drug_vector, self.re_DDI), drug_vector.t())
        drug_drug_reconstruct_loss = th.sum(
            (drug_drug_reconstruct - drug_drug.float()) ** 2)

        drug_disease_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Di), disease_vector.t())
        drug_disease_reconstruct_loss = th.sum(
            (drug_disease_reconstruct - drug_disease.float()) ** 2)

        drug_protein_reconstruct = th.mm(th.mm(drug_vector, self.re_D_P), protein_vector.t())
        drug_protein_reconstruct_loss = th.sum(
            (drug_protein_reconstruct - drug_protein.float()) ** 2)

        protein_protein_reconstruct = th.mm(th.mm(protein_vector, self.re_PPI), protein_vector.t())
        protein_protein_reconstruct_loss = th.sum(
            (protein_protein_reconstruct - protein_protein.float()) ** 2)

        protein_disease_reconstruct = th.mm(th.mm(protein_vector, self.re_P_Di), disease_vector.t())
        protein_disease_reconstruct_loss = th.sum(
            (protein_disease_reconstruct - protein_disease.float()) ** 2)

        drug_sideeffect_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Side), sideeffect_vector.t())
        tmp = th.mul(drug_sideeffect_mask.float(), (drug_sideeffect_reconstruct - drug_sideeffect.float()))
        DSE_potential = drug_sideeffect_reconstruct - drug_sideeffect.float()
        drug_sideeffect_reconstruct_loss = th.sum(tmp ** 2)


        other_loss = drug_drug_reconstruct_loss + drug_disease_reconstruct_loss + \
                     drug_protein_reconstruct_loss + protein_protein_reconstruct_loss + \
                     protein_disease_reconstruct_loss

        L2_loss = 0.
        for name, param in GRDTI.named_parameters(self):
            if 'bias' not in name:
                L2_loss = L2_loss + th.sum(param.pow(2))
        L2_loss = L2_loss * 0.5

        tloss = drug_sideeffect_reconstruct_loss + 1.0 * other_loss + self.reg_lambda * L2_loss

        return tloss, drug_sideeffect_reconstruct_loss, L2_loss, drug_sideeffect_reconstruct, DSE_potential