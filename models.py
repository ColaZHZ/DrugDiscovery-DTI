import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1) 
    loss = loss_fct(n, labels)           
    return n, loss                       


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]  
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))  
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class SiamDTI(nn.Module):
    def __init__(self, **config):
        super(SiamDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_extractor = Protein_Transformer(protein_emb_dim, protein_padding) 


        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        

        self.bcn2 = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
       

        in_dim = 256
        hidden_dim = 512
        out_dim = 256
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, out_binary)

       
    def forward(self, bg_d, v_protein, mode="train"):
        v_p = self.protein_extractor(v_protein)     
        v_d = self.drug_extractor(bg_d)     

       
        f, att = self.bcn(v_d, v_p)
      
        f_tw, att_tw = self.bcn2(v_p, v_p)      


        dist = f - f_tw

        score_tw = 0
        score1 = score_tw

        dist = self.bn1(F.relu(self.fc1(dist)))
        dist = self.bn2(F.relu(self.fc2(dist)))
        dist = self.bn3(F.relu(self.fc3(dist)))
        dist = self.fc4(dist)
        #===========================================
        if mode == "train":
            return v_d, v_p, f, score1, score_tw, dist
        elif mode == "eval":
            return v_d, v_p, score1, att, score_tw, dist

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')       #？？？？
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class Protein_Transformer(nn.Module):
    def __init__(self, embedding_dim, padding=True):
        super(Protein_Transformer, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        #in_ch = [embedding_dim] + num_filters   #[128, 128, 128, 128]
        
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8), num_layers=3)
        #self.PE = PositionalEncoding(embedding_dim,0.1,max_len=1200)

    def forward(self, v):
        v = self.embedding(v.long()) # [batch_size,seq_len,embedding_size]
        #v = self.PE(v) 
        v = self.transformer_encoder(v)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
