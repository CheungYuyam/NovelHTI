import torch
import logging
import random
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

import dgl.nn.pytorch as dglnn

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from scipy.sparse import load_npz

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.init as init


class HerbEncoder(nn.Module):
    def __init__(self, herbs_data_path, embed_dim=64, device='cuda:0'):
        super().__init__()
        
        herbs_data = torch.from_numpy(load_npz(herbs_data_path).toarray()).float()
        self.register_buffer("herbs_data", herbs_data.to(device))
        
        self.proj = nn.Linear(herbs_data.shape[1], embed_dim)
        self._init_weights()

    def _init_weights(self):
        init.xavier_normal_(self.proj.weight)
        init.zeros_(self.proj.bias)

    def forward(self, herb_ids):
        herb_features = self.herbs_data[herb_ids.long()]  # (B, 2364)
        return self.proj(herb_features), herb_features


class ConfidenceScalingFusion(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.proj = nn.Linear(dim2, dim1)
        self.confidence_fc = nn.Sequential(
            nn.Linear(dim2, dim1),
            nn.Sigmoid()
        )
        
        for layer in [self.proj, self.confidence_fc[0]]:
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, output_i, output_c):
        projected_c = self.proj(output_c)
        confidence = self.confidence_fc(output_c)
        return confidence * (output_i + projected_c)


class HeteroGraphConvModel(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, num_rels1, num_rels2, 
                 map1_path, map2_path, map3_path, layer_i, layer_o, device='cuda:0'):
        super().__init__()

        self.attn_fusion = ConfidenceScalingFusion(feature_dim1, feature_dim2)

        self.dim1 = feature_dim1
        self.dim2 = feature_dim2

        self.linear = nn.Linear(feature_dim1, feature_dim2, bias=False)

        self.layer_i = layer_i
        self.layer_o = layer_o

        self.num_rels1 = num_rels1
        self.num_rels2 = num_rels2

        self.device = device

        TCM_MM = load_npz(map1_path).toarray()
        MM_Disease = load_npz(map2_path).toarray()
        Disease_Target = load_npz(map3_path).toarray()
        self.total_map = torch.FloatTensor(TCM_MM @ MM_Disease @ Disease_Target).to(device)
        self.total_map.requires_grad_(False)

        self.attn_relations1 = nn.ModuleList([nn.Linear(feature_dim1, num_rels1) for i in range(layer_i*layer_o)])
        self.attn_relations2 = nn.ModuleList([nn.Linear(feature_dim1, num_rels2) for i in range(layer_o)])

        self.register_buffer('kegg_go_buffer', torch.zeros(1, 311+4650, feature_dim2, device=device))

        self.convs1 = nn.ModuleList()
        for l in range(layer_i*layer_o):
            conv_dict = {}
            for edge_type in range(self.num_rels1):
                conv_dict[f'rel_{edge_type}'] = dglnn.GraphConv(self.dim1, self.dim1, activation=nn.ReLU(), bias=False)
            conv = dglnn.HeteroGraphConv(conv_dict, aggregate='sum')
            self.convs1.append(conv)

        self.convs2 = nn.ModuleList()
        for l in range(layer_o):
            conv_dict = {}
            for edge_type in range(self.num_rels2):
                conv_dict[f'rel_{edge_type+self.num_rels1}'] = dglnn.GraphConv(self.dim2, self.dim2, activation=nn.ReLU(), bias=False)
            conv = dglnn.HeteroGraphConv(conv_dict, aggregate='sum')
            self.convs2.append(conv)

        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, dglnn.GraphConv):
                nn.init.orthogonal_(module.weight)

        self.apply(_init_weights)
    
    def forward(self, graph1, graph2, herb_feature, herb_original_feat, p_targets, n_targets):

        biggraph1 = dgl.batch([graph1] * herb_feature.shape[0])  # repeat graph for B times
        biggraph2 = dgl.batch([graph2] * herb_feature.shape[0])  # repeat graph for B times

        i_features_original = torch.zeros(
            (herb_feature.shape[0], herb_original_feat.shape[1], self.dim1),
            device=self.device
        )
        mask = herb_original_feat.bool()
        batch_indices, node_indices = torch.nonzero(mask, as_tuple=True)
        i_features_original[batch_indices, node_indices] = herb_feature[batch_indices].float()  # (B, 2364, dim1)

        i_features = torch.einsum("bnd,nl->bld", i_features_original, self.total_map)  # (B, 7854, dim1)
        
        kegg_go_features = self.kegg_go_buffer.expand(herb_feature.shape[0], -1, -1)  # (B, 311+4650, dim2), pathways
        combined_features = torch.cat([self.linear(i_features), kegg_go_features], dim=1)  # (B, 7854+311=12815, dim2)

        i_features = i_features.reshape(-1, self.dim1)  # (B*7854, dim1)
        c_features = combined_features.view(-1, self.dim2)  # (B*12815, dim2)

        i_features = {'node': i_features}
        c_features = {'node': c_features}

        for o_item in range(self.layer_o):  # outer loop

            # Stage 1: inner loop (target loop)
            for i_item in range(self.layer_i):
                # (B, dim1), (dim1, num_rels1) --> (B, num_rels1)
                attention_weights1 = torch.sigmoid(self.attn_relations1[o_item*self.layer_i+i_item](herb_feature))
                mod_kwargs = {}
                for edge_type in range(self.num_rels1):
                    edge_type_name = f'rel_{edge_type}'
                    num_edges = graph1.num_edges(edge_type_name)
                    edge_weights = attention_weights1[:, edge_type].repeat_interleave(num_edges)
                    mod_kwargs[edge_type_name] = {'edge_weight': edge_weights}
                i_features = self.convs1[o_item*self.layer_i+i_item](biggraph1, i_features, mod_kwargs=mod_kwargs)

            # Stage 2: pathway loop
            attention_weights2 = torch.sigmoid(self.attn_relations2[o_item](herb_feature))  # (B, dim1), (dim1, num_rels2) --> (B, num_rels1)
            mod_kwargs = {}
            for edge_type in range(self.num_rels2):
                edge_type_name = f'rel_{edge_type+self.num_rels1}'
                num_edges = graph2.num_edges(edge_type_name)
                edge_weights = attention_weights2[:, edge_type].repeat_interleave(num_edges)
                mod_kwargs[edge_type_name] = {'edge_weight': edge_weights}
            c_features = self.convs2[o_item](biggraph2, c_features, mod_kwargs=mod_kwargs)
            c_features = self.convs2[o_item](biggraph2, c_features, mod_kwargs=mod_kwargs)

            # Stage 3: combine features
            output_i_features = i_features['node']  # (B*7854, dim1)
            output_c_features = c_features['node']  # (B*12815, dim2)
            output_c_features = output_c_features.view(attention_weights2.shape[0], -1, self.dim2)  # (B, 12815, dim2)
            output_c_features = output_c_features[:, :7854, :]  # (B, 7854, dim2)
            output_c_features = output_c_features.reshape(-1, self.dim2)  # (B*7854, dim2)
            f_features = self.attn_fusion(output_i_features, output_c_features)  # (B*7854, dim1)

            i_features['node'] = f_features
            f_features_reshape = f_features.view(attention_weights2.shape[0], -1, self.dim1)  # (B, 7854, dim1)
            combined_features = torch.cat([self.linear(f_features_reshape), kegg_go_features], dim=1)  # (B, 7854+311=12815, dim2)
            combined_features = combined_features.reshape(-1, self.dim2)  # (B*12815, dim2)
            c_features['node'] = combined_features

        o_features = f_features  # (B*7854, dim1)
        o_features = o_features.view(attention_weights1.shape[0], -1, self.dim1)  # (B, 7854, dim1)
        
        p_features = o_features[torch.arange(p_targets.size(0)), 
                                p_targets.view(-1).long(),
                                :]
                
        n_features = o_features[torch.arange(p_targets.size(0)),
                                n_targets.view(-1).long(),
                                :]
                        
        return p_features, n_features
    
class MainModel(nn.Module):
    def __init__(self, herbs_data_path, map1_path, map2_path, map3_path, 
                 kg_graph1, kg_graph2, 
                 feature_dim1=8, feature_dim2=4, num_inner_layer=2, num_outer_layer=3,
                 device='cuda:0'):
        super().__init__()
        self.herb_encoder = HerbEncoder(herbs_data_path, embed_dim=feature_dim1, device=device)

        self.kg1 = kg_graph1
        self.kg2 = kg_graph2

        self.hGCN = HeteroGraphConvModel(
            feature_dim1=feature_dim1,
            feature_dim2=feature_dim2,
            num_rels1=len(kg_graph1.etypes),  # 12
            num_rels2=len(kg_graph2.etypes),  # 2+6
            map1_path=map1_path,
            map2_path=map2_path,
            map3_path=map3_path,
            layer_i=num_inner_layer,
            layer_o=num_outer_layer,
            device=device
        )
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim1, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        init.xavier_normal_(self.fc[0].weight)
        init.zeros_(self.fc[0].bias)

        init.xavier_normal_(self.fc[2].weight)
        init.zeros_(self.fc[2].bias)

    def forward(self, herbs, p_targets, n_targets):
        herbs_feat, herbs_original_feat = self.herb_encoder(herbs)  # --> (B, dim1), (B, 2364) one-hot
        p_features, n_features = self.hGCN(graph1=self.kg1, graph2=self.kg2,
                                           herb_feature=herbs_feat, herb_original_feat = herbs_original_feat,
                                           p_targets=p_targets, n_targets=n_targets)  # (2*B, dim1)
        return self.fc(p_features), self.fc(n_features)  # (B, 1), (B, 1)