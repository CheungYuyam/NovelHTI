import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MainModel(nn.Module):
    def __init__(self, herb_num, gene_num, herb_gene_matrix, gene_herb_matrix, d_i_train, d_j_train, device, kg_type, embedding_type): 
        
        super(MainModel, self).__init__()
        self.herb_gene_matrix = herb_gene_matrix
        self.gene_herb_matrix = gene_herb_matrix

        embeddings_herb = np.load(f'../knowledge_graph_embedding/embeddings/{kg_type}_herb_{embedding_type}_embeddings.npy').astype(np.float64)
        embeddings_target = np.load(f'../knowledge_graph_embedding/embeddings/{kg_type}_target_{embedding_type}_embeddings.npy').astype(np.float64)

        factor_num = embeddings_herb.shape[-1]

        self.embed_herb = nn.Embedding(herb_num, factor_num)
        self.embed_gene = nn.Embedding(gene_num, factor_num)
        
        self.embed_herb.weight.data.copy_(torch.from_numpy(embeddings_herb))  # (num_of_herb, dim_embbedings)
        self.embed_gene.weight.data.copy_(torch.from_numpy(embeddings_target))  # (num_of_gene, dim_embbedings)

        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]
       
        self.d_i_train = torch.FloatTensor(d_i_train).to(device)
        self.d_j_train = torch.FloatTensor(d_j_train).to(device)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num) 

        nn.init.normal_(self.embed_herb.weight, std=0.01)
        nn.init.normal_(self.embed_gene.weight, std=0.01)

    def forward(self, herb, gene_i, gene_j): 
        herbs_embedding = self.embed_herb.weight
        genes_embedding = self.embed_gene.weight

        gcn1_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, genes_embedding) + herbs_embedding.mul(self.d_i_train))
        gcn1_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, herbs_embedding) + genes_embedding.mul(self.d_j_train))
        
        gcn2_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, gcn1_genes_embedding) + gcn1_herbs_embedding.mul(self.d_i_train))
        gcn2_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, gcn1_herbs_embedding) + gcn1_genes_embedding.mul(self.d_j_train))

        gcn3_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, gcn2_genes_embedding) + gcn2_herbs_embedding.mul(self.d_i_train))
        gcn3_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, gcn2_herbs_embedding) + gcn2_genes_embedding.mul(self.d_j_train))

        gcn_herbs_embedding = torch.cat((herbs_embedding,gcn1_herbs_embedding, gcn2_herbs_embedding, gcn3_herbs_embedding), -1)
        gcn_genes_embedding = torch.cat((genes_embedding,gcn1_genes_embedding, gcn2_genes_embedding, gcn3_genes_embedding), -1)
        
        herb = F.embedding(herb.long(), gcn_herbs_embedding)
        gene_i = F.embedding(gene_i.long(), gcn_genes_embedding)
        gene_j = F.embedding(gene_j.long(), gcn_genes_embedding)  

        prediction_i = (herb * gene_i).sum(dim=-1)
        prediction_j = (herb * gene_j).sum(dim=-1) 
        l2_regulization = 0.01 * (herb**2 + gene_i**2 + gene_j**2).sum(dim=-1)
        loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()

        return prediction_i, prediction_j, loss