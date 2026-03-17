import torch
from torch import nn
from res_gated_graph_conv import ResGatedGraphConv
from combine import combine_module_monomers

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        scores = self.attn_net(x)
        weights = torch.softmax(scores, dim=1)
        out = (x * weights).sum(dim=1)
        return out

class IDGGNNBlock(nn.Module):
    def __init__(self, in_node_dim: int, in_edge_dim: int):
        super(IDGGNNBlock, self).__init__()

        self.node_h_dim = 128
        self.edge_h_dim = 32
        self.layer_num = 3
        self.concat_hidden = True

        self.lin_node = nn.Linear(in_node_dim, self.node_h_dim)
        self.lin_edge = nn.Linear(in_edge_dim, self.edge_h_dim)

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(self.layer_num):
            conv = ResGatedGraphConv(
                in_channels=self.node_h_dim,
                out_channels=self.node_h_dim,
                edge_dim=self.edge_h_dim
            )
            self.layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(self.node_h_dim))

        self.dropout = nn.Dropout(0.1)

        if self.concat_hidden:
            self.lin_hidden = nn.Linear(
                self.layer_num * self.node_h_dim,
                self.node_h_dim
            )

    def forward(self, node_features, edge_index, edge_attr):
        layer_input = self.lin_node(node_features)
        edge_feats = self.lin_edge(edge_attr)
        hidden_states = [layer_input]

        for i in range(self.layer_num):
            hidden = self.layers[i](layer_input, edge_index, edge_feats)
            hidden = self.dropout(hidden)
            hidden = self.layer_norms[i](hidden)
            hidden_states.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            idg_hidden = torch.cat(hidden_states[1:], dim=-1)
            final_hidden = self.lin_hidden(idg_hidden)
        else:
            final_hidden = hidden_states[-1]

        return final_hidden

class GlobalResidueGNNBlock(nn.Module):
    def __init__(self, in_node_dim: int, in_edge_dim: int):
        super(GlobalResidueGNNBlock, self).__init__()

        self.node_h_dim = 64
        self.edge_h_dim = 32
        self.layer_num = 3
        self.concat_hidden = True

        self.lin_node = nn.Linear(in_node_dim, self.node_h_dim)
        self.lin_edge = nn.Linear(in_edge_dim, self.edge_h_dim)

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(self.layer_num):
            conv = ResGatedGraphConv(
                in_channels=self.node_h_dim,
                out_channels=self.node_h_dim,
                edge_dim=self.edge_h_dim
            )
            self.layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(self.node_h_dim))

        self.dropout = nn.Dropout(0.1)

        if self.concat_hidden:
            self.lin_hidden = nn.Linear(
                self.layer_num * self.node_h_dim,
                self.node_h_dim
            )

    def forward(self, node_features, edge_index, edge_attr):
        layer_input = self.lin_node(node_features)
        edge_feats = self.lin_edge(edge_attr)
        hidden_states = [layer_input]

        for i in range(self.layer_num):
            hidden = self.layers[i](layer_input, edge_index, edge_feats)
            hidden = self.dropout(hidden)
            hidden = self.layer_norms[i](hidden)
            hidden_states.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            idg_hidden = torch.cat(hidden_states[1:], dim=-1)
            final_hidden = self.lin_hidden(idg_hidden)
        else:
            final_hidden = hidden_states[-1]

        return final_hidden

class PostCombineGNNBlock(nn.Module):
    def __init__(self, in_node_dim: int, in_edge_dim: int):
        super(PostCombineGNNBlock, self).__init__()

        self.node_h_dim = 256
        self.edge_h_dim = 32
        self.layer_num = 3
        self.concat_hidden = True

        self.lin_node = nn.Linear(in_node_dim, self.node_h_dim)

        self.lin_edge = nn.Linear(in_edge_dim, self.edge_h_dim)

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(self.layer_num):
            conv = ResGatedGraphConv(
                in_channels=self.node_h_dim,
                out_channels=self.node_h_dim,
                edge_dim=self.edge_h_dim
            )
            self.layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(self.node_h_dim))

        self.dropout = nn.Dropout(0.1)

        if self.concat_hidden:
            self.lin_hidden = nn.Linear(
                self.layer_num * self.node_h_dim,
                self.node_h_dim
            )

    def forward(self, node_features, edge_index, edge_attr):
        layer_input = self.lin_node(node_features)
        edge_feats = self.lin_edge(edge_attr)
        hidden_states = [layer_input]

        for i in range(self.layer_num):
            hidden = self.layers[i](layer_input, edge_index, edge_feats)
            hidden = self.dropout(hidden)
            hidden = self.layer_norms[i](hidden)
            hidden_states.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            idg_hidden = torch.cat(hidden_states[1:], dim=-1)
            final_hidden = self.lin_hidden(idg_hidden)
        else:
            final_hidden = hidden_states[-1]

        return final_hidden

def build_safe_mapping(res_to_index):
    safe_map = {}
    for res, idx in res_to_index.items():
        chain_id = res.get_parent().id
        safe_map[f"{chain_id}_{res.id[1]}"] = idx
    return safe_map

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()

        self.ia_type_emb = nn.Embedding(402, 8)

        self.idg_encoder = nn.Sequential(
            nn.Linear(37, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.monomer_gnn = GlobalResidueGNNBlock(
            in_node_dim=41,
            in_edge_dim=21
        )

        self.post_combine_gnn = PostCombineGNNBlock(
            in_node_dim=256,
            in_edge_dim=26
        )

        self.pooling = AttentionPooling(in_dim=256)
        self.final_lin = nn.Linear(256, 1, bias=True)

    def forward(self, graph_A, graph_B, graph_idg, device):
        idg_node_fea = torch.tensor(graph_idg['node_features']).to(device)
        idg_edge_index = torch.tensor(graph_idg['edge_index']).to(device)
        idg_edge_fea = torch.tensor(graph_idg['edge_attr']).to(device)

        derived_part = idg_node_fea[:, :29]
        type_part = idg_node_fea[:, 29].long()
        emb_part = self.ia_type_emb(type_part)
        idg_node_fea = torch.cat([derived_part, emb_part], dim=1)

        final_idg_features = self.idg_encoder(idg_node_fea)

        gA_node_fea = graph_A['node_features'].to(device)
        gA_edge_index = graph_A['edge_index'].to(device)
        gA_edge_fea = graph_A['edge_attr'].to(device)

        feat_monomer_A = self.monomer_gnn(gA_node_fea, gA_edge_index, gA_edge_fea)

        gB_node_fea = graph_B['node_features'].to(device)
        gB_edge_index = graph_B['edge_index'].to(device)
        gB_edge_fea = graph_B['edge_attr'].to(device)

        feat_monomer_B = self.monomer_gnn(gB_node_fea, gB_edge_index, gB_edge_fea)

        safe_map_A = build_safe_mapping(graph_A['res_to_global_index'])
        safe_map_B = build_safe_mapping(graph_B['res_to_global_index'])

        idg_string_pairs = graph_idg.get('idg_string_pairs', [])

        combine_out = combine_module_monomers(
            idg_string_pairs=idg_string_pairs,
            final_idg_features=final_idg_features,

            feat_A=feat_monomer_A,
            edge_A=gA_edge_index,
            map_A_safe=safe_map_A,
            coords_A=graph_A['coords'],
            seqs_A=graph_A['seqs'],

            feat_B=feat_monomer_B,
            edge_B=gB_edge_index,
            map_B_safe=safe_map_B,
            coords_B=graph_B['coords'],
            seqs_B=graph_B['seqs']
        )

        post_feat = self.post_combine_gnn(combine_out, idg_edge_index, idg_edge_fea)

        pooled_feature = self.pooling(post_feat.unsqueeze(0))
        DockQ_pred = torch.sigmoid(self.final_lin(pooled_feature))

        return DockQ_pred.squeeze()