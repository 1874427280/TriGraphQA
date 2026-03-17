import numpy as np
from Bio.PDB import PDBParser, Selection, Residue
from Bio.Data import IUPACData
from typing import List, Tuple, Dict, Any
import torch

INTRA_SPACE_THRESHOLD = 5.0

try:
    STANDARD_RESIDUE_NAMES = [name.upper() for name in IUPACData.protein_letters_3to1.keys()]
except AttributeError:
    STANDARD_RESIDUE_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                              'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


def get_residue_id_string(res: Residue.Residue) -> str:
    chain_id = res.get_parent().id
    res_id = res.id[1]
    res_name = res.resname
    return f"{chain_id}_{res_name}{res_id}"

def build_single_chain_graph(
        model,
        target_chain_id: str,
        intra_t: float,
        full_1d_matrix: torch.Tensor,
        full_2d_matrix: torch.Tensor
) -> Dict[str, Any]:
    if target_chain_id not in model:
        return {"error": f"未在 PDB 模型中找到链 {target_chain_id}"}

    chain = model[target_chain_id]
    chain_residues = [res for res in chain if res.has_id("CA")]

    node_index_map = {res: i for i, res in enumerate(chain_residues)}
    num_nodes = len(chain_residues)
    current_device = full_1d_matrix.device

    node_features = []
    for i in range(num_nodes):
        if i < full_1d_matrix.shape[1]:
            feat = full_1d_matrix[:, i]
        else:
            feat = torch.zeros(41, dtype=torch.float, device=current_device)
        node_features.append(feat)

    final_nodes = torch.stack(node_features) if node_features else torch.empty((0, 41), device=current_device)

    heavy_coords_list = []
    for res in chain_residues:
        h = np.array([a.get_coord() for a in res.get_atoms() if not a.name.startswith('H')])
        heavy_coords_list.append(h)

    # 3. 边构建
    edges = []
    precalc_edge_feats = []

    for i in range(num_nodes):
        res_i = chain_residues[i]
        h_i = heavy_coords_list[i]

        # --- A. 序列相邻连接 (i, i+1) ---
        if i < num_nodes - 1:
            res_j = chain_residues[i + 1]
            if res_j.id[1] == res_i.id[1] + 1:
                if i + 1 < full_2d_matrix.shape[1]:
                    feat_ij = full_2d_matrix[:, i, i + 1]
                    feat_ji = full_2d_matrix[:, i + 1, i]
                else:
                    feat_ij = torch.zeros(21, dtype=torch.float, device=current_device)
                    feat_ji = torch.zeros(21, dtype=torch.float, device=current_device)

                edges.extend([(i, i + 1), (i + 1, i)])
                precalc_edge_feats.extend([feat_ij, feat_ji])

        if len(h_i) == 0: continue

        for j in range(i + 2, num_nodes):
            h_j = heavy_coords_list[j]
            if len(h_j) == 0: continue

            dist_mat = np.linalg.norm(h_i[:, None, :] - h_j[None, :, :], axis=-1)
            min_dis = np.min(dist_mat)

            if min_dis <= intra_t:
                if j < full_2d_matrix.shape[1]:
                    feat_ij = full_2d_matrix[:, i, j]
                    feat_ji = full_2d_matrix[:, j, i]
                else:
                    feat_ij = torch.zeros(21, dtype=torch.float, device=current_device)
                    feat_ji = torch.zeros(21, dtype=torch.float, device=current_device)

                edges.extend([(i, j), (j, i)])
                precalc_edge_feats.extend([feat_ij, feat_ji])

    if edges:
        edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long,
                                  device=current_device)
        final_edge_attr = torch.stack(precalc_edge_feats)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=current_device)
        final_edge_attr = torch.empty((0, 21), dtype=torch.float, device=current_device)

    coords_tensor = torch.zeros((num_nodes, 3), dtype=torch.float, device=current_device)
    seqs_tensor = torch.zeros(num_nodes, dtype=torch.float, device=current_device)

    for i, res in enumerate(chain_residues):
        seqs_tensor[i] = res.id[1]
        try:
            coords_tensor[i] = torch.tensor(res['CA'].get_coord(), dtype=torch.float, device=current_device)
        except KeyError:
            heavy = [a.get_coord() for a in res.get_atoms() if not a.name.startswith('H')]
            coords_tensor[i] = torch.tensor(np.mean(heavy, axis=0) if heavy else [0, 0, 0], dtype=torch.float,
                                            device=current_device)

    return {
        "num_nodes": num_nodes,
        "num_edges": len(edges) // 2,
        "node_features": final_nodes,
        "edge_index": edge_index,
        "edge_attr": final_edge_attr,
        "res_to_global_index": node_index_map,
        "node_feature_dim": final_nodes.shape[1] if num_nodes > 0 else 41,
        "edge_feature_dim": final_edge_attr.shape[1] if len(edges) > 0 else 21,
        "coords": coords_tensor,
        "seqs": seqs_tensor
    }


def generate_monomer(
        pdb_file_path: str,
        target_chain_id: str,
        global_node_fea: Any,
        global_edge_fea: Any
) -> Dict[str, Any]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('COMPLEX', pdb_file_path)


    model = structure[0]

    if isinstance(global_node_fea, np.ndarray):
        node_features_tensor = torch.from_numpy(global_node_fea).squeeze(0).float()
    else:
        node_features_tensor = global_node_fea.squeeze(0).float()

    if isinstance(global_edge_fea, np.ndarray):
        edge_features_tensor = torch.from_numpy(global_edge_fea).squeeze(0).float()
    else:
        edge_features_tensor = global_edge_fea.squeeze(0).float()

    monomer_graph_result = build_single_chain_graph(
        model=model,
        target_chain_id=target_chain_id,
        intra_t=INTRA_SPACE_THRESHOLD,
        full_1d_matrix=node_features_tensor,
        full_2d_matrix=edge_features_tensor
    )

    return monomer_graph_result