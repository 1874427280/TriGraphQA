import numpy as np
from Bio.PDB import PDBParser, Residue
from typing import List, Tuple, Dict, Any
import torch
from itertools import combinations
from collections import defaultdict

INTER_CHAIN_CONTACT_THRESHOLD = 7.0
RBF_CENTERS = torch.arange(0.0, 16.0, 1.0)
RBF_WIDTH = 1.0
ANGLE_BINS = 8


def get_residue_center_cb(res: Residue.Residue) -> np.ndarray:
    try:
        if res.resname == 'GLY':
            return res['CA'].get_coord()
        else:
            return res['CB'].get_coord()
    except KeyError:
        heavy_atoms = [atom for atom in res.get_atoms() if not atom.name.startswith('H')]
        if heavy_atoms:
            return np.mean([atom.get_coord() for atom in heavy_atoms], axis=0)
        else:
            return np.array([0.0, 0.0, 0.0])


def calculate_angle(center_shared: np.ndarray, center_A: np.ndarray, center_B: np.ndarray) -> float:
    vec_SA = center_A - center_shared
    vec_SB = center_B - center_shared
    dot_product = np.dot(vec_SA, vec_SB)
    norm_SA = np.linalg.norm(vec_SA)
    norm_SB = np.linalg.norm(vec_SB)
    if norm_SA < 1e-6 or norm_SB < 1e-6: return 0.0
    cos_theta = np.clip(dot_product / (norm_SA * norm_SB), -1.0, 1.0)
    return np.arccos(cos_theta)


def rbf_distance_encoding(distance: float) -> torch.Tensor:
    distance_tensor = torch.tensor([distance], dtype=torch.float)
    rbf_features = torch.exp(-((distance_tensor.unsqueeze(1) - RBF_CENTERS) ** 2) / (2 * RBF_WIDTH ** 2))
    return rbf_features.squeeze(0)


def angle_one_hot_encoding(angle_rad: float) -> torch.Tensor:
    angle_rad = np.clip(angle_rad, 0.0, np.pi)
    bin_size = np.pi / ANGLE_BINS
    bin_index = int(angle_rad // bin_size)
    bin_index = np.clip(bin_index, 0, ANGLE_BINS - 1)
    one_hot = torch.zeros(ANGLE_BINS, dtype=torch.float)
    one_hot[bin_index] = 1.0
    return one_hot


def build_idg_with_graphpep_features_integrated(
        pdb_path: str,
        full_2d_matrix: np.ndarray,
        res_to_index_map: Dict[Residue.Residue, int]
) -> Dict[str, Any]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('COMPLEX', pdb_path)
    except Exception as e:
        return {"error": f"{e}"}

    model = structure[0]

    if isinstance(full_2d_matrix, np.ndarray):
        interaction_features_tensor = torch.from_numpy(full_2d_matrix).squeeze(0).float()
    elif isinstance(full_2d_matrix, torch.Tensor):
        interaction_features_tensor = full_2d_matrix.squeeze(0).float()
    else:
        raise TypeError("full_2d_matrix must NumPy or PyTorch Tensor。")

    protein_chains_map: Dict[str, List[Residue.Residue]] = defaultdict(list)
    res_center_map: Dict[Residue.Residue, np.ndarray] = {}

    for chain in model:
        for res in chain:
            if res.has_id("CA"):
                protein_chains_map[chain.id].append(res)
                res_center_map[res] = get_residue_center_cb(res)

    valid_chain_ids = list(protein_chains_map.keys())

    chain1_id, chain2_id = valid_chain_ids[0], valid_chain_ids[1]
    residues_1 = protein_chains_map[chain1_id]
    residues_2 = protein_chains_map[chain2_id]

    idg_nodes = []
    res_1_to_nodes: Dict[Residue.Residue, List[int]] = defaultdict(list)
    res_2_to_nodes: Dict[Residue.Residue, List[int]] = defaultdict(list)
    node_features_64d_list: List[torch.Tensor] = []

    idg_node_string_pairs = []

    for res_1 in residues_1:
        h1 = np.array([a.get_coord() for a in res_1.get_atoms() if not a.name.startswith('H')])
        if len(h1) == 0: continue

        for res_2 in residues_2:
            h2 = np.array([a.get_coord() for a in res_2.get_atoms() if not a.name.startswith('H')])
            if len(h2) == 0: continue

            dist_mat = np.linalg.norm(h1[:, None, :] - h2[None, :, :], axis=-1)
            min_dis = np.min(dist_mat)

            if min_dis <= INTER_CHAIN_CONTACT_THRESHOLD:
                node_idx = len(idg_nodes)
                idg_nodes.append((res_1, res_2, min_dis, node_idx))
                res_1_to_nodes[res_1].append(node_idx)
                res_2_to_nodes[res_2].append(node_idx)

                key_1 = f"{res_1.get_parent().id}_{res_1.id[1]}"
                key_2 = f"{res_2.get_parent().id}_{res_2.id[1]}"
                idg_node_string_pairs.append((key_1, key_2))

                idx_i = res_to_index_map.get(res_1)
                idx_j = res_to_index_map.get(res_2)

                if idx_i is None or idx_j is None:
                    feature_vector = torch.zeros(30, dtype=torch.float)
                else:
                    feature_vector = interaction_features_tensor[:, idx_i, idx_j]

                node_features_64d_list.append(feature_vector)

    num_nodes = len(idg_nodes)
    if num_nodes == 0:
        return {"error": "No useful interaction"}

    node_features = torch.stack(node_features_64d_list, dim=0)

    edges_list: List[Tuple[int, int]] = []
    edge_features_list: List[torch.Tensor] = []

    for res_shared_1, node_indices in res_1_to_nodes.items():
        if len(node_indices) >= 2:
            for u, v in combinations(node_indices, 2):
                res_A = idg_nodes[u][1]
                res_B = idg_nodes[v][1]
                center_shared = res_center_map[res_shared_1]
                center_A = res_center_map[res_A]
                center_B = res_center_map[res_B]

                distance = np.linalg.norm(center_A - center_B)
                angle_rad = calculate_angle(center_shared, center_A, center_B)

                rbf_feat = rbf_distance_encoding(distance)
                angle_feat = angle_one_hot_encoding(angle_rad)
                shared_one_hot = torch.tensor([1.0, 0.0], dtype=torch.float)

                feat_uv = torch.cat([rbf_feat, angle_feat, shared_one_hot], dim=0)
                edges_list.append((u, v));
                edge_features_list.append(feat_uv)
                edges_list.append((v, u));
                edge_features_list.append(feat_uv)

    for res_shared_2, node_indices in res_2_to_nodes.items():
        if len(node_indices) >= 2:
            for u, v in combinations(node_indices, 2):
                res_A = idg_nodes[u][0]
                res_B = idg_nodes[v][0]
                center_shared = res_center_map[res_shared_2]
                center_A = res_center_map[res_A]
                center_B = res_center_map[res_B]

                distance = np.linalg.norm(center_A - center_B)
                angle_rad = calculate_angle(center_shared, center_A, center_B)

                rbf_feat = rbf_distance_encoding(distance)
                angle_feat = angle_one_hot_encoding(angle_rad)
                shared_one_hot = torch.tensor([0.0, 1.0], dtype=torch.float)

                feat_uv = torch.cat([rbf_feat, angle_feat, shared_one_hot], dim=0)
                edges_list.append((u, v));
                edge_features_list.append(feat_uv)
                edges_list.append((v, u));
                edge_features_list.append(feat_uv)

    edge_index = torch.tensor([[e[0] for e in edges_list], [e[1] for e in edges_list]], dtype=torch.long)
    edge_attr = torch.stack(edge_features_list, dim=0) if edge_features_list else torch.empty((0, 26))

    return {
        "chain_1_id": chain1_id,
        "chain_2_id": chain2_id,
        "num_nodes": num_nodes,
        "num_edges": len(edges_list) // 2,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "idg_string_pairs": idg_node_string_pairs
    }

def generate_interface(pdb_file_path, interface_fea):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('COMPLEX', pdb_file_path)
    all_valid_residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.has_id("CA"):
                    all_valid_residues.append(res)

    mock_res_to_index_map = {res: i for i, res in enumerate(all_valid_residues)}

    return build_idg_with_graphpep_features_integrated(
        pdb_path=pdb_file_path,
        full_2d_matrix=interface_fea,
        res_to_index_map=mock_res_to_index_map
    )