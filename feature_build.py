import argparse
import os
import re
import sys
import multiprocessing
import subprocess
import traceback
import math
from collections import defaultdict
import numpy as np
import torch
from Bio.PDB import PDBParser

try:
    from pyrosetta import init, pose_from_pdb
    import pyrosetta
    from scipy.spatial import distance_matrix

    init("-mute all")
except ImportError:
    print("Warning：This env is lack of PyRosetta")

AAS_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_TO_IDX = {aa: i for i, aa in enumerate(AAS_LIST)}

MEILER_DICT = {
    'ALA': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23], 'CYS': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    'ASP': [1.60, 0.11, 2.78, 0.54, 3.80, 0.74, 0.22], 'GLU': [1.56, 0.15, 3.78, 0.64, 3.00, 0.78, 0.11],
    'PHE': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38], 'GLY': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
    'HIS': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30], 'ILE': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
    'LYS': [1.89, 0.22, 4.77, 0.99, 9.99, 3.20, 0.27], 'LEU': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
    'MET': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32], 'ASN': [1.60, 0.13, 2.95, 0.31, 4.80, 0.52, 0.25],
    'PRO': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34], 'GLN': [1.56, 0.18, 3.95, 0.42, 5.65, 0.56, 0.22],
    'ARG': [2.34, 0.29, 6.13, 1.01, 10.74, 2.77, 0.00], 'SER': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    'THR': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36], 'VAL': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
    'TRP': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42], 'TYR': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41]
}

AA_to_tip = {
    "ALA": "CB", "CYS": "SG", "ASP": "CG", "ASN": "CG", "GLU": "CD",
    "GLN": "CD", "PHE": "CZ", "HIS": "NE2", "ILE": "CD1", "GLY": "CA",
    "LEU": "CG", "MET": "SD", "ARG": "CZ", "LYS": "NZ", "PRO": "CG",
    "VAL": "CB", "TYR": "OH", "TRP": "CH2", "SER": "OG", "THR": "OG1"
}

try:
    energy_terms = [
        pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,
        pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,
        pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,
        pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_wtd,
        pyrosetta.rosetta.core.scoring.ScoreType.fa_elec,
        pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,
        pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc
    ]
except:
    energy_terms = []

def run_command(cmd, env_name=None):
    full_cmd = f"conda run -n {env_name} bash -c '{cmd}'" if env_name else cmd
    try:
        subprocess.run(full_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.returncode}")

def step1_prepare_and_clean_pdbs(input_dir, clean_pdb_dir):
    os.makedirs(clean_pdb_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.pdb'):
            src_path = os.path.join(input_dir, file)
            dest_path = os.path.join(clean_pdb_dir, file)
            with open(src_path, 'r') as infile, open(dest_path, 'w') as outfile:
                for line in infile:
                    if not line.startswith("HEADER"):
                        outfile.write(line)

def process_single_pdb_split(pdb_file_path, output_A_dir, output_B_dir):
    chains, chains_order, current_chain = {}, [], None
    try:
        with open(pdb_file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    chain_id = line[21]
                    current_chain = chain_id
                    if chain_id not in chains:
                        chains_order.append(chain_id)
                        chains[chain_id] = []
                    chains[chain_id].append(line)
                elif line.startswith('TER') and current_chain is not None:
                    chains[current_chain].append(line)
    except Exception:
        return

    if len(chains_order) != 2:
        print(f"{pdb_file_path} has not 2 chains! Error!")
        return

    chain_A_id, chain_B_id = chains_order[0], chains_order[1]
    pdb_basename = os.path.basename(pdb_file_path)
    filename_A, filename_B = f"{pdb_basename[:-4]}_A.pdb", f"{pdb_basename[:-4]}_B.pdb"

    with open(os.path.join(output_A_dir, filename_A), 'w') as f_a:
        for line in chains[chain_A_id]: f_a.write(line[:21] + 'A' + line[22:])
        f_a.write('END\n')

    with open(os.path.join(output_B_dir, filename_B), 'w') as f_b:
        for line in chains[chain_B_id]: f_b.write(line[:21] + 'B' + line[22:])
        f_b.write('END\n')

def step2_run_voronota(clean_pdb_dir, voronota_bin, voronota_out_dir):
    os.makedirs(voronota_out_dir, exist_ok=True)
    annotated_balls_dir = os.path.join(voronota_out_dir, "contact_balls")
    contact_area_dir = os.path.join(voronota_out_dir, "contact_area")
    contact_orientation_dir = os.path.join(voronota_out_dir, "contact_orientation")

    os.makedirs(annotated_balls_dir, exist_ok=True)
    os.makedirs(contact_area_dir, exist_ok=True)
    os.makedirs(contact_orientation_dir, exist_ok=True)

    for pdb_file in os.listdir(clean_pdb_dir):
        if pdb_file.endswith(".pdb"):
            pdb_path = os.path.join(clean_pdb_dir, pdb_file)
            filename = pdb_file[:-4]

            out_orient = os.path.join(contact_orientation_dir, f"{filename}_annotated_contacts_orientation.txt")
            if not os.path.exists(out_orient):
                cmd1 = f"{voronota_bin} get-balls-from-atoms-file --annotated < {pdb_path} > {annotated_balls_dir}/{filename}_annotated_balls.txt"
                cmd2 = f"{voronota_bin} calculate-contacts --annotated < {annotated_balls_dir}/{filename}_annotated_balls.txt > {contact_area_dir}/{filename}_annotated_contacts_area.txt"
                cmd3 = f"{voronota_bin} calculate-contacts --annotated --draw < {annotated_balls_dir}/{filename}_annotated_balls.txt > {out_orient}"
                run_command(f"{cmd1} && {cmd2} && {cmd3}")

def get_valid_residues_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    valid_residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.has_id("CA"):
                    valid_residues.append((chain.id, res.id[1]))
        break
    return valid_residues

def parse_file(filename, valid_residues_list):
    residue_pairs = defaultdict(lambda: {'normals': [], 'count': 0})
    residue_to_idx = {res: idx for idx, res in enumerate(valid_residues_list)}
    N = len(valid_residues_list)

    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('c<'): continue
            parts = line.strip().split()
            if len(parts) < 12: continue

            atom1, atom2 = parts[0], parts[1]
            match_chain1 = re.search(r'c<([^>]+)>', atom1)
            match_res1 = re.search(r'r<([-]?\d+)', atom1)

            if not match_chain1 or not match_res1: continue
            res1 = (match_chain1.group(1), int(match_res1.group(1)))

            if 'solvent' in atom2: continue

            match_chain2 = re.search(r'c<([^>]+)>', atom2)
            match_res2 = re.search(r'r<([-]?\d+)', atom2)
            if not match_chain2 or not match_res2: continue
            res2 = (match_chain2.group(1), int(match_res2.group(1)))

            try:
                normals = list(map(float, parts[10:13]))
            except ValueError:
                continue

            if res1 in residue_to_idx and res2 in residue_to_idx:
                residue_pairs[(res1, res2)]['normals'].append(normals)
                residue_pairs[(res1, res2)]['count'] += 1

    return residue_pairs, residue_to_idx, N

def build_matrix(residue_pairs, residue_to_idx, N):
    matrix = np.zeros((7, N, N), dtype=np.float32)
    for (res1, res2), data in residue_pairs.items():
        i = residue_to_idx[res1]
        j = residue_to_idx[res2]
        if data['normals']:
            matrix[0:3, i, j] = np.mean(data['normals'], axis=0)
        matrix[3, i, j] = data['count']
        if data['normals']:
            matrix[4:7, i, j] = np.sum(data['normals'], axis=0)
    return matrix

def residue_contact_orientation(file_path, pdb_path):
    valid_residues = get_valid_residues_from_pdb(pdb_path)
    residue_pairs, residue_map, total_res = parse_file(file_path, valid_residues)
    return build_matrix(residue_pairs, residue_map, total_res)

def dist_transform(X, cutoff=4, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime) / scaling

def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i, j] = abs(i - j) * 1.0 / normalizer - 1.0
    return np.expand_dims(ret, axis)

def extract_1d_features(pose):
    seqlen = pose.size()
    geom_mat = np.zeros((6, seqlen))
    energies_mat = np.zeros((4, seqlen))
    ss_mat = np.zeros((4, seqlen))

    score_types = [
        pyrosetta.rosetta.core.scoring.ScoreType.p_aa_pp,
        pyrosetta.rosetta.core.scoring.ScoreType.rama_prepro,
        pyrosetta.rosetta.core.scoring.ScoreType.omega,
        pyrosetta.rosetta.core.scoring.ScoreType.fa_dun
    ]

    def angle_between(v1, v2):
        try:
            return math.acos(v1.dot(v2) / (v1.norm() * v2.norm()))
        except:
            return 0.0

    dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    dssp.insert_ss_into_pose(pose)
    ss_map = {"H": 1, "L": 2, "E": 3}

    for i in range(1, seqlen + 1):
        ss = pose.secstruct(i)
        ss_mat[ss_map.get(ss, 0), i - 1] = 1

        res_e = pose.energies().residue_total_energies(i)
        for j, st in enumerate(score_types):
            energies_mat[j, i - 1] = res_e[st]

        N_curr = pose.residue(i).xyz("N")
        CA_curr = pose.residue(i).xyz("CA")
        C_curr = pose.residue(i).xyz("C")
        C_prev = pose.residue(i - 1).xyz("C") if i > 1 else None
        N_next = pose.residue(i + 1).xyz("N") if i < seqlen else None

        NcCAc = CA_curr - N_curr
        CAcCc = C_curr - CA_curr
        CpNc = N_curr - C_prev if C_prev else None
        CcNn = N_next - C_curr if N_next else None

        geom_mat[0, i - 1] = NcCAc.norm()
        geom_mat[1, i - 1] = CAcCc.norm()
        geom_mat[2, i - 1] = CcNn.norm() if CcNn else 0.0
        geom_mat[3, i - 1] = angle_between(CpNc.negated(), NcCAc) if CpNc else 0.0
        geom_mat[4, i - 1] = angle_between(NcCAc.negated(), CAcCc)
        geom_mat[5, i - 1] = angle_between(CAcCc.negated(), CcNn) if CcNn else 0.0

    averages = np.array([1.456790, 1.524227, 1.333378, 2.125835, 1.947459, 2.039060]).reshape(6, 1)
    geom_mat = np.tanh(geom_mat - averages)

    for i in range(4):
        if i != 3:
            energies_mat[i] = np.tanh(energies_mat[i])
        else:
            energies_mat[i] = np.arcsinh(energies_mat[i]) - 1

    return np.concatenate([geom_mat, energies_mat, ss_mat], axis=0)

def get_distmaps(pose, atom1="CA", atom2="CA", default="CA"):
    psize = pose.size()
    xyz1, xyz2 = np.zeros((psize, 3)), np.zeros((psize, 3))
    for i in range(1, psize + 1):
        r = pose.residue(i)
        xyz1[i - 1, :] = np.array(r.xyz(atom1)) if isinstance(atom1, str) and r.has(atom1) else np.array(
            r.xyz(atom1.get(r.name()[:3], default) if not isinstance(atom1, str) else default))
        xyz2[i - 1, :] = np.array(r.xyz(atom2)) if isinstance(atom2, str) and r.has(atom2) else np.array(
            r.xyz(atom2.get(r.name()[:3], default) if not isinstance(atom2, str) else default))
    return distance_matrix(xyz1, xyz2)

def extract_multi_distance_map(pose):
    return np.stack([
        get_distmaps(pose, atom1="CB", atom2="CB", default="CA"),
        get_distmaps(pose, atom1=AA_to_tip, atom2=AA_to_tip),
        get_distmaps(pose, atom1="CA", atom2=AA_to_tip),
        get_distmaps(pose, atom1=AA_to_tip, atom2="CA")
    ], axis=-1)

def getEulerOrientation(pose):
    psize = pose.size()
    trans_z, rot_z = np.zeros((psize, psize, 3)), np.zeros((psize, psize, 3))
    for i in range(1, psize + 1):
        for j in range(1, psize + 1):
            if i == j: continue
            rt6 = pyrosetta.rosetta.core.scoring.motif.get_residue_pair_rt6(pose, i, pose, j)
            trans_z[i - 1][j - 1] = np.array([rt6[1], rt6[2], rt6[3]])
            rot_z[i - 1][j - 1] = np.array([rt6[4], rt6[5], rt6[6]])
    return np.concatenate([np.deg2rad(trans_z), np.deg2rad(rot_z)], axis=2)

def get_hbonds(pose):
    hb_srbb = []
    hb_lrbb = []

    hbond_set = pose.energies().data().get(pyrosetta.rosetta.core.scoring.EnergiesCacheableDataType.HBOND_SET)
    for i in range(1, hbond_set.nhbonds()):
        hb = hbond_set.hbond(i)
        if hb:
            acceptor = hb.acc_res()
            donor = hb.don_res()
            wtype = pyrosetta.rosetta.core.scoring.hbonds.get_hbond_weight_type(hb.eval_type())
            energy = hb.energy()

            is_acc_bb = hb.acc_atm_is_protein_backbone()
            is_don_bb = hb.don_hatm_is_protein_backbone()

            if is_acc_bb and is_don_bb:
                if wtype == pyrosetta.rosetta.core.scoring.hbonds.hbw_SR_BB:
                    hb_srbb.append((acceptor, donor, energy))
                elif wtype == pyrosetta.rosetta.core.scoring.hbonds.hbw_LR_BB:
                    hb_lrbb.append((acceptor, donor, energy))

    return hb_srbb, hb_lrbb

def extract_EnergyDistM(pose, energy_terms):
    length = int(pose.total_residue())
    tensor = np.zeros((1 + len(energy_terms) + 2, length, length))
    energies = pose.energies()
    graph = energies.energy_graph()

    aas = []
    for i in range(length):
        index1 = i + 1
        aas.append(pose.residue(index1).name().split(":")[0].split("_")[0])

        iru = graph.get_node(index1).const_edge_list_begin()
        irue = graph.get_node(index1).const_edge_list_end()

        while iru != irue:
            edge = iru.__mul__()

            evals = [edge[e] for e in energy_terms]
            index2 = edge.get_other_ind(index1)

            count = 1
            for k in range(len(evals)):
                e = evals[k]
                t = energy_terms[k]

                if t == pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc or t == pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc:
                    if e != 0.0:
                        tensor[count, index1 - 1, index2 - 1] = 1
                else:
                    tensor[count, index1 - 1, index2 - 1] = e

                count += 1
            iru.plus_plus()

    for i in range(1, 1 + len(evals)):
        temp = tensor[i]
        if i == 1 or i == 2:
            tensor[i] = np.arcsinh(np.abs(temp)) / 3.0
        elif i == 3 or i == 4 or i == 5:
            tensor[i] = np.tanh(temp)
    xyzs = []
    for i in range(length):
        index1 = i + 1
        if (pose.residue(index1).has("CB")):
            xyzs.append(pose.residue(index1).xyz("CB"))
        else:
            xyzs.append(pose.residue(index1).xyz("CA"))
    for i in range(length):
        for j in range(length):
            index1 = i + 1
            index2 = j + 1

            vector1 = xyzs[i]
            vector2 = xyzs[j]

            distance = vector1.distance(vector2)

            tensor[0, index1 - 1, index2 - 1] = distance

    hbonds = get_hbonds(pose)
    for hb in hbonds[0]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1 - 1, index2 - 1] = 1
    count += 1
    for hb in hbonds[1]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1 - 1, index2 - 1] = 1

    return tensor, aas

def extract_EnergyDistM_complex(pose, energy_terms):
    length = int(pose.total_residue())
    tensor = np.zeros((1 + len(energy_terms) + 4, length, length))
    energies = pose.energies()
    graph = energies.energy_graph()

    aas = []
    for i in range(length):
        index1 = i + 1
        aas.append(pose.residue(index1).name().split(":")[0].split("_")[0])

        iru = graph.get_node(index1).const_edge_list_begin()
        irue = graph.get_node(index1).const_edge_list_end()

        while iru != irue:
            edge = iru.__mul__()

            evals = [edge[e] for e in energy_terms]
            index2 = edge.get_other_ind(index1)

            count = 1
            for k in range(len(evals)):
                e = evals[k]
                t = energy_terms[k]

                if t == pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc or t == pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc:
                    if e != 0.0:
                        tensor[count, index1 - 1, index2 - 1] = 1
                else:
                    tensor[count, index1 - 1, index2 - 1] = e

                count += 1
            iru.plus_plus()

    for i in range(1, 1 + len(evals)):
        temp = tensor[i]
        if i == 1 or i == 2:
            tensor[i] = np.arcsinh(np.abs(temp)) / 3.0
        elif i == 3 or i == 4 or i == 5:
            tensor[i] = np.tanh(temp)
    xyzs = []
    for i in range(length):
        index1 = i + 1
        if (pose.residue(index1).has("CB")):
            xyzs.append(pose.residue(index1).xyz("CB"))
        else:
            xyzs.append(pose.residue(index1).xyz("CA"))
    for i in range(length):
        for j in range(length):
            index1 = i + 1
            index2 = j + 1

            vector1 = xyzs[i]
            vector2 = xyzs[j]

            distance = vector1.distance(vector2)

            tensor[0, index1 - 1, index2 - 1] = distance

    hbonds = get_hbonds(pose)
    for hb in hbonds[0]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1 - 1, index2 - 1] = 1

    count += 1
    for hb in hbonds[1]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1 - 1, index2 - 1] = 1

    count += 1
    for hb in hbonds[2]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1 - 1, index2 - 1] = 1

    count += 1
    for hb in hbonds[3]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1 - 1, index2 - 1] = 1

    return tensor, aas

def extract_monomer_tensors_from_pose(pose):
    Nres = pose.size()

    fa_scorefxn = pyrosetta.create_score_function("ref2015")
    fa_scorefxn(pose)

    maps = extract_multi_distance_map(pose)
    tbt, aas = extract_EnergyDistM(pose, energy_terms)
    raw_euler = getEulerOrientation(pose)

    maps_T = dist_transform(maps, cutoff=0).transpose(2, 0, 1)
    tbt[0, :, :] = dist_transform(tbt[0, :, :], cutoff=4)
    euler_sincos = np.concatenate([np.sin(raw_euler[:, :, -3:]), np.cos(raw_euler[:, :, -3:])], axis=-1)
    euler_T = euler_sincos.transpose(2, 0, 1)
    sep_T = seqsep(Nres).transpose(2, 0, 1)

    monomer_edge_fea = np.concatenate([maps_T, tbt, euler_T, sep_T], axis=0)
    edge_tensor = torch.tensor(monomer_edge_fea, dtype=torch.float32)

    obt = extract_1d_features(pose)
    residue_one_hot = np.zeros((20, Nres))
    meiler_features = np.zeros((7, Nres))

    for i, aa in enumerate(aas):
        if aa in AA_TO_IDX:
            residue_one_hot[AA_TO_IDX[aa], i] = 1
        meiler_features[:, i] = np.array(MEILER_DICT.get(aa, MEILER_DICT['ALA'])) / 5.0

    monomer_node_fea = np.concatenate([obt, residue_one_hot, meiler_features], axis=0)
    node_tensor = torch.tensor(monomer_node_fea, dtype=torch.float32)

    return node_tensor, edge_tensor

def build_graphs_end_to_end(clean_pdb_dir, split_dir, voronota_out_dir, graph_out_dir, script_dir):
    os.makedirs(graph_out_dir, exist_ok=True)
    split_A, split_B = os.path.join(split_dir, "A"), os.path.join(split_dir, "B")
    os.makedirs(split_A, exist_ok=True)
    os.makedirs(split_B, exist_ok=True)

    if script_dir not in sys.path: sys.path.insert(0, script_dir)
    try:
        from protein_interaction_extractor import extract_interaction_features
        from global_graph_feature import generate_monomer
        from interface_graph_feature import generate_interface
    except ImportError as e:
        print(f"Error: {e}")
        return

    for pdb_file in os.listdir(clean_pdb_dir):
        if not pdb_file.endswith('.pdb'): continue
        file_id = pdb_file[:-4]
        pred_pdb_path = os.path.join(clean_pdb_dir, pdb_file)
        final_graph_path = os.path.join(graph_out_dir, f"{file_id}_graphs.pt")

        if os.path.exists(final_graph_path):
            continue

        try:
            process_single_pdb_split(pred_pdb_path, split_A, split_B)

            pose_A = pose_from_pdb(os.path.join(split_A, f"{file_id}_A.pdb"))
            node_A, edge_A = extract_monomer_tensors_from_pose(pose_A)

            pose_B = pose_from_pdb(os.path.join(split_B, f"{file_id}_B.pdb"))
            node_B, edge_B = extract_monomer_tensors_from_pose(pose_B)

            pose_complex = pose_from_pdb(pred_pdb_path)
            euler_complex = getEulerOrientation(pose_complex)
            maps_complex = extract_multi_distance_map(pose_complex)
            tbt_complex, _ = extract_EnergyDistM_complex(pose_complex, energy_terms)

            euler_np = np.concatenate([np.sin(euler_complex[:, :, -3:]), np.cos(euler_complex[:, :, -3:])], axis=-1)
            euler_permuted = torch.tensor(euler_np, dtype=torch.float32).permute(2, 0, 1)

            tbt_complex[0, :, :] = dist_transform(tbt_complex[0, :, :], cutoff=4)
            tbt_permuted = torch.tensor(tbt_complex, dtype=torch.float32)

            maps_permuted = torch.tensor(dist_transform(maps_complex, cutoff=0), dtype=torch.float32).permute(2, 0, 1)

            contact_orien_path = os.path.join(voronota_out_dir, "contact_orientation",
                                              f"{file_id}_annotated_contacts_orientation.txt")
            contact_orien_fea = torch.tensor(residue_contact_orientation(contact_orien_path, pred_pdb_path),
                                             dtype=torch.float32)

            derived_node_fea = torch.cat([maps_permuted, tbt_permuted, euler_permuted, contact_orien_fea], dim=0)

            interaction_type_np, _ = extract_interaction_features(pred_pdb_path)
            interaction_type = torch.tensor(interaction_type_np, dtype=torch.long)

            interface_graph_node_fea = torch.cat([derived_node_fea, interaction_type.unsqueeze(0).float()], dim=0)
            graph_idg = generate_interface(pred_pdb_path, interface_graph_node_fea.unsqueeze(0))

            if graph_idg is None:
                print(f"[{file_id}] error")
                continue

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('COMPLEX', pred_pdb_path)
            chain_ids = [chain.id for chain in structure[0].get_chains()]

            graph_A_obj = generate_monomer(pred_pdb_path, chain_ids[0], node_A, edge_A)
            graph_B_obj = generate_monomer(pred_pdb_path, chain_ids[1], node_B, edge_B)

            if graph_A_obj is None or graph_B_obj is None:
                print(f"[{file_id}] error!")
                continue

            graph_data = {
                'pdb_id': file_id,
                'graph_A': graph_A_obj,
                'graph_B': graph_B_obj,
                'graph_idg': graph_idg
            }
            torch.save(graph_data, final_graph_path)

        except Exception as e:
            print(f"{file_id}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature building for TriGraphQA.")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw input PDB files")
    parser.add_argument("--work_dir", type=str, required=True, help="Base directory for intermediate and final outputs")

    parser.add_argument("--voronota_bin", type=str, default="/voronota_1.29.4307/voronota", help="Path to voronota binary")
    parser.add_argument("--script_dir", type=str, default="/TriGraphQA/tool", help="Directory containing helper scripts")

    args = parser.parse_args()

    CLEAN_PDB_DIR = os.path.join(args.work_dir, "1_cleaned_pdbs")
    VORONOTA_OUT_DIR = os.path.join(args.work_dir, "2_voronota")
    SPLIT_DIR = os.path.join(args.work_dir, "3_split_chains")
    FINAL_GRAPH_DIR = os.path.join(args.work_dir, "4_final_graphs_pt")

    step1_prepare_and_clean_pdbs(args.input_dir, CLEAN_PDB_DIR)
    step2_run_voronota(CLEAN_PDB_DIR, args.voronota_bin, VORONOTA_OUT_DIR)
    build_graphs_end_to_end(
        clean_pdb_dir=CLEAN_PDB_DIR,
        split_dir=SPLIT_DIR,
        voronota_out_dir=VORONOTA_OUT_DIR,
        graph_out_dir=FINAL_GRAPH_DIR,
        script_dir=args.script_dir
    )