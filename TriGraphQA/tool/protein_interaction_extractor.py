import os

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.SeqUtils import seq1
from collections import defaultdict
from scipy.spatial.distance import cdist
import torch



standard_res =[
        "GLY" , 'G',
        "ALA" , 'A',
        "VAL" , 'V',
        "LEU" , 'L',
        "ILE" , 'I',
        "PRO" , 'P',
        "PHE" , 'F',
        "TYR" , 'Y',
        "TRP" , 'W',
        "SER" , 'S',
        "THR" , 'T',
        "CYS" , 'C',
        "MET" , 'M',
        "ASN" , 'N',
        "GLN" , 'Q',
        "ASP" , 'D',
        "GLU" , 'E',
        "LYS" , 'K',
        "ARG" , 'R',
        "HIS" , 'H'
        ]

amino_acid_to_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'else']
degrees = [0, 1, 2, 3, 4, 'else']
hybridizations = ['s', 'sp', 'sp2', 'sp3', 'sp3d', 'sp3d2', 'else']
charges = [-2, -1, 0, 1, 2, 3, 'else']
amino_acids = list("LAGVSETIRDPKQNFYMHW") + ["C", "others"]
num_elements = (20 * 21) // 2
upper_tri_values = np.arange(1, num_elements + 1)
symmetric_interaction_type_matrix = np.zeros((20, 20), dtype=int)
upper_tri_indices = np.triu_indices(20)
symmetric_interaction_type_matrix[upper_tri_indices] = upper_tri_values
symmetric_interaction_type_matrix = symmetric_interaction_type_matrix + symmetric_interaction_type_matrix.T - np.diag(symmetric_interaction_type_matrix.diagonal())

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1.coord - atom2.coord)

def is_hydrogen_bond(res1, res2):
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element in ['N', 'O', 'F'] and atom2.element in ['N', 'O', 'F']:
                distance = calculate_distance(atom1, atom2)
                if 2.7 <= distance <= 3.5:
                    count+=1
    return count

def is_halogen_bond(res1, res2):
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element in ['Cl', 'Br', 'I'] and atom2.element in ['N', 'O', 'F']:
                distance = calculate_distance(atom1, atom2)
                if 3.0 <= distance <= 4.0:
                    count+=1
    return count

def is_sulfur_bond(res1, res2):
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element == 'S' and atom2.element == 'S':
                distance = calculate_distance(atom1, atom2)
                if 3.5 <= distance <= 5.5:
                    count+=1
    return count

def is_pi_stack(res1, res2):
    count=0
    pi_residues = ['PHE', 'TYR', 'TRP']
    if res1.resname in pi_residues and res2.resname in pi_residues:
        for atom1 in res1.get_atoms():
            for atom2 in res2.get_atoms():
                distance = calculate_distance(atom1, atom2)
                if 3.3 <= distance <= 4.5:
                    count+=1
    return count

def is_salt_bridge(res1, res2):
    count = 0
    cationic_atoms = [('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ')]
    anionic_atoms = [('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')]

    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            res1_atom_pair = (res1.resname, atom1.name)
            res2_atom_pair = (res2.resname, atom2.name)
            if (res1_atom_pair in cationic_atoms and res2_atom_pair in anionic_atoms) or \
               (res1_atom_pair in anionic_atoms and res2_atom_pair in cationic_atoms):
                distance = calculate_distance(atom1, atom2)
                if 2.8 <= distance <= 4.0:
                    count += 1
    return count

def is_cation_pi(res1, res2):
    count = 0
    cationic_atoms = [('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ')]
    pi_residues = ['PHE', 'TYR', 'TRP']
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            res1_atom_pair = (res1.resname, atom1.name)
            res2_resname = res2.resname
            res2_atom_pair = (res2.resname, atom2.name)
            res1_resname = res1.resname
            if (res1_atom_pair in cationic_atoms and res2_resname in pi_residues) or \
               (res2_atom_pair in cationic_atoms and res1_resname in pi_residues):
                distance = calculate_distance(atom1, atom2)
                if 4.0 <= distance <= 6.0:
                    count += 1
    return count

def extract_interaction_features(pdb_file):
    parser = PDBParser(QUIET=True)
    pdbid = os.path.basename(pdb_file).split(".")
    structure = parser.get_structure(pdbid[0] + '.' + pdbid[1], pdb_file)
    protein_name = structure.id
    sequence = ""
    coord_matrix = []
    features = []
    interface_atoms = defaultdict(list)

    res_mass_centor = []
    all_atom_coords = []
    all_atom_chains = []
    all_atoms = []
    res_index = -1
    all_res_chain = []
    residue_list = [res for res in structure.get_residues()]
    n_residues = len(residue_list)
    final_res_list = []
    absolute_index_res = -1
    matrix_slice_list = []
    hetatm_res_list = []
    is_fatal_atom = np.zeros(n_residues)
    for model in structure:
        for chain in model:
            for residue in chain:
                absolute_index_res+=1
                res_atoms=[]
                res_atom_coords=[]
                res_atom_chains=[]
                exist_flag=True
                n_atom, ca_atom, c_atom = None, None, None
                for atom in residue:
                    coord = atom.get_vector().get_array()
                    if atom.get_id() == 'N':
                        n_atom = coord
                    elif atom.get_id() == 'CA':
                        ca_atom = coord
                    elif atom.get_id() == 'C':
                        c_atom = coord
                    res_atoms.append(atom)
                    res_atom_coords.append(coord)
                    res_atom_chains.append(chain.id)
                if n_atom is None or ca_atom is None or c_atom is None:
                    exist_flag=False
                    is_fatal_atom[absolute_index_res]=True
                if residue.get_resname() == 'HOH':
                    res_name = 'HOH'
                    all_atoms.extend(res_atoms)
                    all_atom_coords.extend(res_atom_coords)
                    all_atom_chains.extend(res_atom_chains)
                elif exist_flag:
                    res_name = seq1(residue.get_resname())
                    sequence += res_name
                    all_res_chain.append(chain.id)
                    coord_matrix.append([n_atom, ca_atom, c_atom])
                    res_index += 1
                    all_atoms.extend(res_atoms)
                    all_atom_coords.extend(res_atom_coords)
                    all_atom_chains.extend(res_atom_chains)
                    if residue.get_resname() in standard_res:
                        final_res_list.append(residue)
                        res_mass_centor.append(np.mean(res_atom_coords,axis=0))
                        matrix_res2_list=[]
                        for idx2, res2 in enumerate(residue_list):
                            if absolute_index_res <= idx2  or residue.get_resname() == 'HOH':
                                continue
                            if is_fatal_atom[idx2]:
                                continue
                            if res2.get_resname() not in standard_res:
                                continue
                            matrix_res2_slice=np.zeros(6)
                            matrix_res2_slice[0]=is_hydrogen_bond(residue, res2)
                            matrix_res2_slice[1]=is_halogen_bond(residue, res2)
                            matrix_res2_slice[2]=is_sulfur_bond(residue, res2)
                            matrix_res2_slice[3]=is_pi_stack(residue, res2)
                            matrix_res2_slice[4]=is_salt_bridge(residue, res2)
                            matrix_res2_slice[5]=is_cation_pi(residue, res2)
                            matrix_res2_list.append(matrix_res2_slice)
                        matrix_slice_list.append(matrix_res2_list)

                    else:
                        hetatm_res_list.append(residue)
                else:
                    hetatm_res_list.append(residue)

    n_valid_residues = len(matrix_slice_list)
    interaction_matrix = np.zeros((n_valid_residues, n_valid_residues, 6))

    for idx1, matrix_slice in enumerate(matrix_slice_list):
        if idx1 == 0:
            continue
        for idx2, matrix_res2_slice in enumerate(matrix_slice):
            interaction_matrix[idx1, idx2, :] = matrix_res2_slice
            interaction_matrix[idx2, idx1, :] = matrix_res2_slice
    seq = sequence.replace("X", "").replace("Z", "")
    n = len(seq)
    interaction_type = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            aa1 = seq[i]
            aa2 = seq[j]
            idx1 = amino_acid_to_index[aa1]
            idx2 = amino_acid_to_index[aa2]
            interaction_value = symmetric_interaction_type_matrix[idx1, idx2]
            interaction_type[i, j] = interaction_value

    all_atom_coords = np.array(all_atom_coords)
    residue_to_index = {}
    current_index = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() != 'HOH':
                    residue_to_index[residue] = current_index
                    current_index += 1

    coord_matrix = np.array(coord_matrix, dtype=object)

    all_atom_coords = np.array([atom.get_coord() for atom in all_atoms])
    distance_matrix = cdist(all_atom_coords, all_atom_coords)
    within_7A = distance_matrix <= 7.0
    for i in range(len(all_atoms)):
        interface_atoms[i] = np.where(
            (within_7A[i]) &
            (np.arange(len(all_atoms)) != i) &
            (np.array(all_atom_chains) != all_atom_chains[i])
        )[0].tolist()

    grouped_interface = defaultdict(list)
    for i, atom_if in enumerate(list(interface_atoms.values())):
        index = residue_to_index.get(all_atoms[i].get_parent(), -1)
        if index != -1:
            grouped_interface[index].append(atom_if)

    indices_to_delete = [i for i, char in enumerate(sequence) if char == 'X' or char == 'Z']
    coord_matrix = np.delete(coord_matrix, indices_to_delete, axis=0)

    interaction_type = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            aa1 = seq[i]
            aa2 = seq[j]
            idx1 = amino_acid_to_index[aa1]
            idx2 = amino_acid_to_index[aa2]
            interaction_value = symmetric_interaction_type_matrix[idx1, idx2]
            interaction_type[i, j] = interaction_value

    return interaction_type.astype(np.int32), interaction_matrix.astype(np.int32)