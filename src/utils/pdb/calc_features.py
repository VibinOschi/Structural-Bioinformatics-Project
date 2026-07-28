import pandas as pd
import warnings

from Bio.PDB import DSSP, HSExposureCB, PPBuilder, is_aa, NeighborSearch
from Bio.SeqUtils import seq1



# Suppression of DSSP warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="Bio.PDB.DSSP"
)


def calculate_contact_features(pdb_file, structure, pdb_id, config):
    # Ramachandran regions
    regions_matrix = []
    with open(config['ramachandran_file']) as f:
        for line in f:
            if line:
                regions_matrix.append([int(ele) for ele in line.strip().split()])

    # Atchely scales
    atchley_scale = {}
    with open(config['atchley_file']) as f:
        next(f)
        for line in f:
            line = line.strip().split("\t")
            atchley_scale[line[0]] = line[1:]

    # Get valid residues
    residues = [residue for residue in structure[0].get_residues() if is_aa(residue) and residue.id[0] == ' ']
    if not residues:
        print("{} no valid residues error  (skipping prediction)".format(pdb_id))
        raise ValueError("no valid residues")

    # Calculate DSSP
    try:
        dssp = dict(DSSP(structure[0], pdb_file, dssp=config['dssp_file']))
    except Exception:
        print("{} DSSP error".format(pdb_id))
        raise

    # Calculate Half Sphere Exposure
    hse = {}
    try:
        hse = dict(HSExposureCB(structure[0]))
    except Exception:
        print("{} HSE error".format(pdb_id))

    # Calculate ramachandran values
    rama_dict = {}  # {(chain_id, residue_id): [phi, psi, ss_class], ...}
    ppb = PPBuilder()
    for chain in structure[0]:
        for pp in ppb.build_peptides(chain):
            phi_psi = pp.get_phi_psi_list()  # [(phi_residue_1, psi_residue_1), ...]
            for i, residue in enumerate(pp):
                phi, psi = phi_psi[i]
                ss_class = None
                if phi is not None and psi is not None:
                    for x, y, width, height, ss_c, color in config["rama_ss_ranges"]:
                        if x <= phi < x + width and y <= psi < y + height:
                            ss_class = ss_c
                            break
                rama_dict[(chain.id, residue.id)] = [phi, psi, ss_class]

    # Generate contacts and add features
    data = []
    ns = NeighborSearch([atom for residue in residues for atom in residue])
    for residue_1, residue_2 in ns.search_all(config["distance_threshold"], level="R"):
        index_1 = residues.index(residue_1)
        index_2 = residues.index(residue_2)

        if abs(index_1 - index_2) >= config["sequence_separation"]:
            aa_1 = seq1(residue_1.get_resname())
            aa_2 = seq1(residue_2.get_resname())
            chain_1 = residue_1.get_parent().id
            chain_2 = residue_2.get_parent().id

            data.append((pdb_id,
                         chain_1,
                         *residue_1.id[1:],
                         aa_1,
                         *dssp.get((chain_1, residue_1.id), [None, None, None, None])[2:4],
                         *hse.get((chain_1, residue_1.id), [None, None])[:2],
                         *rama_dict.get((chain_1, residue_1.id), [None, None, None]),
                         *atchley_scale[aa_1],
                         chain_2,
                         *residue_2.id[1:],
                         aa_2,
                         *dssp.get((chain_2, residue_2.id), [None, None, None, None])[2:4],
                         *hse.get((chain_2, residue_2.id), [None, None])[:2],
                         *rama_dict.get((chain_2, residue_2.id), [None, None, None]),
                         *atchley_scale[aa_2]))

    if not data:
        raise ValueError("no contacts error (skipping prediction)")

    # TODO add sequence separation

    # Create a DataFrame and save to file
    df = pd.DataFrame(data, columns=['pdb_id',
                                     's_ch', 's_resi', 's_ins', 's_resn', 's_ss8', 's_rsa', 's_up', 's_down', 's_phi',
                                     's_psi', 's_ss3', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                                     't_ch', 't_resi', 't_ins', 't_resn', 't_ss8', 't_rsa', 't_up', 't_down', 't_phi',
                                     't_psi', 't_ss3', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']).round(3)

    return df[['pdb_id', 's_ch', 's_resi', 's_ins', 's_resn', 's_ss8', 's_rsa', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                         't_ch', 't_resi', 't_ins', 't_resn', 't_ss8', 't_rsa', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]