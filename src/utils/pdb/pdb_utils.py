from pathlib import Path
from Bio.PDB import MMCIFParser

from src.utils.pdb.calc_features import calculate_contact_features
from src.utils.pdb.calc_3di import calculate_3di_features


def get_extracted_features_from_directory_of_pdb(config):
    collection_of_pdb = collect_structures_from_path(config['pdb_directory'])

    collection_of_pdb_features = []
    for pdb_file, structure, pdb_id in collection_of_pdb:
        contact_features_df = calculate_contact_features(pdb_file, structure, pdb_id, config)
        fs_3di_features_df = calculate_3di_features(structure, pdb_id, config['3di_model_dir'])

        merged_features = merge_contact_features_with_3di_features(contact_features_df, fs_3di_features_df)
        collection_of_pdb_features.append(merged_features)

    return collection_of_pdb_features


def collect_structures_from_path(directory_path):
    collection = []

    for pdb_file in Path(directory_path).glob("*.cif"):
        pdb_id = pdb_file.stem
        structure = MMCIFParser(QUIET=True).get_structure(pdb_id, pdb_file)
        collection.append((pdb_file, structure, pdb_id))

    return collection


def merge_contact_features_with_3di_features(contact_features, fs_3di_features):
    di_cols = ['3di_state', '3di_letter']

    source_di = fs_3di_features[['ch', 'resi', 'ins'] + di_cols].rename(columns={'ch': 's_ch', 'resi': 's_resi', 'ins': 's_ins'})
    source_di = source_di.rename(columns={c: 's_' + c for c in di_cols})

    target_di = fs_3di_features[['ch', 'resi', 'ins'] + di_cols].rename(columns={'ch': 't_ch', 'resi': 't_resi', 'ins': 't_ins'})
    target_di = target_di.rename(columns={c: 't_' + c for c in di_cols})

    merged = contact_features.merge(source_di, on=['s_ch', 's_resi', 's_ins'], how='left')
    merged = merged.merge(target_di, on=['t_ch', 't_resi', 't_ins'], how='left')
    return merged