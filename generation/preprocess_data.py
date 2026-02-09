from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

def _quantile_time_transformation(time_df: pd.DataFrame, entity: pd.DataFrame, numeric_arr):
    times = []
    for _, t in time_df.sec.items():
        time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
        times.append(time.total_seconds())
    time_df['sec'] = times
        
    print("Running quantile transformation")
    qt = QuantileTransformer(n_quantiles=10, random_state=0)
    time_df['sec'] = qt.fit_transform(time_df['sec'].values.reshape(-1,1)).reshape(-1)

    # Create a mapping from entity name to numeric id
    entity_map = dict(zip(entity.entity, entity.id))
    # Map entity names to ids
    numeric_ids = time_df['t'].map(entity_map)
    # Assign the transformed seconds to the numeric array
    numeric_arr = numeric_arr.ravel()
    numeric_arr[numeric_ids.values] = time_df['sec'].values
    return numeric_arr

def preprocess_sphn_kg(node_df, entity, time_opt, num_patients):
    numeric_df = node_df.loc[
        node_df['r'] == '<http://sphn.org/hasValue>', ['t']
    ].copy()

    numeric_df['numeric'] = pd.to_numeric(numeric_df['t'], errors='coerce')

    entity_map = entity.set_index('entity')['id']
    numeric_df['id'] = numeric_df['t'].map(entity_map)
    numeric_arr = np.zeros((len(entity), 1))
    valid = numeric_df['id'].notna()
    numeric_arr[
        numeric_df.loc[valid, 'id'].astype(int).to_numpy()
    ] = numeric_df.loc[valid, 'numeric'].to_numpy().reshape(-1, 1)
    
    if time_opt == 'NT':
        np.save(f"processed_data/sphn_pc_NT_numeric_{num_patients}.npy", numeric_arr)
        print("Literals NT saved.")
    elif time_opt == 'TR':
        np.save(f"processed_data/sphn_pc_TR_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TR saved.")
    elif time_opt == 'TS':
        time_df = node_df[node_df['r'].str.contains('<http://sphn.org/hasStartDateTime>|<http://sphn.org/hasDeterminationDateTime>')].copy()
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
        numeric_arr = _quantile_time_transformation(time_df, entity, numeric_arr)
        np.save(f"processed_data/sphn_pc_TS_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TS saved.")
    elif time_opt == 'TS_TR':
        time_df = node_df[node_df['r'].str.contains('<http://sphn.org/hasStartDateTime>|<http://sphn.org/hasDeterminationDateTime>')].copy()
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
        numeric_arr = _quantile_time_transformation(time_df, entity, numeric_arr)
        np.save(f"processed_data/sphn_pc_TS_TR_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TS TR saved.")

def preprocess_meds_kg(node_df: pd.DataFrame, entity: pd.DataFrame, time_opt, num_patients, prefix="meds"):
    MEDS_NAMESPACE = "https://teamheka.github.io/meds-ontology#"
    numeric_df = node_df.loc[
        node_df['r'] == f'<{MEDS_NAMESPACE}numericValue>', ['t']
    ].copy()

    values = numeric_df['t'].str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#double>')
    numeric_df['numeric'] = pd.to_numeric(values, errors='coerce').round(2)

    entity_map = entity.set_index('entity')['id']
    numeric_df['id'] = numeric_df['t'].map(entity_map)
    numeric_arr = np.zeros((len(entity), 1))
    valid = numeric_df['id'].notna()
    numeric_arr[
        numeric_df.loc[valid, 'id'].astype(int).to_numpy()
    ] = numeric_df.loc[valid, 'numeric'].to_numpy().reshape(-1, 1)

    if time_opt == 'TS':
        time_df = node_df[node_df['r'].str.contains(f'<{MEDS_NAMESPACE}time>')].copy() # can be improved using Graph
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>') # can be improved using Graph
        numeric_arr = _quantile_time_transformation(time_df, entity, numeric_arr)

    np.save(f"processed_data/{prefix}_{time_opt}_numeric_{num_patients}.npy", numeric_arr)
    print("Literals saved.")

def preprocess_kg(num_patients, input_dir: Path, output_dir: Path, data_model: str = "sphn_pc", time_opt="TS"):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = input_dir / f"{data_model}_{time_opt}_{num_patients}.nt"

    node_df = pd.read_csv(input_file, sep=" ", header=None)

    node_df.drop(columns=node_df.columns[-1], axis=1, inplace=True)
    node_df.columns=['h', 'r', 't']

    # Map id to entities and relations.
    ent_to_id = {k: v for v, k in enumerate(set(node_df['h']).union(set(node_df['t'])), start=0)}
    rel_to_id = {k: v for v, k in enumerate(set(node_df['r']), start=0)}

    triples = node_df.copy()
    triples["h"] = node_df.h.map(ent_to_id)
    triples["t"] = node_df.t.map(ent_to_id)
    triples["r"] = node_df.r.map(rel_to_id)    

    entity = pd.DataFrame({'id': list(ent_to_id.values()), 'entity': list(ent_to_id)})
    relation = pd.DataFrame({'id': list(rel_to_id.values()), 'relation': list(rel_to_id)})

    # Save triples, entities and relations.
    triples.to_csv(output_dir / f"{data_model}_{time_opt}_triples_{num_patients}.tsv", sep='\t', index=False, header=False)
    entity.to_csv(output_dir / f"{data_model}_{time_opt}_entities_{num_patients}.tsv", sep='\t', index=False, header=False)
    relation.to_csv(output_dir / f"{data_model}_{time_opt}_relations_{num_patients}.tsv", sep='\t', index=False, header=False)
    
    print(f"[Triples]: {len(triples)} - [Entity]: {len(entity)} - [Relation]: {len(relation)}")

    if data_model=="sphn_pc":
        preprocess_sphn_kg(node_df=node_df, entity=entity, time_opt=time_opt, num_patients=num_patients)
    elif data_model=="meds":
        preprocess_meds_kg(node_df=node_df, entity=entity, time_opt=time_opt, num_patients=num_patients)
