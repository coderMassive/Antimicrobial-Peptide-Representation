import os
import glob
import hashlib
import shutil
import pandas as pd
import numpy as np

from contact_map import ContactFrequency
import mdtraj as md

from Bio import SeqIO

out_dir = "dataset"
os.makedirs(out_dir, exist_ok=True)

target_size = 256

for record in SeqIO.parse("uniprot_sprot.fasta", "fasta"):
    sequence = str(record.seq)
    peptide_id = record.id

    query_sequence = "".join(sequence.split())
    basejobname = "test"
    jobname = basejobname + "_" + hashlib.sha1(query_sequence.encode()).hexdigest()[:5]

    try:
      pdb = glob.glob(f"pdbs/{jobname}.pdb")[0]
    except:
      print(jobname, "skipped")
      continue
    traj = md.load(pdb)
    cm = ContactFrequency(traj[0])
    matrix = cm.residue_contacts.df.to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0)

    current_size = matrix.shape[0]
    if current_size < target_size:
        pad_width = target_size - current_size
        matrix = np.pad(matrix, ((0, pad_width), (0, pad_width)), mode='constant', constant_values=0.0)
    elif current_size > target_size:
        matrix = matrix[:target_size, :target_size]

    np.save(os.path.join(out_dir, f"{peptide_id}.npy"), matrix)
