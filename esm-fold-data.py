#!/usr/bin/env python3

import os
import re
import gc
import shutil
import hashlib
import subprocess

import numpy as np
import torch
from Bio import SeqIO

FASTA_FILE = "uniprot_sprot.fasta"
OUTPUT_DIR = "pdbs"

START_INDEX = 13383
MAX_SEQUENCE_LENGTH = 1000

CHAIN_LINKER = 25
JOBNAME = "test"

def download_file(url, output_path):
    if os.path.isfile(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return
    print(f"Downloading {output_path}...\nURL: {url}")
    subprocess.run(
        ["curl", "-L", "-A", "Mozilla/5.0", "-o", output_path, url],
        check=True
    )
    print("Download complete.")

def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()

def clean_sequence(sequence, copies=1):
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/", ":").upper())
    sequence = re.sub(":+", ":", sequence)
    sequence = re.sub("^[:]+", "", sequence)
    sequence = re.sub(":+$", "", sequence)
    if copies == "" or copies <= 0:
        copies = 1
    sequence = ":".join([sequence] * copies)
    return sequence

def main():
    if not os.path.isfile(FASTA_FILE):
        raise FileNotFoundError(
            f"Could not find {FASTA_FILE}. "
            f"Make sure it is in the same directory as this script."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device)
    model.esm = model.esm.half()  # convert ESM2 backbone to fp16 to save GPU memory
    model.eval()
    print("Model loaded.")

    processed = 0
    skipped_before_start = 0
    skipped_too_long = 0
    failed = 0

    for i, record in enumerate(SeqIO.parse(FASTA_FILE, "fasta")):
        if i <= START_INDEX:
            skipped_before_start += 1
            continue

        raw_sequence = str(record.seq)

        if len(raw_sequence) > MAX_SEQUENCE_LENGTH:
            skipped_too_long += 1
            continue

        sequence = clean_sequence(raw_sequence, copies=1)

        if not sequence:
            print(f"Skipping empty sequence at index {i}")
            continue

        jobname = re.sub(r"\W+", "", JOBNAME)[:50]
        sequence_id = jobname + "_" + get_hash(sequence)[:5]

        seqs = sequence.split(":")
        lengths = [len(s) for s in seqs]
        length = sum(lengths)

        unique_seqs = list(set(seqs))
        if len(seqs) == 1:
            mode = "mono"
        elif len(unique_seqs) == 1:
            mode = "homo"
        else:
            mode = "hetero"

        print("-" * 60)
        print(f"Record index: {i}")
        print(f"Record ID: {record.id}")
        print(f"Output ID: {sequence_id}")
        print(f"Length: {length}")
        print(f"Mode: {mode}")

        try:
            seq_for_model = sequence.replace(":", "X" * CHAIN_LINKER)

            tokenized = tokenizer(seq_for_model, return_tensors="pt", add_special_tokens=False)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            with torch.no_grad():
                output = model(**tokenized)

            pdb_strings = model.output_to_pdb(output)
            pdb_str = pdb_strings[0]

            plddt = output.plddt[0].mean().item()
            try:
                ptm = output.ptm[0].item()
            except (AttributeError, TypeError, IndexError):
                ptm = 0.0

            print(f"ptm: {ptm:.3f}")
            print(f"plddt: {plddt:.3f}")

            pdb_path = os.path.join(OUTPUT_DIR, f"{sequence_id}.pdb")
            with open(pdb_path, "w") as out:
                out.write(pdb_str)

            print(f"Saved: {pdb_path}")
            processed += 1

            del output
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            failed += 1
            print(f"RuntimeError on record {i}: {e}")
            if "out of memory" in str(e).lower():
                print("CUDA out of memory. Clearing cache and continuing.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue

        except Exception as e:
            failed += 1
            print(f"Error on record {i}: {e}")
            continue

    if os.path.isdir(OUTPUT_DIR):
        archive_path = shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)

if __name__ == "__main__":
    main()
