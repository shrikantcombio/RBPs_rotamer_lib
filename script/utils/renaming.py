#!/usr/bin/env python3

import argparse
from pathlib import Path
from mmcif_clean_reader import parse_file


def find_atom_site_loop(lines):
    """
    Find atom_site loop boundaries:
    returns header_start, data_start, data_end, atom_columns
    """
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols = []

            while j < len(lines) and lines[j].startswith("_"):
                cols.append(lines[j].strip())
                j += 1

            if any(c.startswith("_atom_site.") for c in cols):
                data_start = j
                data_end = data_start

                while data_end < len(lines):
                    s = lines[data_end].strip()
                    if s == "#":
                        break
                    if s.startswith("loop_"):
                        break
                    if s.startswith("_") and not s.startswith("_atom_site."):
                        break
                    data_end += 1

                return i + 1, data_start, data_end, cols

        i += 1

    raise ValueError("No _atom_site loop found.")


def replace_label_with_auth_in_original_cif(in_file, out_file):
    """
    Preserve all metadata.
    Only rewrite atom_site data lines after replacing:
      label_asym_id <- auth_asym_id
      label_seq_id  <- auth_seq_id
    """

    text = Path(in_file).read_text()
    lines = text.splitlines()

    _, data_start, data_end, columns = find_atom_site_loop(lines)

    col_index = {c: i for i, c in enumerate(columns)}

    required = [
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_seq_id",
    ]

    missing = [c for c in required if c not in col_index]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    label_chain_i = col_index["_atom_site.label_asym_id"]
    auth_chain_i = col_index["_atom_site.auth_asym_id"]

    label_resid_i = col_index["_atom_site.label_seq_id"]
    auth_resid_i = col_index["_atom_site.auth_seq_id"]

    new_lines = lines[:]

    for k in range(data_start, data_end):
        line = lines[k]

        if not line.strip():
            continue

        if not line.startswith(("ATOM", "HETATM")):
            continue

        fields = line.split()

        if len(fields) < len(columns):
            print(f"[WARNING] Skipping malformed line in {Path(in_file).name}: {line}")
            continue

        fields[label_chain_i] = fields[auth_chain_i]
        fields[label_resid_i] = fields[auth_resid_i]

        new_lines[k] = " ".join(fields)

    Path(out_file).write_text("\n".join(new_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replace label_asym_id/label_seq_id with auth_asym_id/auth_seq_id while preserving full mmCIF metadata."
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory containing .cif/.mmcif files")
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(list(input_dir.glob("*.cif")) + list(input_dir.glob("*.mmcif")))

    if not cif_files:
        print("[INFO] No .cif or .mmcif files found.")
        return

    n_ok = 0

    for in_file in cif_files:
        out_file = output_dir / in_file.name

        try:
            replace_label_with_auth_in_original_cif(in_file, out_file)
            n_ok += 1
            print(f"[OK] {in_file.name} -> {out_file}")
        except Exception as e:
            print(f"[ERROR] {in_file.name}: {e}")

    print(f"\nDone. Processed {n_ok}/{len(cif_files)} files.")


if __name__ == "__main__":
    main()
