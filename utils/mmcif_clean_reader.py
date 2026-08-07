#!/usr/bin/env python3
"""
mmcif_clean_reader.py
---------------------
Reads mmCIF or PDB and returns atom_site-like rows (list[dict]) with stable keys.

Default behavior:
- If multi-model (NMR): keeps FIRST model only

If all_models=True:
- Returns a list of models (each model is (column_list, data_rows))

Features (both formats):
- Removes hydrogens
- Removes duplicate atoms within same chain/residue (keeps highest occupancy)
- Removes inter-chain duplicates with same xyz (keeps first)

Author: Satyabrata Maiti (extended with PDB + model handling)
"""

import os
import numpy as np

# ==========================================================
# ----------------- Helper Functions -----------------------
# ==========================================================

def strip_quotes_for_atom_name(s):
    """Removes quotes from atom names and returns (name, quote_char)."""
    if not s:
        return s, ''
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1], s[0]
    return s, ''


def find_index(header_lines, key):
    """Finds the column index in the header list for a given key fragment."""
    for i, line in enumerate(header_lines):
        if key in line:
            return i
    return None


def _canon_atom_name(name: str) -> str:
    return (name or "").replace("′", "'").replace("*", "'").replace("`", "'").strip()


def _round3(x: float) -> float:
    return float(np.round(x, 3))


# ==========================================================
# ------------------- Duplicate Removal --------------------
# ==========================================================

def remove_duplicates_in_atom_site_block(block_text, keep_first_model=True):
    """
    Given a text block containing '_atom_site.' lines,
    removes duplicates and returns cleaned text.

    Rules:
      1. If keep_first_model: keep only first model if multiple.
      2. Skip hydrogens.
      3. Intra-chain duplicates: same (chain, resid, ins, atom) -> highest occupancy.
      4. Inter-chain duplicates: same (resid, resname, ins, atom, x,y,z) -> keep first.
    """
    lines = block_text.splitlines()
    header_lines = [l for l in lines if l.startswith('_atom_site.')]
    data_lines   = [l for l in lines if not l.startswith('_atom_site.') and not l.startswith('loop_')]

    model_index       = find_index(header_lines, 'pdbx_PDB_model_num')
    chain_index       = find_index(header_lines, 'auth_asym_id')
    atom_index        = find_index(header_lines, 'auth_atom_id')
    residue_index     = find_index(header_lines, 'auth_comp_id')
    resseq_index      = find_index(header_lines, 'auth_seq_id')
    occupancy_index   = find_index(header_lines, 'occupancy')
    cart_x_index      = find_index(header_lines, 'Cartn_x')
    cart_y_index      = find_index(header_lines, 'Cartn_y')
    cart_z_index      = find_index(header_lines, 'Cartn_z')
    type_symbol_index = find_index(header_lines, 'type_symbol')
    ins_code          = find_index(header_lines, 'pdbx_PDB_ins_code')

    # Fallback if auth_seq_id missing (some CIFs)
    if resseq_index is None:
        resseq_index = find_index(header_lines, 'label_seq_id')

    essential = [
        chain_index, atom_index, residue_index, ins_code, resseq_index,
        occupancy_index, cart_x_index, cart_y_index, cart_z_index
    ]
    if any(idx is None for idx in essential):
        # If CIF is unusual, return unmodified
        # (Your rna_prot.py uses auth_* first, but build_residues falls back to label_*.)
        return block_text

    # Step 1: Keep only first model (optional)
    first_model = None
    filtered_lines = []
    for line in data_lines:
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        fields = line.split()
        if type_symbol_index is not None and type_symbol_index < len(fields):
            if fields[type_symbol_index] == 'H':
                continue

        if keep_first_model and (model_index is not None) and (model_index < len(fields)):
            model_num = fields[model_index]
            if first_model is None:
                first_model = model_num
            elif model_num != first_model:
                continue

        filtered_lines.append(line)

    # Step 2: Intra-chain duplicates (keep highest occupancy)
    unique_atoms = {}
    for line in filtered_lines:
        fields = line.split()
        key = (fields[chain_index], fields[resseq_index], fields[ins_code], fields[atom_index])
        try:
            occ = float(fields[occupancy_index])
        except Exception:
            occ = 1.0
        if key not in unique_atoms or occ > unique_atoms[key][1]:
            unique_atoms[key] = (line, occ)

    # Step 3: Inter-chain duplicates (same xyz)
    seen_positions = {}
    final_atom_lines = []
    for (chain_id, rid, ins, aname), (line, occ) in unique_atoms.items():
        fields = line.split()
        try:
            x = _round3(float(fields[cart_x_index]))
            y = _round3(float(fields[cart_y_index]))
            z = _round3(float(fields[cart_z_index]))
        except Exception:
            x = y = z = 0.0
        resname = fields[residue_index]
        key2 = (rid, resname, ins, aname, x, y, z)
        if key2 not in seen_positions:
            seen_positions[key2] = line
            final_atom_lines.append(line)

    cleaned_block_lines = []
    cleaned_block_lines.append('loop_')
    cleaned_block_lines.extend(header_lines)
    cleaned_block_lines.extend(final_atom_lines)
    cleaned_block_lines.append('#')

    return "\n".join(cleaned_block_lines)


# ==========================================================
# ------------------- mmCIF Parser --------------------------
# ==========================================================

def _extract_atom_site_block(full_text: str) -> str:
    """
    Extract only the _atom_site loop block from mmCIF text.
    """
    if "_atom_site." not in full_text:
        raise ValueError("No _atom_site block found in file.")

    lines = full_text.splitlines()
    block_lines = []
    in_block = False

    for line in lines:
        if line.startswith("loop_"):
            # loop_ may start a new loop; we will begin collecting only after _atom_site.
            in_block = False

        if line.startswith("_atom_site."):
            in_block = True

        if in_block:
            block_lines.append(line)

        # stop after block ends
        if in_block and line.startswith("#"):
            break

    return "\n".join(block_lines)


def _parse_cleaned_atom_site_block(cleaned_block: str):
    """
    Parse a CLEANED atom_site block into (column_list, data_rows).
    """
    lines = cleaned_block.splitlines()
    column_list = []
    data_rows   = []
    in_atom_site_loop = False
    line_idx = 0

    for line in lines:
        if line.startswith("#"):
            in_atom_site_loop = False
            line_idx += 1
            continue
        if line.startswith("loop_"):
            in_atom_site_loop = False
            line_idx += 1
            continue
        if line.startswith("_atom_site."):
            col_name = line.split()[0]
            column_list.append(col_name)
            in_atom_site_loop = True
            line_idx += 1
            continue
        if in_atom_site_loop:
            fields = line.split()
            if len(fields) < len(column_list):
                line_idx += 1
                continue
            row = {}
            for i, col in enumerate(column_list):
                raw_val = fields[i]
                if col in ("_atom_site.auth_atom_id", "_atom_site.label_atom_id"):
                    unq, qchar = strip_quotes_for_atom_name(raw_val)
                    row[col] = unq
                    row[col + "__quote"] = qchar
                else:
                    if ((raw_val.startswith('"') and raw_val.endswith('"')) or
                        (raw_val.startswith("'") and raw_val.endswith("'"))):
                        raw_val = raw_val[1:-1]
                    row[col] = raw_val
            row["_original_index"] = line_idx
            data_rows.append(row)
        line_idx += 1

    return column_list, data_rows


def parse_mmcif(filename: str):
    """
    Default: FIRST model only (clean duplicates).
    Returns: (column_list, data_rows)
    """
    with open(filename, "r") as f:
        full_text = f.read()

    block_text = _extract_atom_site_block(full_text)
    cleaned_block = remove_duplicates_in_atom_site_block(block_text, keep_first_model=True)
    return _parse_cleaned_atom_site_block(cleaned_block)


def parse_mmcif_all_models(filename: str):
    """
    Return list of models: [(column_list, data_rows), ...]
    If file is single-model, list length is 1.
    """
    with open(filename, "r") as f:
        full_text = f.read()

    block_text = _extract_atom_site_block(full_text)
    lines = block_text.splitlines()
    header_lines = [l for l in lines if l.startswith('_atom_site.')]
    data_lines   = [l for l in lines if not l.startswith('_atom_site.') and not l.startswith('loop_')]

    model_index = find_index(header_lines, 'pdbx_PDB_model_num')
    if model_index is None:
        # no model column -> treat as single model
        cleaned_block = remove_duplicates_in_atom_site_block(block_text, keep_first_model=True)
        return [ _parse_cleaned_atom_site_block(cleaned_block) ]

    # split by model value
    model_to_lines = {}
    for line in data_lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        fields = line.split()
        if model_index >= len(fields):
            continue
        m = fields[model_index]
        model_to_lines.setdefault(m, []).append(line)

    # build per-model blocks and clean duplicates (without model filtering since already split)
    models_sorted = sorted(model_to_lines.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    out = []
    for m in models_sorted:
        one_block = "\n".join(["loop_"] + header_lines + model_to_lines[m] + ["#"])
        cleaned = remove_duplicates_in_atom_site_block(one_block, keep_first_model=False)
        out.append(_parse_cleaned_atom_site_block(cleaned))
    return out


# ==========================================================
# ------------------- PDB Parser ----------------------------
# ==========================================================

def _parse_pdb_atom_line(line: str):
    """
    PDB fixed width format (ATOM/HETATM). Returns dict or None.
    """
    rec = line[0:6].strip()
    if rec not in ("ATOM", "HETATM"):
        return None

    name = line[12:16].rstrip()  # keep left spacing trimmed
    altLoc = line[16:17]
    resname = line[17:20].strip()
    chainID = line[21:22].strip()
    resSeq  = line[22:26].strip()
    icode   = line[26:27].strip()

    try:
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
    except Exception:
        return None

    # occupancy is 54:60
    occ = 1.0
    try:
        occ_str = line[54:60].strip()
        if occ_str:
            occ = float(occ_str)
    except Exception:
        occ = 1.0

    elem = (line[76:78].strip() or (name.strip()[0].upper() if name.strip() else "C")).strip()

    return {
        "record": rec,
        "name": _canon_atom_name(name.strip()),
        "altLoc": altLoc.strip(),
        "resname": resname,
        "chain": chainID if chainID else "A",
        "resSeq": resSeq,
        "icode": icode if icode else ".",
        "xyz": np.array([x, y, z], dtype=float),
        "elem": elem,
        "occ": occ,
    }


def _pdb_split_models(lines: list[str]) -> list[list[str]]:
    """
    Robust split for multi-model PDB (NMR).

    Uses PDB record name = columns 1-6 (line[0:6].strip()) so it works even if
    there are leading spaces.

    Handles:
      - Proper MODEL ... ENDMDL blocks
      - Missing ENDMDL (new MODEL starts => flush previous)
      - No MODEL records => single model
    """
    models: list[list[str]] = []
    cur: list[str] = []

    saw_model = False
    in_model = False

    for ln in lines:
        rec = ln[0:6].strip() if len(ln) >= 6 else ln.strip()

        if rec == "MODEL":
            saw_model = True
            # flush previous model if ENDMDL was missing
            if cur:
                models.append(cur)
                cur = []
            in_model = True
            continue

        if rec == "ENDMDL":
            if cur:
                models.append(cur)
                cur = []
            in_model = False
            continue

        if rec in ("ATOM", "HETATM"):
            if saw_model:
                if in_model:
                    cur.append(ln)
            else:
                cur.append(ln)

    if cur:
        models.append(cur)

    if saw_model and not models:
        return [[]]

    return models


def _pdb_rows_to_atom_site(rows_atoms: list[dict], model_num: int):
    """
    Convert parsed PDB atoms (list of dict) to mmCIF-like column_list and data_rows.
    """
    column_list = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.auth_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.pdbx_PDB_model_num",
    ]

    data_rows = []
    idx = 1
    for a in rows_atoms:
        if a["elem"].upper() == "H":
            continue
        row = {
            "_atom_site.group_PDB": a["record"],
            "_atom_site.id": str(idx),
            "_atom_site.type_symbol": a["elem"],
            "_atom_site.label_atom_id": a["name"],
            "_atom_site.auth_atom_id": a["name"],
            "_atom_site.label_comp_id": a["resname"],
            "_atom_site.auth_comp_id": a["resname"],
            "_atom_site.label_asym_id": a["chain"],
            "_atom_site.auth_asym_id": a["chain"],
            "_atom_site.label_seq_id": a["resSeq"],
            "_atom_site.auth_seq_id": a["resSeq"],
            "_atom_site.pdbx_PDB_ins_code": a["icode"] if a["icode"] else ".",
            "_atom_site.Cartn_x": f"{a['xyz'][0]:.3f}",
            "_atom_site.Cartn_y": f"{a['xyz'][1]:.3f}",
            "_atom_site.Cartn_z": f"{a['xyz'][2]:.3f}",
            "_atom_site.occupancy": f"{a['occ']:.2f}",
            "_atom_site.pdbx_PDB_model_num": str(model_num),
        }
        row["_original_index"] = idx
        data_rows.append(row)
        idx += 1

    return column_list, data_rows


def _pdb_remove_duplicates(rows_atoms: list[dict]):
    """
    Apply duplicate rules to parsed PDB atom dicts:
      - Keep highest occupancy for same (chain, resSeq, icode, atomname)
      - Inter-chain duplicates same (resSeq, resname, icode, atomname, x,y,z) keep first
      - Skip hydrogens (already done later too)
      - NOTE: we ignore altLoc by occupancy selection via key; this is what you want.
    """
    # Intra-chain duplicates by occ
    uniq = {}
    for a in rows_atoms:
        if a["elem"].upper() == "H":
            continue
        key = (a["chain"], a["resSeq"], a["icode"], a["name"])
        occ = float(a.get("occ", 1.0))
        if key not in uniq or occ > uniq[key]["occ"]:
            uniq[key] = a

    # Inter-chain duplicates by xyz
    seen = set()
    out = []
    for key, a in uniq.items():
        x, y, z = a["xyz"]
        k2 = (a["resSeq"], a["resname"], a["icode"], a["name"], _round3(x), _round3(y), _round3(z))
        if k2 in seen:
            continue
        seen.add(k2)
        out.append(a)

    return out


def parse_pdb(filename: str):
    """
    Default: FIRST model only.
    Returns: (column_list, data_rows) with mmCIF-like keys.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    models = _pdb_split_models(lines)
    first = models[0] if models else []
    atoms = []
    for ln in first:
        if ln.startswith(("ATOM", "HETATM")):
            a = _parse_pdb_atom_line(ln)
            if a is not None:
                atoms.append(a)

    atoms = _pdb_remove_duplicates(atoms)
    return _pdb_rows_to_atom_site(atoms, model_num=1)


def parse_pdb_all_models(filename: str):
    """
    Returns list of models: [(column_list, data_rows), ...]
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    models = _pdb_split_models(lines)
    out = []
    for mi, mlines in enumerate(models, start=1):
        atoms = []
        for ln in mlines:
            if ln.startswith(("ATOM", "HETATM")):
                a = _parse_pdb_atom_line(ln)
                if a is not None:
                    atoms.append(a)
        atoms = _pdb_remove_duplicates(atoms)
        out.append(_pdb_rows_to_atom_site(atoms, model_num=mi))
    return out


# ==========================================================
# ------------------- Dispatcher API ------------------------
# ==========================================================

def parse_file(filename: str, all_models: bool = False):
    """
    Unified dispatcher:
      - .cif/.mmcif -> mmCIF
      - .pdb -> PDB

    Returns:
      if all_models=False: (column_list, data_rows)
      if all_models=True:  list[(column_list, data_rows)]
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".cif", ".mmcif"):
        return parse_mmcif_all_models(filename) if all_models else parse_mmcif(filename)
    elif ext == ".pdb":
        return parse_pdb_all_models(filename) if all_models else parse_pdb(filename)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (expected .cif/.mmcif or .pdb)")


# ==========================================================
# -------------------------- CLI ----------------------------
# ==========================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mmcif_clean_reader.py input.cif|input.pdb [-all]")
        sys.exit(1)

    fn = sys.argv[1]
    allm = ("-all" in sys.argv[2:])
    result = parse_file(fn, all_models=allm)

    if allm:
        print(f"[INFO] Models: {len(result)}")
        for i, (cols, rows) in enumerate(result, start=1):
            print(f"  Model {i}: Columns={len(cols)} Atoms={len(rows)}")
    else:
        cols, rows = result
        print(f"Columns: {len(cols)}, Atoms: {len(rows)}")
