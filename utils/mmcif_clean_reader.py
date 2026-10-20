#!/usr/bin/env python3
"""
mmcif_clean_reader.py
---------------------
Read mmCIF or PDB coordinate files and return atom_site-like rows with stable
keys for NAPxplorer.

Default behavior
----------------
* For multi-model structures, retain the first model only.
* With ``all_models=True``, return each model separately.
* Remove H and D atoms.
* Resolve residue microheterogeneity by retaining one residue identity as a
  whole (highest median heavy-atom occupancy; mean occupancy and input order
  are deterministic tie-breakers).
* Select one coherent alternate conformer for the retained residue identity,
  while retaining atoms with a blank/shared alternate identifier.
* Remove duplicate atom names and exact inter-chain coordinate duplicates.

The reader accepts .cif, .mmcif, .pdb and .ent files, including gzip-compressed
variants such as .pdb.gz and .cif.gz.
"""

from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_MISSING = {"", ".", "?"}


_ION_ELEMENT_BY_RESNAME = {
    "H": "H", "D": "D", "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
    "LI": "Li", "NA": "Na", "K": "K", "RB": "Rb", "CS": "Cs",
    "MG": "Mg", "CA": "Ca", "SR": "Sr", "BA": "Ba",
    "AL": "Al", "GA": "Ga", "ZN": "Zn", "CD": "Cd", "HG": "Hg",
    "MN": "Mn", "FE": "Fe", "CO": "Co", "NI": "Ni", "CU": "Cu",
    "AG": "Ag", "AU": "Au", "PT": "Pt", "IR": "Ir", "PB": "Pb",
    "SE": "Se", "SI": "Si",
}
_TWO_LETTER_ELEMENTS = {v.upper() for v in _ION_ELEMENT_BY_RESNAME.values() if len(v) == 2}


def _open_text(filename: str | os.PathLike[str], mode: str = "rt"):
    """Open plain or gzip-compressed text transparently."""
    path = str(filename)
    if path.lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return open(path, mode, encoding="utf-8", errors="replace")


def _warn(message: str) -> None:
    print(f"[WARNING] mmcif_clean_reader: {message}", file=sys.stderr)


def strip_quotes_for_atom_name(s: str):
    """Remove balanced outer quotes and return ``(value, quote_char)``."""
    if not s:
        return s, ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1], s[0]
    return s, ""


def _strip_outer_quotes(value: str) -> str:
    value = str(value or "")
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


def _present(value: object) -> bool:
    return str(value or "").strip() not in _MISSING


def _canon_atom_name(name: str) -> str:
    return (name or "").replace("′", "'").replace("*", "'").replace("`", "'").strip()


def _round3(x: float) -> float:
    return float(np.round(x, 3))


def _normalize_element(elem: str) -> str:
    s = str(elem or "").strip()
    if not s:
        return ""
    u = s.upper()
    if u == "D":
        return "D"
    if len(u) == 1:
        return u
    return u[0] + u[1:].lower()


def _strip_leading_atom_digits(atom_name: str) -> str:
    s = _canon_atom_name(atom_name).strip()
    while s and s[0].isdigit():
        s = s[1:]
    return s


def _infer_element(atom_name: str, resname: str = "", record: str = "", raw_atom_field: str = "") -> str:
    """Infer an element conservatively when the deposited element field is blank."""
    atom = _strip_leading_atom_digits(atom_name)
    resn = str(resname or "").strip().upper()
    rec = str(record or "").strip().upper()

    if not atom:
        return "C"

    first = atom[0].upper()
    if first == "H":
        return "H"
    if first == "D":
        return "D"


    if rec == "HETATM" and resn in _ION_ELEMENT_BY_RESNAME:
        return _ION_ELEMENT_BY_RESNAME[resn]


    raw = str(raw_atom_field or "")
    letters = "".join(ch for ch in atom if ch.isalpha())
    if rec == "HETATM" and raw and not raw.startswith(" ") and len(letters) >= 2:
        pair = letters[:2].upper()
        if pair in _TWO_LETTER_ELEMENTS:
            return pair[0] + pair[1].lower()

    return first


def _is_hydrogen_or_deuterium(elem: str, atom_name: str = "") -> bool:
    e = str(elem or "").strip().upper()
    if e in {"H", "D"}:
        return True


    if e and e[0].isalpha():
        return False
    atom = _strip_leading_atom_digits(atom_name).upper()
    return bool(atom) and atom[0] in {"H", "D"}


def _occ_float(value: object) -> float:
    try:
        if not _present(value):
            return 1.0
        return float(value)
    except Exception:
        return 1.0


def _median_mean(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 1.0, 1.0
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.mean(arr))


def _header_names(header_lines: Sequence[str]) -> List[str]:
    return [line.split()[0] for line in header_lines]


def _find_exact(header_names: Sequence[str], *names: str) -> Optional[int]:
    for name in names:
        try:
            return header_names.index(name)
        except ValueError:
            pass
    return None


def _first_present_field(fields: Sequence[str], indices: Sequence[Optional[int]], default: str = "") -> str:
    for idx in indices:
        if idx is None or idx < 0 or idx >= len(fields):
            continue
        val = _strip_outer_quotes(fields[idx]).strip()
        if _present(val):
            return val
    return default


def _field(fields: Sequence[str], idx: Optional[int], default: str = "") -> str:
    if idx is None or idx < 0 or idx >= len(fields):
        return default
    return _strip_outer_quotes(fields[idx]).strip()


def _build_cif_column_indices(header_lines: Sequence[str]) -> Dict[str, object]:
    h = _header_names(header_lines)
    return {
        "group": _find_exact(h, "_atom_site.group_PDB"),
        "model": _find_exact(h, "_atom_site.pdbx_PDB_model_num"),
        "chain": [
            _find_exact(h, "_atom_site.auth_asym_id"),
            _find_exact(h, "_atom_site.label_asym_id"),
        ],
        "atom": [
            _find_exact(h, "_atom_site.auth_atom_id"),
            _find_exact(h, "_atom_site.label_atom_id"),
        ],
        "resname": [
            _find_exact(h, "_atom_site.auth_comp_id"),
            _find_exact(h, "_atom_site.label_comp_id"),
        ],
        "resseq": [
            _find_exact(h, "_atom_site.auth_seq_id"),
            _find_exact(h, "_atom_site.label_seq_id"),
        ],
        "ins": [
            _find_exact(h, "_atom_site.pdbx_PDB_ins_code"),
            _find_exact(h, "_atom_site.auth_ins_code"),
            _find_exact(h, "_atom_site.label_ins_code"),
        ],
        "alt": [
            _find_exact(h, "_atom_site.label_alt_id"),
            _find_exact(h, "_atom_site.auth_alt_id"),
            _find_exact(h, "_atom_site.pdbx_PDB_alt_id"),
        ],
        "occ": _find_exact(h, "_atom_site.occupancy"),
        "x": _find_exact(h, "_atom_site.Cartn_x"),
        "y": _find_exact(h, "_atom_site.Cartn_y"),
        "z": _find_exact(h, "_atom_site.Cartn_z"),
        "elem": _find_exact(h, "_atom_site.type_symbol"),
    }


def _validate_cif_essential(indices: Dict[str, object]) -> None:
    missing = []
    for label in ("chain", "atom", "resname", "resseq"):
        vals = indices[label]
        assert isinstance(vals, list)
        if all(v is None for v in vals):
            missing.append(label)
    for label in ("x", "y", "z"):
        if indices[label] is None:
            missing.append(label)
    if missing:
        raise ValueError(
            "mmCIF _atom_site loop lacks essential column(s): " + ", ".join(missing)
        )


def _parse_cif_records(
    header_lines: Sequence[str],
    data_lines: Sequence[str],
    keep_first_model: bool,
) -> List[dict]:
    indices = _build_cif_column_indices(header_lines)
    _validate_cif_essential(indices)
    ncols = len(header_lines)

    records: List[dict] = []
    first_model: Optional[str] = None
    malformed = 0

    for order, line in enumerate(data_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < ncols:
            malformed += 1
            continue

        group_idx = indices["group"]
        group = (
            _field(fields, group_idx, "ATOM").upper()
            if isinstance(group_idx, int)
            else "ATOM"
        )
        if group not in {"ATOM", "HETATM"}:
            continue

        model_idx = indices["model"]
        model = _field(fields, model_idx if isinstance(model_idx, int) else None, "1") or "1"
        if keep_first_model:
            if first_model is None:
                first_model = model
            if model != first_model:
                continue

        chain = _first_present_field(fields, indices["chain"], "_")
        atom = _canon_atom_name(_first_present_field(fields, indices["atom"], ""))
        resname = _first_present_field(fields, indices["resname"], "UNK").upper()
        resid = _first_present_field(fields, indices["resseq"], "0")
        ins = _first_present_field(fields, indices["ins"], ".")
        alt = _first_present_field(fields, indices["alt"], ".")
        occ_idx = indices["occ"]
        occ = _occ_float(_field(fields, occ_idx if isinstance(occ_idx, int) else None, "1.0"))

        try:
            x = float(_field(fields, indices["x"] if isinstance(indices["x"], int) else None))
            y = float(_field(fields, indices["y"] if isinstance(indices["y"], int) else None))
            z = float(_field(fields, indices["z"] if isinstance(indices["z"], int) else None))
        except Exception:
            malformed += 1
            continue

        elem_idx = indices["elem"]
        elem = _normalize_element(_field(fields, elem_idx if isinstance(elem_idx, int) else None, ""))
        if not elem:
            elem = _infer_element(atom, resname, group)
        if _is_hydrogen_or_deuterium(elem, atom):
            continue

        records.append({
            "line": line,
            "fields": fields,
            "order": order,
            "group": group,
            "model": model,
            "chain": chain if _present(chain) else "_",
            "atom": atom,
            "resname": resname,
            "resid": resid,
            "ins": ins if _present(ins) else ".",
            "alt": alt if _present(alt) else ".",
            "occ": occ,
            "xyz": np.asarray([x, y, z], dtype=float),
            "elem": elem,
        })

    if malformed:
        _warn(f"skipped {malformed} malformed/incomplete _atom_site row(s)")
    return records


def _select_residue_identity_and_conformer(records: Sequence[dict]) -> List[dict]:
    """Select one residue identity and one coherent alternate conformer per site."""
    by_position: Dict[Tuple[str, str, str], List[dict]] = {}
    for rec in records:
        key = (str(rec["chain"]), str(rec["resid"]), str(rec["ins"]))
        by_position.setdefault(key, []).append(rec)

    selected: List[dict] = []
    for pos_records in by_position.values():
        by_resname: Dict[str, List[dict]] = {}
        for rec in pos_records:
            by_resname.setdefault(str(rec["resname"]).upper(), []).append(rec)

        def residue_rank(item: Tuple[str, List[dict]]) -> Tuple[float, float, int]:
            _resname, vals = item
            med, mean = _median_mean([float(v.get("occ", 1.0)) for v in vals])
            first = min(int(v.get("order", 0)) for v in vals)
            return med, mean, -first

        chosen_resname, residue_rows = max(by_resname.items(), key=residue_rank)
        if len(by_resname) > 1:
            discarded = sorted(set(by_resname) - {chosen_resname})
            _warn(
                f"microheterogeneity at {pos_records[0]['chain']}:{pos_records[0]['resid']}:{pos_records[0]['ins']}; "
                f"retained {chosen_resname}, discarded {','.join(discarded)}"
            )

        by_alt: Dict[str, List[dict]] = {}
        shared: List[dict] = []
        for rec in residue_rows:
            alt = str(rec.get("alt", "."))
            if not _present(alt):
                shared.append(rec)
            else:
                by_alt.setdefault(alt, []).append(rec)

        chosen_alt: Optional[str] = None
        if by_alt:
            def alt_rank(item: Tuple[str, List[dict]]) -> Tuple[float, float, int]:
                _alt, vals = item
                med, mean = _median_mean([float(v.get("occ", 1.0)) for v in vals])
                first = min(int(v.get("order", 0)) for v in vals)
                return med, mean, -first
            chosen_alt = max(by_alt.items(), key=alt_rank)[0]

        allowed = list(shared)
        if chosen_alt is not None:
            allowed.extend(by_alt[chosen_alt])
        elif not shared:
            allowed = list(residue_rows)


        by_atom: Dict[str, dict] = {}
        for rec in allowed:
            atom = str(rec["atom"])
            old = by_atom.get(atom)
            if old is None:
                by_atom[atom] = rec
                continue
            new_rank = (float(rec.get("occ", 1.0)), -int(rec.get("order", 0)))
            old_rank = (float(old.get("occ", 1.0)), -int(old.get("order", 0)))
            if new_rank > old_rank:
                by_atom[atom] = rec
        selected.extend(by_atom.values())

    return sorted(selected, key=lambda r: int(r.get("order", 0)))


def _remove_exact_interchain_duplicates(records: Sequence[dict]) -> List[dict]:
    seen = set()
    out = []
    for rec in records:
        x, y, z = rec["xyz"]
        key = (
            str(rec["resid"]), str(rec["resname"]), str(rec["ins"]), str(rec["atom"]),
            _round3(float(x)), _round3(float(y)), _round3(float(z)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def remove_duplicates_in_atom_site_block(block_text: str, keep_first_model: bool = True) -> str:
    """Clean one mmCIF ``_atom_site`` loop while tolerating optional columns."""
    lines = block_text.splitlines()
    header_lines = [line for line in lines if line.startswith("_atom_site.")]
    data_lines = [
        line for line in lines
        if not line.startswith("_atom_site.") and not line.startswith("loop_") and not line.startswith("#")
    ]
    if not header_lines:
        raise ValueError("No _atom_site columns found in block")

    records = _parse_cif_records(header_lines, data_lines, keep_first_model=keep_first_model)
    records = _select_residue_identity_and_conformer(records)
    records = _remove_exact_interchain_duplicates(records)

    cleaned = ["loop_", *header_lines, *(rec["line"] for rec in records), "#"]
    return "\n".join(cleaned)


def _extract_atom_site_block(full_text: str) -> str:
    """Extract the first ``_atom_site`` loop from an mmCIF document."""
    if "_atom_site." not in full_text:
        raise ValueError("No _atom_site block found in file.")

    lines = full_text.splitlines()
    block_lines: List[str] = []
    collecting = False
    saw_atom_header = False

    for line in lines:
        if line.startswith("loop_"):
            if collecting and saw_atom_header:
                break
            collecting = False
            saw_atom_header = False
            block_lines = [line]
            continue

        if line.startswith("_atom_site."):
            if not block_lines:
                block_lines = ["loop_"]
            collecting = True
            saw_atom_header = True
            block_lines.append(line)
            continue

        if collecting:
            block_lines.append(line)
            if line.startswith("#"):
                break

    if not saw_atom_header:
        raise ValueError("No _atom_site loop found in file.")
    return "\n".join(block_lines)


def _parse_cleaned_atom_site_block(cleaned_block: str):
    """Parse a cleaned atom_site block into ``(column_list, data_rows)``."""
    lines = cleaned_block.splitlines()
    column_list: List[str] = []
    data_rows: List[dict] = []
    in_atom_site_loop = False
    malformed = 0

    for line_idx, line in enumerate(lines):
        if line.startswith("#"):
            in_atom_site_loop = False
            continue
        if line.startswith("loop_"):
            in_atom_site_loop = False
            continue
        if line.startswith("_atom_site."):
            column_list.append(line.split()[0])
            in_atom_site_loop = True
            continue
        if not in_atom_site_loop or not line.strip():
            continue

        fields = line.split()
        if len(fields) < len(column_list):
            malformed += 1
            continue
        row = {}
        for i, col in enumerate(column_list):
            raw_val = fields[i]
            if col in ("_atom_site.auth_atom_id", "_atom_site.label_atom_id"):
                unquoted, quote_char = strip_quotes_for_atom_name(raw_val)
                row[col] = unquoted
                row[col + "__quote"] = quote_char
            else:
                row[col] = _strip_outer_quotes(raw_val)
        row["_original_index"] = line_idx
        data_rows.append(row)

    if malformed:
        _warn(f"skipped {malformed} malformed row(s) while materializing atom_site dictionaries")
    return column_list, data_rows


def parse_mmcif(filename: str):
    """Parse and clean the first model from an mmCIF file."""
    with _open_text(filename, "rt") as handle:
        full_text = handle.read()
    block_text = _extract_atom_site_block(full_text)
    cleaned = remove_duplicates_in_atom_site_block(block_text, keep_first_model=True)
    return _parse_cleaned_atom_site_block(cleaned)


def parse_mmcif_all_models(filename: str):
    """Return every mmCIF model as ``[(column_list, data_rows), ...]``."""
    with _open_text(filename, "rt") as handle:
        full_text = handle.read()
    block_text = _extract_atom_site_block(full_text)
    lines = block_text.splitlines()
    header_lines = [line for line in lines if line.startswith("_atom_site.")]
    data_lines = [
        line for line in lines
        if not line.startswith("_atom_site.") and not line.startswith("loop_") and not line.startswith("#")
    ]
    indices = _build_cif_column_indices(header_lines)
    _validate_cif_essential(indices)
    model_idx = indices["model"]
    if model_idx is None:
        cleaned = remove_duplicates_in_atom_site_block(block_text, keep_first_model=True)
        return [_parse_cleaned_atom_site_block(cleaned)]

    model_to_lines: Dict[str, List[str]] = {}
    ncols = len(header_lines)
    malformed = 0
    for line in data_lines:
        fields = line.strip().split()
        if len(fields) < ncols:
            malformed += 1
            continue
        group_idx = indices["group"]
        group = (
            _field(fields, group_idx, "ATOM").upper()
            if isinstance(group_idx, int)
            else "ATOM"
        )
        if group not in {"ATOM", "HETATM"}:
            continue
        model = _field(fields, int(model_idx), "1") or "1"
        model_to_lines.setdefault(model, []).append(line)
    if malformed:
        _warn(f"skipped {malformed} malformed row(s) while splitting mmCIF models")

    def model_sort(value: str):
        try:
            return (0, int(value))
        except Exception:
            return (1, value)

    out = []
    for model in sorted(model_to_lines, key=model_sort):
        one_block = "\n".join(["loop_", *header_lines, *model_to_lines[model], "#"])
        cleaned = remove_duplicates_in_atom_site_block(one_block, keep_first_model=False)
        out.append(_parse_cleaned_atom_site_block(cleaned))
    return out or [_parse_cleaned_atom_site_block(remove_duplicates_in_atom_site_block(block_text, True))]


def _parse_pdb_atom_line(line: str):
    """Parse one fixed-width PDB ATOM/HETATM line; malformed lines return None."""
    if len(line) < 54:
        return None
    rec = line[0:6].strip()
    if rec not in {"ATOM", "HETATM"}:
        return None

    raw_name = line[12:16]
    name = _canon_atom_name(raw_name.strip())
    altloc = line[16:17].strip() if len(line) >= 17 else ""
    resname = line[17:20].strip() if len(line) >= 20 else "UNK"
    chain = line[21:22].strip() if len(line) >= 22 else ""
    resseq = line[22:26].strip() if len(line) >= 26 else ""
    icode = line[26:27].strip() if len(line) >= 27 else ""
    if not name or not resseq:
        return None

    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except Exception:
        return None

    occ = 1.0
    if len(line) >= 60:
        occ = _occ_float(line[54:60].strip())

    explicit = line[76:78].strip() if len(line) >= 78 else ""
    elem = _normalize_element(explicit)
    if not elem:
        elem = _infer_element(name, resname, rec, raw_atom_field=raw_name)

    return {
        "record": rec,
        "name": name,
        "altLoc": altloc if _present(altloc) else ".",
        "resname": resname.upper(),
        "chain": chain if chain else "_",
        "resSeq": resseq,
        "icode": icode if icode else ".",
        "xyz": np.asarray([x, y, z], dtype=float),
        "elem": elem,
        "occ": occ,
    }


def _pdb_split_models(lines: Sequence[str]) -> List[List[str]]:
    """Split PDB lines into MODEL blocks, tolerating missing ENDMDL records."""
    models: List[List[str]] = []
    current: List[str] = []
    saw_model = False
    in_model = False

    for line in lines:
        rec = line[0:6].strip() if len(line) >= 6 else line.strip()
        if rec == "MODEL":
            saw_model = True
            if current:
                models.append(current)
                current = []
            in_model = True
            continue
        if rec == "ENDMDL":
            if current:
                models.append(current)
                current = []
            in_model = False
            continue
        if rec in {"ATOM", "HETATM"}:
            if not saw_model or in_model:
                current.append(line)

    if current:
        models.append(current)
    if saw_model and not models:
        return [[]]
    return models


def _pdb_remove_duplicates(atoms: Sequence[dict]) -> List[dict]:
    """Apply residue-level identity/conformer selection and atom deduplication."""
    records = []
    for order, atom in enumerate(atoms):
        if _is_hydrogen_or_deuterium(atom.get("elem", ""), atom.get("name", "")):
            continue
        records.append({
            **atom,
            "order": order,
            "line": "",
            "chain": atom["chain"],
            "resid": atom["resSeq"],
            "ins": atom["icode"],
            "atom": atom["name"],
            "alt": atom.get("altLoc", "."),
        })

    records = _select_residue_identity_and_conformer(records)
    records = _remove_exact_interchain_duplicates(records)
    return [
        {
            "record": rec["record"], "name": rec["name"], "altLoc": rec.get("altLoc", "."),
            "resname": rec["resname"], "chain": rec["chain"], "resSeq": rec["resSeq"],
            "icode": rec["icode"], "xyz": rec["xyz"], "elem": rec["elem"], "occ": rec["occ"],
        }
        for rec in records
    ]


def _pdb_rows_to_atom_site(atoms: Sequence[dict], model_num: int):
    """Convert parsed PDB atoms to mmCIF-like rows used by NAPxplorer."""
    column_list = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.auth_atom_id", "_atom_site.label_alt_id",
        "_atom_site.label_comp_id", "_atom_site.auth_comp_id",
        "_atom_site.label_asym_id", "_atom_site.auth_asym_id",
        "_atom_site.label_seq_id", "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
        "_atom_site.occupancy", "_atom_site.pdbx_PDB_model_num",
    ]

    rows = []
    for idx, atom in enumerate(atoms, start=1):
        if _is_hydrogen_or_deuterium(atom["elem"], atom["name"]):
            continue
        row = {
            "_atom_site.group_PDB": atom["record"],
            "_atom_site.id": str(idx),
            "_atom_site.type_symbol": atom["elem"],
            "_atom_site.label_atom_id": atom["name"],
            "_atom_site.auth_atom_id": atom["name"],
            "_atom_site.label_alt_id": atom.get("altLoc", ".") or ".",
            "_atom_site.label_comp_id": atom["resname"],
            "_atom_site.auth_comp_id": atom["resname"],
            "_atom_site.label_asym_id": atom["chain"],
            "_atom_site.auth_asym_id": atom["chain"],
            "_atom_site.label_seq_id": atom["resSeq"],
            "_atom_site.auth_seq_id": atom["resSeq"],
            "_atom_site.pdbx_PDB_ins_code": atom["icode"] or ".",
            "_atom_site.Cartn_x": f"{atom['xyz'][0]:.3f}",
            "_atom_site.Cartn_y": f"{atom['xyz'][1]:.3f}",
            "_atom_site.Cartn_z": f"{atom['xyz'][2]:.3f}",
            "_atom_site.occupancy": f"{float(atom['occ']):.2f}",
            "_atom_site.pdbx_PDB_model_num": str(model_num),
            "_original_index": idx,
            "_napxplorer_sequence_source": "pdb",
        }
        rows.append(row)
    return column_list, rows


def _read_pdb_lines(filename: str) -> List[str]:
    with _open_text(filename, "rt") as handle:
        return handle.readlines()


def parse_pdb(filename: str):
    """Parse and clean the first model from a PDB/ENT file."""
    models = _pdb_split_models(_read_pdb_lines(filename))
    first = models[0] if models else []
    atoms = []
    malformed = 0
    for line in first:
        atom = _parse_pdb_atom_line(line)
        if atom is None:
            malformed += 1
        else:
            atoms.append(atom)
    if malformed:
        _warn(f"skipped {malformed} malformed PDB ATOM/HETATM line(s)")
    return _pdb_rows_to_atom_site(_pdb_remove_duplicates(atoms), model_num=1)


def parse_pdb_all_models(filename: str):
    """Return every PDB model as ``[(column_list, data_rows), ...]``."""
    models = _pdb_split_models(_read_pdb_lines(filename))
    out = []
    for model_num, model_lines in enumerate(models, start=1):
        atoms = []
        malformed = 0
        for line in model_lines:
            atom = _parse_pdb_atom_line(line)
            if atom is None:
                malformed += 1
            else:
                atoms.append(atom)
        if malformed:
            _warn(f"model {model_num}: skipped {malformed} malformed PDB ATOM/HETATM line(s)")
        out.append(_pdb_rows_to_atom_site(_pdb_remove_duplicates(atoms), model_num=model_num))
    return out


def _coordinate_extension(filename: str) -> str:
    path = Path(filename)
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def parse_file(filename: str, all_models: bool = False):
    """Unified dispatcher for mmCIF, PDB and gzip-compressed variants."""
    ext = _coordinate_extension(filename)
    if ext in {".cif", ".mmcif"}:
        return parse_mmcif_all_models(filename) if all_models else parse_mmcif(filename)
    if ext in {".pdb", ".ent"}:
        return parse_pdb_all_models(filename) if all_models else parse_pdb(filename)
    raise ValueError(
        f"Unsupported coordinate-file extension: {ext or '<none>'} "
        "(expected .cif/.mmcif/.pdb/.ent, optionally followed by .gz)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mmcif_clean_reader.py input.cif|input.pdb [-all]")
        raise SystemExit(1)

    result = parse_file(sys.argv[1], all_models=("-all" in sys.argv[2:]))
    if "-all" in sys.argv[2:]:
        print(f"[INFO] Models: {len(result)}")
        for i, (cols, rows) in enumerate(result, start=1):
            print(f"  Model {i}: Columns={len(cols)} Atoms={len(rows)}")
    else:
        cols, rows = result
        print(f"Columns: {len(cols)}, Atoms: {len(rows)}")
