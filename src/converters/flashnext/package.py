#!/usr/bin/env python3
"""Package standalone Qwen3.8-Flash-Next-TFQ3 checkpoint from 8-bit source + TFQ3 MoE + B-bit PLE."""
import argparse
import glob
import json
import os
import shutil
import struct
import time


def read_st_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def write_safetensors_streaming(out_path: str, items):
    """items: list of (name, dtype, shape, src_path, src_abs_start, nbytes)"""
    header = {"__metadata__": {"format": "mlx", "source": "Qwen3.8-Flash-Next-TFQ3"}}
    cur = 0
    for name, dtype, shape, _sp, _off, nbytes in items:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cur, cur + nbytes]}
        cur += nbytes
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hb) % 8)) % 8
    hb += b" " * pad
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(hb)))
        fout.write(hb)
        for _name, _dtype, _shape, src_path, src_abs_start, nbytes in items:
            with open(src_path, "rb") as fin:
                fin.seek(src_abs_start)
                rem = nbytes
                while rem > 0:
                    chunk = fin.read(min(rem, 16 * 1024 * 1024))
                    if not chunk:
                        raise RuntimeError(f"Short read on {src_path}")
                    fout.write(chunk)
                    rem -= len(chunk)
    os.replace(tmp, out_path)
    return cur


def extract_nonmoe(src_dir: str, out_dir: str):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for k, fn in idx.items():
        if ".switch_mlp." in k or ".ngram_embedding." in k:
            continue
        by_file.setdefault(fn, []).append(k)

    # Split into 2 balanced shards (~3.3 GB each)
    items_all = []
    for fn in sorted(by_file.keys()):
        sp = os.path.join(src_dir, fn)
        hdr, data_base = read_st_header(sp)
        for k in sorted(by_file[fn]):
            meta = hdr[k]
            a, b = meta["data_offsets"]
            items_all.append((k, meta["dtype"], meta["shape"], sp, data_base + a, b - a))

    total_bytes = sum(x[5] for x in items_all)
    half = total_bytes // 2
    shards = [[], []]
    acc = 0
    for it in items_all:
        if acc >= half and shards[0]:
            shards[1].append(it)
        else:
            shards[0].append(it)
            acc += it[5]

    for si, shard_items in enumerate(shards):
        out_p = os.path.join(out_dir, f"nonmoe_q8_{si:02d}.safetensors")
        nb = write_safetensors_streaming(out_p, shard_items)
        print(f"wrote {out_p}: {len(shard_items)} tensors, {nb / 1e9:.3f} GB ({time.time() - t0:.1f}s)", flush=True)

    copy_files = [
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "LICENSE",
        "qwen-logo.png",
    ]
    for cf in copy_files:
        sp = os.path.join(src_dir, cf)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(out_dir, cf))
    print(f"Copied tokenizer/metadata files ({time.time() - t0:.1f}s)", flush=True)


def write_index_and_config(src_dir: str, out_dir: str, ple_bits: int, ple_dir: str, link_ple: bool):
    # Remove any old ple_*.safetensors symlinks or files if switching PLE bit width
    for old in glob.glob(os.path.join(out_dir, "ple_b*.safetensors")):
        if os.path.islink(old):
            os.unlink(old)

    if ple_bits == 8:
        raise ValueError("Use dedicated PLE shards for ple_bits in {2,3,4}")

    ple_files = sorted(glob.glob(os.path.join(ple_dir, f"ple_b{ple_bits}_*.safetensors")))
    if len(ple_files) != 8:
        raise RuntimeError(f"Expected 8 ple_b{ple_bits}_*.safetensors in {ple_dir}, found {len(ple_files)}")

    for pf in ple_files:
        dst = os.path.join(out_dir, os.path.basename(pf))
        if os.path.abspath(pf) != os.path.abspath(dst):
            if os.path.lexists(dst):
                os.unlink(dst)
            if link_ple:
                os.symlink(os.path.abspath(pf), dst)
            else:
                shutil.move(pf, dst)

    st_files = sorted(glob.glob(os.path.join(out_dir, "nonmoe_q8_*.safetensors"))) + sorted(
        glob.glob(os.path.join(out_dir, f"ple_b{ple_bits}_*.safetensors"))
    )
    weight_map = {}
    total_size = 0
    for stp in st_files:
        bn = os.path.basename(stp)
        hdr, _ = read_st_header(stp)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if k in weight_map:
                raise RuntimeError(f"Duplicate tensor {k} in {bn} and {weight_map[k]}")
            weight_map[k] = bn
            a, b = v["data_offsets"]
            total_size += b - a

    idx_obj = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx_obj, f, indent=2)

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    q_obj = {"group_size": 32, "bits": 8, "mode": "affine"}
    if ple_bits != 8:
        for s in range(128):
            key = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}"
            q_obj[key] = {"group_size": 32, "bits": ple_bits, "mode": "affine"}
    # Carry the source's per-tensor width overrides (newer mlx-vlm stores router gates at 8-bit/64)
    # for every tensor still in the packaged index; PLE shards get their own entries above.
    src_q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    packaged = {k.rsplit(".", 1)[0] for k in weight_map}
    for k, v in src_q.items():
        if isinstance(v, dict) and ".ngram_embedding." not in k and k in packaged and k not in q_obj:
            q_obj[k] = v
    cfg["quantization"] = q_obj
    cfg["quantization_config"] = q_obj
    if isinstance(cfg.get("text_config"), dict):
        if "quantization" in cfg["text_config"]:
            cfg["text_config"]["quantization"] = q_obj
        if "quantization_config" in cfg["text_config"]:
            cfg["text_config"]["quantization_config"] = q_obj
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    tffpw_bytes = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out_dir, "*.tffpw")))
    tbl_p = os.path.join(out_dir, "pw_v3_tables.bin")
    if os.path.exists(tbl_p):
        tffpw_bytes += os.path.getsize(tbl_p)
    tot = total_size + tffpw_bytes
    print(
        f"Configured {out_dir} (PLE {ple_bits}-bit/32): safetensors={total_size/(1<<30):.2f} GiB + "
        f"TFQ3={tffpw_bytes/(1<<30):.2f} GiB => TOTAL={tot/(1<<30):.2f} GiB ({tot/1e9:.2f} GB)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Package standalone Qwen3.8-Flash-Next TFQ3 checkpoint.")
    ap.add_argument("--src", required=True, help="Source MLX 8-bit/32 Qwen3.8-Flash-Next checkpoint directory")
    ap.add_argument("--out", required=True, help="Output TFQ3 checkpoint directory")
    ap.add_argument("--extract-nonmoe", action="store_true", help="Copy non-MoE, MTP non-MoE, and vision weights at 8-bit/32")
    ap.add_argument("--write-index", action="store_true", help="Write model.safetensors.index.json and config.json")
    ap.add_argument("--ple-bits", type=int, default=3, choices=[3, 4], help="PLE bit width (default: 3)")
    ap.add_argument("--ple-dir", required=False, default="", help="Directory containing requantized PLE shards")
    ap.add_argument("--move-ple", action="store_true", help="Move PLE shards into --out instead of symlinking")
    args = ap.parse_args()
    if args.extract_nonmoe:
        extract_nonmoe(args.src, args.out)
    if args.write_index:
        ple_dir = args.ple_dir or args.out
        write_index_and_config(args.src, args.out, args.ple_bits, ple_dir, link_ple=not args.move_ple)


if __name__ == "__main__":
    main()
