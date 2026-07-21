#!/usr/bin/env python3
"""check_s3_datasets.py — Tier B: cheap S3 metadata check for the published datasets.

NO data download. It reads each object's parquet FOOTER over ranged GETs (pyarrow's
S3FileSystem does this natively), so it stays seconds-fast and disk-free even though
partsupp.sf40 alone is 27GB.

Checks per bucket:
  * sizes match manifest.json  (ContentLength vs the manifest the upload publishes)
  * expected TABLES present
  * expected COLUMNS present, and for TPC-H the embedding columns are APPENDED LAST
    (the column-index invariant the projection-pushdown goldens depend on)
  * embedding DIMS are the EXTERNAL ones (96/100) — published TPC-H datasets are external
    by contract, so this catches a SYNTHETIC dataset (dim 8) being published by mistake.
    Free: derived from footer num_values/num_rows, no data pages read.
  * ROW COUNTS vs the scale factor, from footer num_rows

Expectations (row-count formulas, column lists, embedding dims) are NOT defined here —
they are imported from dataset_checks, the single source of truth shared with
validate_tpch.py / validate_tpcds.py, so the local and remote notions of "correct" can
never drift apart.

manifest.json remains the COMPLETION SENTINEL:
  * absent  -> SKIP (pass): empty bucket OR mid-upload. Must never be red.
  * present -> assert sizes AND schema AND row counts. Red only on a genuinely broken
               publish.

Usage:  python3 scripts/check_s3_datasets.py --endpoint https://storage.eu-north1.nebius.cloud:443
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_checks as dc

REGION = "eu-north-1"
# bucket name IS the spec: tpch-sf40 -> (tpch, 40). Expectations are computed from it,
# never hardcoded per bucket.
BUCKET_RE = re.compile(r"^(tpch|tpcds)-sf(\d+)$")
BUCKETS = ["tpch-sf40", "tpch-sf200", "tpcds-sf200"]


def aws(endpoint, *args):
    r = subprocess.run(["aws", "--endpoint-url", endpoint, "--region", REGION, *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def list_objects(endpoint, bucket):
    rc, out, err = aws(endpoint, "s3api", "list-objects-v2", "--bucket", bucket, "--output", "json")
    if rc != 0:
        return None, err.strip()
    return {o["Key"]: o["Size"] for o in (json.loads(out or "{}").get("Contents") or [])}, None


def get_manifest(endpoint, bucket):
    rc, out, _ = aws(endpoint, "s3", "cp", f"s3://{bucket}/manifest.json", "-")
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def make_fs(endpoint):
    from pyarrow import fs
    scheme, _, hostport = endpoint.partition("://")
    return fs.S3FileSystem(
        access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        endpoint_override=hostport or endpoint,
        scheme=scheme or "https",
        region=REGION,
    )


def check_bucket(endpoint, bucket, fs):
    import pyarrow.parquet as pq

    m = BUCKET_RE.match(bucket)
    if not m:
        print(f"  FAIL: bucket name '{bucket}' doesn't encode bench/SF"); return 1
    bench, sf = m.group(1), int(m.group(2))

    objs, err = list_objects(endpoint, bucket)
    if objs is None:
        print(f"  FAIL: cannot list bucket ({err})"); return 1
    if "manifest.json" not in objs:
        n = len([k for k in objs if k != "manifest.json"])
        print(f"  SKIP: no manifest.json — not yet published ({n} objects present) — pass")
        return 0
    manifest = get_manifest(endpoint, bucket)
    if not manifest:
        print("  FAIL: manifest.json present but unreadable/empty"); return 1

    bad = 0
    # 1) sizes vs manifest + manifest covers the expected tables
    for f, sz in manifest.items():
        if f not in objs:
            print(f"  FAIL: manifest lists {f} but it is missing"); bad += 1
        elif objs[f] <= 0:
            print(f"  FAIL: {f} ContentLength={objs[f]}"); bad += 1
        elif objs[f] != sz:
            print(f"  FAIL: {f} size {objs[f]} != manifest {sz}"); bad += 1
    expected_tables = dc.tables_for(bench)
    missing = [f"{t}.parquet" for t in expected_tables if f"{t}.parquet" not in manifest]
    if missing:
        print(f"  FAIL: manifest missing expected parquet {missing}"); bad += len(missing)

    # 2) per-table footer checks (schema, embedding dims, row counts) — ranged GETs only
    for t in expected_tables:
        key = f"{t}.parquet"
        if key not in objs:
            continue
        try:
            md = pq.ParquetFile(f"{bucket}/{key}", filesystem=fs).metadata
            schema = pq.ParquetFile(f"{bucket}/{key}", filesystem=fs).schema_arrow
        except Exception as e:
            print(f"  FAIL: {key} footer unreadable ({type(e).__name__}: {e})"); bad += 1
            continue
        names = [f.name for f in schema]

        req, trailing = dc.expected_columns(bench, t)
        miss = [c for c in req if c not in names]
        if miss:
            print(f"  FAIL: {t} missing columns {miss}"); bad += 1
        if trailing:
            miss_e = [c for c in trailing if c not in names]
            if miss_e:
                print(f"  FAIL: {t} missing embedding columns {miss_e}"); bad += 1
            elif names[-len(trailing):] != list(trailing):
                print(f"  FAIL: {t} embedding columns not appended LAST (tail={names[-len(trailing):]})"); bad += 1
            else:
                dims = dc.embedding_dims_from_metadata(md)
                for c in trailing:
                    want = dc.TPCH_EMB_DIMS.get(c)
                    if want is None:      # ps_tag is not a vector
                        continue
                    got = dims.get(c)
                    if got != want:
                        print(f"  FAIL: {t}.{c} dim {got} != expected {want} "
                              f"(published datasets must be EXTERNAL, not synthetic)"); bad += 1

        exp, tol = dc.expected_rows(bench, t, sf)
        if exp is not None:
            n = md.num_rows
            ok = abs(n - exp) <= tol * exp if tol else n == exp
            if not ok:
                print(f"  FAIL: {t} row count {n:,} != {exp:,}" + (f" (±{tol:.0%})" if tol else "")); bad += 1

    if bad == 0:
        print(f"  PASS: {len(expected_tables)} tables — sizes match manifest, columns+"
              f"embedding dims OK, row counts match SF={sf}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--buckets", nargs="*", default=BUCKETS)
    args = ap.parse_args()
    fs = make_fs(args.endpoint)
    failures = 0
    for b in args.buckets:
        print(f"== {b} ==")
        failures += check_bucket(args.endpoint, b, fs)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
