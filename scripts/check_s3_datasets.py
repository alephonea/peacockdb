#!/usr/bin/env python3
"""check_s3_datasets.py — Tier B: cheap S3 metadata check for the large datasets.

NO download, no disk: pure list/head against the bucket. For each expected bucket it
asserts the expected parquet files are present, ContentLength > 0, and — when the bucket
carries a manifest.json (filename -> byte size, written by the Phase-3 upload) — that the
sizes match.

manifest.json is the COMPLETION SENTINEL. Phase 3 uploads all parquet FIRST and the
manifest LAST, so its mere presence means "this dataset is fully published". That folds
empty / mid-upload / complete into one rule and removes the partial-upload red window:
  * NO manifest.json  -> SKIP (pass): "not yet published" (covers an empty bucket AND a
                         bucket mid-upload — a multi-hour window that must not be red).
  * manifest PRESENT  -> assert every file it lists exists with matching ContentLength.
                         The ONLY way to be red is "manifest says complete but a file is
                         missing/zero/wrong-size" = a genuinely broken publish.

Auth: AWS_* env (from repo secrets) + an explicit --endpoint-url (Nebius), region
eu-north-1. Every aws call passes the endpoint or it would hit real AWS.

Usage:  python3 scripts/check_s3_datasets.py --endpoint https://storage.eu-north1.nebius.cloud:443
"""
import argparse
import json
import subprocess
import sys

REGION = "eu-north-1"

TPCH = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
TPCDS = [
    "call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer",
    "customer_address", "customer_demographics", "date_dim", "household_demographics",
    "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
    "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
    "web_sales", "web_site",
]
# bucket -> expected <table>.parquet basenames (flat at bucket root)
BUCKETS = {
    "tpch-sf40": [f"{t}.parquet" for t in TPCH],
    "tpch-sf200": [f"{t}.parquet" for t in TPCH],
    "tpcds-sf200": [f"{t}.parquet" for t in TPCDS],
}


def aws(endpoint, *args):
    r = subprocess.run(["aws", "--endpoint-url", endpoint, "--region", REGION, *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def list_objects(endpoint, bucket):
    rc, out, err = aws(endpoint, "s3api", "list-objects-v2", "--bucket", bucket,
                       "--output", "json")
    if rc != 0:
        return None, err.strip()
    data = json.loads(out or "{}")
    return {o["Key"]: o["Size"] for o in data.get("Contents", [])}, None


def get_manifest(endpoint, bucket, objs):
    if "manifest.json" not in objs:
        return None
    rc, out, err = aws(endpoint, "s3api", "get-object", "--bucket", bucket,
                       "--key", "manifest.json", "/dev/stdout")
    if rc != 0:
        return None
    try:
        # get-object prints metadata json AFTER the body to /dev/stdout; parse the body,
        # which is the first JSON object. Fall back to a fresh fetch via s3 cp if needed.
        return json.loads(out.split("\n{")[0])
    except Exception:
        rc, out, _ = aws(endpoint, "s3", "cp", f"s3://{bucket}/manifest.json", "-")
        try:
            return json.loads(out)
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    endpoint = ap.parse_args().endpoint

    failures = 0
    for bucket, expected in BUCKETS.items():
        print(f"== {bucket} ==")
        objs, err = list_objects(endpoint, bucket)
        if objs is None:
            print(f"  FAIL: cannot list bucket ({err})")
            failures += 1
            continue
        # manifest.json is the completion sentinel: absent => empty OR mid-upload => SKIP.
        if "manifest.json" not in objs:
            data_n = len([k for k in objs if k != "manifest.json"])
            print(f"  SKIP: no manifest.json — not yet published ({data_n} objects present) — pass")
            continue
        manifest = get_manifest(endpoint, bucket, objs)
        if not manifest:
            print("  FAIL: manifest.json present but unreadable/empty")
            failures += 1
            continue
        print(f"  manifest lists {len(manifest)} files; asserting against {len(objs)} objects")
        bad = 0
        # every file the manifest claims must exist, be non-zero, and match its size
        for f, sz in manifest.items():
            if f not in objs:
                print(f"  FAIL: manifest lists {f} but it is missing"); bad += 1
            elif objs[f] <= 0:
                print(f"  FAIL: {f} ContentLength={objs[f]}"); bad += 1
            elif objs[f] != sz:
                print(f"  FAIL: {f} size {objs[f]} != manifest {sz}"); bad += 1
        # and the expected parquet set must be covered by the manifest
        missing = [f for f in expected if f not in manifest]
        if missing:
            print(f"  FAIL: manifest missing expected parquet {missing}"); bad += len(missing)
        if bad == 0:
            print(f"  PASS: all {len(expected)} parquet present, non-zero, sizes match manifest")
        failures += bad

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
