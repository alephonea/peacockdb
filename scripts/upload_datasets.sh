#!/bin/bash
#
# Publish the large datasets to S3 (Nebius). LOCAL-ONLY to the generation host
# (the datasets live there; nothing else has the disk). Idempotent + resumable.
#
# CONTRACT (the thing Tier B / scripts/check_s3_datasets.py depends on):
#   For each dataset, upload ALL parquet FIRST, then upload manifest.json LAST. The
#   manifest's presence is the COMPLETION SENTINEL — Tier B treats "no manifest" as
#   not-yet-published (skip) and only asserts once the manifest exists. Uploading it last
#   means a mid-upload bucket is never seen as complete-but-broken.
#   manifest.json = {"<table>.parquet": <byte size>, ...} for every parquet in the dataset.
#
# Layout matches the existing tpch-sf20 convention: parquet FLAT at the bucket root.
#
# Usage (on the generation host):  scripts/upload_datasets.sh [dataset ...]
#   default: all three. e.g. scripts/upload_datasets.sh tpch.sf40
set -euo pipefail

AWS=${AWS:-/home/info/bin/aws}
ENDPOINT=https://storage.eu-north1.nebius.cloud:443
REGION=eu-north-1
BASE=${BASE:-/home/info/peacock-datasets/testdata}

# dataset dir -> bucket
declare -A BUCKET=(
  [tpch.sf40]=tpch-sf40
  [tpch.sf200]=tpch-sf200
  [tpcds.sf200]=tpcds-sf200
)

aws_s3() { "$AWS" --endpoint-url "$ENDPOINT" --region "$REGION" "$@"; }

DATASETS=("$@"); [ ${#DATASETS[@]} -eq 0 ] && DATASETS=(tpch.sf40 tpch.sf200 tpcds.sf200)

for ds in "${DATASETS[@]}"; do
  bucket=${BUCKET[$ds]:-}
  dir="$BASE/$ds"
  [ -n "$bucket" ] || { echo "error: unknown dataset '$ds'" >&2; exit 1; }
  [ -d "$dir" ]    || { echo "error: missing $dir" >&2; exit 1; }
  echo "==> $ds -> s3://$bucket"

  # 1) build the manifest locally (do NOT upload it yet)
  manifest="$dir/manifest.json"
  python3 - "$dir" "$manifest" <<'PY'
import json, os, sys
d, out = sys.argv[1], sys.argv[2]
m = {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".parquet")}
assert m, "no parquet files to publish"
json.dump(m, open(out, "w"), indent=0)
print(f"  manifest: {len(m)} parquet, {sum(m.values())/1e9:.1f} GB total")
PY

  # 2) parquet FIRST (resumable; --size-only so a re-run skips already-uploaded files)
  echo "  syncing parquet ..."
  aws_s3 s3 sync "$dir/" "s3://$bucket/" --exclude "*" --include "*.parquet" --size-only

  # 3) verify every manifest file landed with the right size BEFORE publishing the sentinel
  echo "  verifying remote sizes ..."
  python3 - "$manifest" "$bucket" "$ENDPOINT" "$REGION" "$AWS" <<'PY'
import json, subprocess, sys
manifest, bucket, endpoint, region, aws = sys.argv[1:6]
m = json.load(open(manifest))
r = subprocess.run([aws, "--endpoint-url", endpoint, "--region", region,
                    "s3api", "list-objects-v2", "--bucket", bucket, "--output", "json"],
                   capture_output=True, text=True, check=True)
import json as j
have = {o["Key"]: o["Size"] for o in (j.loads(r.stdout or "{}").get("Contents") or [])}
bad = [f for f, sz in m.items() if have.get(f) != sz]
if bad:
    print("  ERROR: remote mismatch/missing:", bad[:10], file=sys.stderr); sys.exit(1)
print(f"  verified {len(m)} objects match manifest sizes")
PY

  # 4) manifest LAST — the completion sentinel
  echo "  publishing manifest.json (sentinel) ..."
  aws_s3 s3 cp "$manifest" "s3://$bucket/manifest.json"
  echo "  DONE $ds"
done
echo "ALL UPLOADS COMPLETE"
