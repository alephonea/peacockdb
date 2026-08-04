#!/usr/bin/env bash
# Local cost-report preview (was an unscripted workflow): renders the widget HTML from
# the committed goldens + cost-registry.csv. No GPU, no dataset needed.
#   scripts/cost-report-preview.sh [out.html]
set -euo pipefail
out="${1:-cost_report.html}"
cargo run -q -p cost-report -- --sha "$(git rev-parse HEAD)" --html "$out"
echo "wrote $out"
