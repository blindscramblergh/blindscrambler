#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

keyword="${1:-hw02}"
scripts_dir="$(cd .. && pwd)"

for run in 1 2 3 4 5; do
  echo "Running training ${run}/5 with keyword '${keyword}'..."
  (cd "${scripts_dir}" && python multiclass_impl.py "${keyword}")
done

echo "Aggregating metrics and generating boxplot..."
(cd "${scripts_dir}" && python multiclass_eval.py "${keyword}")
