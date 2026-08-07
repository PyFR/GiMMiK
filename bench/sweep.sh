#!/usr/bin/env bash
# Sweep several operator shapes and report the best OpenCL vs SYCL kernel each.
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
REPS=${1:-100}

# m  k   sparsity   label
CONFIGS=(
  "32 48 0.5"
  "64 64 0.6"
  "96 96 0.7"
  "24 40 0.3"
  "125 64 0.8"
)

printf "%-16s %-22s %10s %10s | %-22s %10s %10s\n" \
  "shape(m x k)" "OpenCL best" "ms" "GB/s" "SYCL best" "ms" "GB/s"
printf '%.0s-' {1..112}; echo

for cfg in "${CONFIGS[@]}"; do
  read m k sp <<< "$cfg"
  GMK_M=$m GMK_K=$k GMK_N=${GMK_N:-200000} GMK_SPARSITY=$sp \
    bash bench/run.sh "$REPS" > bench/build/sweep_out.txt 2>&1 || {
      echo "run failed for $cfg"; cat bench/build/sweep_out.txt; exit 1; }

  # best (min ms) line per platform
  ocl=$(awk '/OpenCL ####/{p=1;next}/SYCL ####/{p=0}p&&/^gmk_/{print $1,$2,$4}' \
        bench/build/sweep_out.txt | sort -k2 -n | head -1)
  scl=$(awk '/SYCL ####/{p=1;next}p&&/^gmk_/{print $1,$2,$4}' \
        bench/build/sweep_out.txt | sort -k2 -n | head -1)

  oname=$(echo $ocl | awk '{gsub("gmk_opencl_[0-9]+_","",$1);gsub("_w1","",$1);print $1}')
  oms=$(echo $ocl | awk '{print $2}');  ogb=$(echo $ocl | awk '{print $3}')
  sname=$(echo $scl | awk '{gsub("gmk_sycl_[0-9]+_","",$1);gsub("_w1","",$1);print $1}')
  sms=$(echo $scl | awk '{print $2}');  sgb=$(echo $scl | awk '{print $3}')

  printf "%-16s %-22s %10s %10s | %-22s %10s %10s\n" \
    "${m} x ${k} (${sp})" "$oname" "$oms" "$ogb" "$sname" "$sms" "$sgb"
done
