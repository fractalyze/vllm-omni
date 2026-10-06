#!/bin/bash
# Sample host RSS of every vllm process, host available memory and GPU memory
# every 2 s into $1 until no vllm process is left (or $2 seconds pass).
out=$1; limit=${2:-1800}; t0=$(date +%s)
echo "t_s rss_total_gb avail_gb gpu_mib" > "$out"
while :; do
  rss=$(ps -eo rss,comm | awk '$2 ~ /vllm|VLLM/ {s+=$1} END{printf "%.1f", s/1048576}')
  avail=$(free -m | awk '/^Mem:/{printf "%.1f", $7/1024}')
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$(( $(date +%s) - t0 )) $rss $avail $gpu" >> "$out"
  [ $(( $(date +%s) - t0 )) -gt $limit ] && break
  sleep 2
done
