#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# ncu_run.sh <out-name> <kernel-regex> <launch-skip> <count> <harness.py>  -- Track M, bs2, real clocks
NCU=/usr/local/cuda/bin/ncu
$NCU --target-processes all --profile-from-start off --clock-control none \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats --section Occupancy --section LaunchStats \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,gpu__time_duration.sum,sm__cycles_active.avg \
  -k "regex:$2" --launch-skip "$3" -c "$4" --export "$1" --force-overwrite \
  "${PYTHON:-python}" "$5"
