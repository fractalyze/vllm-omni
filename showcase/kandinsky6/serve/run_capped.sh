#!/bin/bash
# Run a server under a hard host-memory cap: if it exceeds MemoryMax the kernel
# kills the server's cgroup, not this session, tmux or the other jobs on the box.
# A 54 GB worker on this 59 GB host took the whole session down at 20:50.
exec systemd-run --user --scope --quiet -p MemoryMax=${K6_MEMMAX:-42G} -p MemorySwapMax=0 "$@"
