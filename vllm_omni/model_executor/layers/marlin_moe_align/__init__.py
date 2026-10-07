# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""vLLM PR #48032's deterministic Marlin MoE route alignment on vllm==0.30.0.

align.py is the PR's Python side and Marlin call-site selection; csrc/ holds
its kernels, built by _ext.py on the first alignment. vllm_omni/patch.py
installs it for every Marlin MoE.
"""
