# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3-Omni's thinker decode and prefill on single-launch megakernels.

thinker.py runs them for Qwen3OmniMoeThinkerForConditionalGeneration behind
the VLLM_OMNI_THINKER_MEGAKERNEL switches (vllm_omni/envs.py). csrc/ holds the
kernels, built on first use by _ext.py for sm_120a.

Taken from fractalyze/decode-mk at d48ef71a268df8ad46f8e47cab0f57a8c3fbe73a
(branch feat/qwen3omni-megakernels): the thinker's slice of s2mk/ and its
vLLM-Omni patch, integration/vllm_omni/s2mk_thinker_patch.py.
"""
