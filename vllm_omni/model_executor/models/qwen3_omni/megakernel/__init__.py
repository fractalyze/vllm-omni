# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3-Omni's thinker, talker and code predictor on single-launch megakernels.

thinker.py runs the thinker's decode and prefill for
Qwen3OmniMoeThinkerForConditionalGeneration behind the
VLLM_OMNI_THINKER_MEGAKERNEL switches; talker.py runs the talker's decode steps
and its code predictor behind VLLM_OMNI_TALKER_MEGAKERNEL and
VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL (vllm_omni/envs.py). csrc/ holds the
kernels, built on first use by _ext.py for sm_120a.

Taken from fractalyze/decode-mk at d48ef71a268df8ad46f8e47cab0f57a8c3fbe73a
(branch feat/qwen3omni-megakernels): the Qwen3-Omni slice of s2mk/ and its
vLLM-Omni patches, integration/vllm_omni/s2mk_thinker_patch.py,
s2mk_talker_patch.py and s2mk_cp_patch.py.
"""
