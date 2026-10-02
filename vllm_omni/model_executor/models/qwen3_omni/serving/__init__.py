# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3-Omni speech-serving changes that shorten the time to first audio,
each behind its own switch in vllm_omni/envs.py and off by default.

The runner, scheduler, chunk adapter, input processor, model and API call
into these modules only when their switch is on:

- frame0.py (VLLM_OMNI_FRAME0): the talker's first audio frame ships with its
  prefill step;
- frame0_audio.py (VLLM_OMNI_FRAME0_AUDIO): the talker decodes that frame on
  code2wav's weights and sends the first audio chunk straight to the API;
- thinker_yield.py (VLLM_OMNI_THINKER_YIELD): the thinker's decode waits while
  the talker makes a request's first frame;
- talker_preprefill.py (VLLM_OMNI_TALKER_PREPREFILL): the talker prefills all
  but its last prompt position while the thinker prefills;
- thinker_embedding.py: the thinker stage's embedding lookup, which
  VLLM_OMNI_EARLY_CHUNK uses to send the talker its prefill input at the
  thinker's first token, and VLLM_OMNI_TALKER_PREPREFILL its pre chunk.

Taken from fractalyze/decode-mk at d48ef71a268df8ad46f8e47cab0f57a8c3fbe73a
(branch feat/qwen3omni-megakernels), integration/vllm_omni/, where each was a
monkey-patch on vLLM-Omni 69de153.
"""
