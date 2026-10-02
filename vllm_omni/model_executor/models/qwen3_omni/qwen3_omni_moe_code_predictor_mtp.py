# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni Code Predictor -- thin wrapper over CodePredictorWrapper."""

from collections.abc import Iterable, Sequence

import torch
from vllm.config import VllmConfig

from vllm_omni import envs
from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from vllm_omni.model_executor.models.qwen3_omni.megakernel.talker import CodePredictorMegakernel
from vllm_omni.platforms import current_omni_platform


class Qwen3OmniMoeTalkerCodePredictor(CodePredictorWrapper):
    """Qwen3-Omni code predictor (no CUDA graphs, VocabParallelEmbedding)."""

    # Set by __init__ when VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL is on.
    megakernel: CodePredictorMegakernel | None = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        cp_config = vllm_config.model_config.hf_config.code_predictor_config
        super().__init__(
            vllm_config=vllm_config,
            cp_config=cp_config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=current_omni_platform.is_npu(),
                use_parallel_embedding=True,
                use_projection=False,
                return_proj_buf=True,
                sampling_mode="stored",
            ),
            talker_hidden_size=cp_config.hidden_size,
            prefix=prefix,
        )
        if envs.VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL:
            self.megakernel = CodePredictorMegakernel(ctas=envs.VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        if self.megakernel is not None:
            self.megakernel.load_weights(self)
        return loaded

    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
        sample_uniforms: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.megakernel is None:
            return super().forward(
                layer0_code,
                layer0_embed,
                last_talker_hidden,
                do_sample,
                temperature,
                top_k,
                top_p,
                generator,
                generators,
                sample_uniforms,
            )
        with torch.inference_mode():
            return self.megakernel.forward(
                self, layer0_code, layer0_embed, last_talker_hidden, generator, generators, sample_uniforms
            )
