# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if hasattr(torch, dtype_name):
        attr = getattr(torch, dtype_name)
        if isinstance(attr, torch.dtype):
            return attr
    raise ValueError(f"Unsupported dtype '{dtype_name}' for teacher fusion")


class HFTeacherFusionLogitsProcessor(LogitsProcessor):
    """HF logits processor that mixes student logits with a frozen teacher."""

    def __init__(
        self,
        model_path: str,
        alpha: float,
        temperature: float,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.temperature = temperature
        self.device = get_torch_device()
        dtype = _resolve_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores is None:
            return scores

        teacher_logits = self._compute_teacher_logits(input_ids, target_dim=scores.size(-1))
        teacher_logits = teacher_logits.to(device=scores.device, dtype=scores.dtype)

        mixed = self.alpha * teacher_logits + (1 - self.alpha) * scores
        if self.temperature not in (None, 1.0):
            mixed = mixed / self.temperature
        return mixed

    @torch.no_grad()
    def _compute_teacher_logits(self, input_ids: torch.LongTensor, target_dim: int) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids.to(self.device))
        teacher_logits = outputs.logits[:, -1, :]
        if teacher_logits.size(-1) != target_dim:
            raise ValueError(
                "Teacher vocab size does not match student vocab size: "
                f"teacher={teacher_logits.size(-1)}, student={target_dim}"
            )
        return teacher_logits


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.teacher_fusion_processor = self._maybe_build_teacher_fusion_processor()

    def _maybe_build_teacher_fusion_processor(self):
        teacher_cfg = getattr(self.config, "teacher_fusion", None)
        if teacher_cfg is None or not getattr(teacher_cfg, "enable", False):
            return None

        teacher_model_path = getattr(teacher_cfg, "teacher_model_path", None)
        if not teacher_model_path:
            raise ValueError("teacher_model_path must be provided when enabling teacher fusion.")

        return HFTeacherFusionLogitsProcessor(
            model_path=teacher_model_path,
            alpha=getattr(teacher_cfg, "alpha", 0.5),
            temperature=getattr(teacher_cfg, "temperature", 1.0),
            torch_dtype=getattr(teacher_cfg, "torch_dtype", "bfloat16"),
            trust_remote_code=getattr(teacher_cfg, "trust_remote_code", False),
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # already repeat in ray_trainer
                # https://github.com/volcengine/verl/blob/2fdfbdcba6f2e076f64bc47922d8fe6cf7dc7da5/verl/trainer/ppo/ray_trainer.py#L1117
                "num_return_sequences": 1,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)
        logits_processor = None
        if self.teacher_fusion_processor is not None:
            logits_processor = LogitsProcessorList([self.teacher_fusion_processor])

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                logits_processor=logits_processor,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
