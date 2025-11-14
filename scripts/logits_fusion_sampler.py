#!/usr/bin/env python3
"""Minimal script to test token-wise logits fusion between two HF models.

Example:
    python scripts/logits_fusion_sampler.py \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_model Qwen/Qwen2.5-3B-Instruct \
        --prompt "How many 'r' characters are in the word 'character'?" \
        --alpha 0.7 \
        --do_sample
"""

import argparse
import math
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keeps only top-k logits per row."""
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, top_k)
    cut_off = values[..., -1, None]
    mask = logits < cut_off
    logits = logits.masked_fill(mask, float("-inf"))
    return logits


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keeps smallest set of tokens whose cumulative prob >= top_p."""
    if top_p is None or top_p <= 0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)
    mask = cumulative_probs > top_p
    # Ensure at least one token remains
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


def fuse_logits(student: torch.Tensor, teacher: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear combination of two logit tensors."""
    if student.shape != teacher.shape:
        raise ValueError(f"Logit shapes differ: {student.shape} vs {teacher.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * teacher + (1 - alpha) * student


def sample_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample or take argmax from logits."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
    return next_token.item()


@torch.no_grad()
def generate_single_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    args: argparse.Namespace,
    eos_token_id: Optional[int],
) -> str:
    """Generate text from a single model using the provided decoding settings."""
    input_ids = prompt_input_ids.clone()
    attention_mask = prompt_attention_mask.clone()

    for _ in range(args.max_new_tokens):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        next_token_id = sample_token(
            logits,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        next_token_tensor = torch.tensor([[next_token_id]], device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token_tensor, device=input_ids.device)],
            dim=-1,
        )

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


@torch.no_grad()
def generate_with_fusion(args: argparse.Namespace) -> dict[str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = getattr(torch, args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    
    m = [{"role": "user", "content": args.prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_input_ids = encoded["input_ids"]
    prompt_attention_mask = encoded["attention_mask"]

    eos_token_id = tokenizer.eos_token_id
    generated: list[int] = []
    input_ids = prompt_input_ids.clone()
    attention_mask = prompt_attention_mask.clone()

    for step in range(args.max_new_tokens):
        student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        fused_logits = fuse_logits(student_logits, teacher_logits, args.alpha)

        next_token_id = sample_token(
            fused_logits,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        generated.append(next_token_id)

        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token_tensor, device=device)],
            dim=-1,
        )

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    output_ids = input_ids[0].tolist()
    fused_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    if args.show_token_ids:
        print("Generated token ids:", output_ids)

    student_text = generate_single_model(
        student, tokenizer, prompt_input_ids, prompt_attention_mask, args, eos_token_id
    )
    teacher_text = generate_single_model(
        teacher, tokenizer, prompt_input_ids, prompt_attention_mask, args, eos_token_id
    )

    return {
        "fused": fused_text,
        "student": student_text,
        "teacher": teacher_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-wise logits fusion sampler.")
    parser.add_argument("--student_model", required=True, help="HuggingFace path/name of on-policy model.")
    parser.add_argument("--teacher_model", required=True, help="HuggingFace path/name of teacher model.")
    parser.add_argument("--prompt", required=True, help="Prompt text to feed both models.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend coefficient for student logits.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k cutoff (0 disables).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus threshold.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate.")
    parser.add_argument("--do_sample", action="store_true", help="Enable multinomial sampling instead of argmax.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--torch_dtype", default="bfloat16", help="Torch dtype string for model weights (e.g., float16).")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to AutoModel/Tokenizer.")
    parser.add_argument("--show_token_ids", action="store_true", help="Print generated token ids for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = generate_with_fusion(args)
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Completion (fused) ===")
    print(outputs["fused"].strip())
    print("\n=== Completion (student) ===")
    print(outputs["student"].strip())
    print("\n=== Completion (teacher) ===")
    print(outputs["teacher"].strip())


if __name__ == "__main__":
    main()
