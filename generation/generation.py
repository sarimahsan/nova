import sys
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import load_config
from models.transformer import TransformerLM


def load_nova_model(config_path, checkpoint_path, device="cuda"):
    """
    Loads TransformerLM model configuration and Safetensors weights.
    """
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = len(tokenizer)

    model = TransformerLM(config)

    if os.path.exists(checkpoint_path):
        state_dict = load_file(checkpoint_path, device="cpu")
        # Restore tied word embeddings if omitted in safetensors
        if "embed.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed.weight"]
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded successfully from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model.to(device)
    model.eval()

    return model, tokenizer, config


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=250,
    temperature=0.75,
    top_k=40,
    top_p=0.90,
    repetition_penalty=1.15,
    recent_penalty=1.05,
    recent_window=32,
    device="cuda"
):
    """
    Fast O(N) text generation using KV-Caching.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("\n--- Generating story ---")
    print(prompt, end="", flush=True)

    # Initial prompt prefill with KV caching enabled
    start_pos = 0
    logits, past_key_values = model(
        input_ids,
        past_key_values=None,
        start_pos=start_pos,
        use_cache=True
    )
    next_token_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Temperature scaling
            next_token_logits = next_token_logits / temperature

            # Repetition penalty across unique history tokens
            for token in torch.unique(input_ids):
                next_token_logits[:, token] /= repetition_penalty

            # Recent token penalty
            recent_tokens = input_ids[0, -recent_window:]
            for token in recent_tokens:
                next_token_logits[:, token] /= recent_penalty

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(
                    next_token_logits,
                    min(top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < values[:, [-1]]] = -float("inf")

            # Top-p (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits,
                    descending=True,
                    dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices.masked_select(sorted_indices_to_remove)
                next_token_logits[:, indices_to_remove] = -float("inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        token_id = next_token.item()
        token = tokenizer.decode([token_id])
        print(token, end="", flush=True)

        if token_id == tokenizer.eos_token_id:
            break

        # Pass only single new token with past_key_values cache
        start_pos = past_key_values[0][0].size(2)
        logits, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            start_pos=start_pos,
            use_cache=True
        )
        next_token_logits = logits[:, -1, :]

    print("\n------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Nova 14M Text Generation with KV-Cache")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default="model/model.safetensors", help="Path to safetensors model")
    parser.add_argument("--prompt", type=str, default="Once upon a time, a little girl named Lily found a shiny key in the garden.", help="Prompt for story generation")
    parser.add_argument("--max_tokens", type=int, default=250, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.90, help="Top-p nucleus sampling parameter")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    checkpoint_path = os.path.join(project_root, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint

    model, tokenizer, _ = load_nova_model(config_path, checkpoint_path, device=device)

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )


if __name__ == "__main__":
    main()
