"""从自己训练的 checkpoint 生成文本并记录采样设置。"""

import argparse

import torch

from generation_lab_code.sampling import generate
from tokenizer_lab_code.tokenizer_lab.storage import load_model, write_json
from training_lab_code.checkpoint import read_checkpoint
from training_lab_code.data import Dataset
from transformer_lab_code.model import ModelConfig, TransformerLM


def run(checkpoint, data, prompt, out, device="cpu", max_new_tokens=128, temperature=1.0, top_p=0.9, seed=42):
    if torch.device(device).type == "cpu":
        torch.set_num_threads(2)
    dataset = Dataset(data)
    saved = read_checkpoint(checkpoint)
    if saved["extra"]["data_identity"] != dataset.identity:
        raise ValueError("checkpoint 和数据/tokenizer 不配套")
    model = TransformerLM(ModelConfig(**saved["extra"]["config"]["model"]), device=device)
    model.load_state_dict(saved["model"])
    tokenizer = load_model(dataset.tokenizer_path)
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt 不能为空；也可以显式使用 <|endoftext|>")
    tokens = generate(
        model, torch.tensor([ids], device=device), max_new_tokens, temperature, top_p, dataset.manifest["eos_id"], seed
    )[0].tolist()
    record = dict(
        prompt=prompt,
        prompt_ids=ids,
        continuation_ids=tokens[len(ids) :],
        continuation=tokenizer.decode(tokens[len(ids) :]),
        text=tokenizer.decode(tokens),
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_new_tokens=max_new_tokens,
        checkpoint_step=saved["iteration"],
        data_identity=dataset.identity,
        tokenizer_sha256=dataset.manifest["tokenizer_sha256"],
    )
    write_json(out, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "data", "prompt", "out"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        run(
            args.checkpoint,
            args.data,
            args.prompt,
            args.out,
            args.device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )["text"]
    )


if __name__ == "__main__":
    main()
