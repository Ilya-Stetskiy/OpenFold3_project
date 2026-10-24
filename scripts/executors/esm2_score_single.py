#!/usr/bin/env python3
"""Score one single-site mutation with an ESM2 masked-marginal approximation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scoring-mode", default="masked_marginal", choices=["masked_marginal"])
    parser.add_argument("--max-sequence-length", type=int)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    sequence = str(payload["sequence"]).upper()
    position = int(payload["position"])
    wild_type = str(payload["wild_type_aa"]).upper()
    mutant = str(payload["mutant_aa"]).upper()
    if args.max_sequence_length is not None and len(sequence) > args.max_sequence_length:
        raise SystemExit(f"Sequence length {len(sequence)} exceeds max_sequence_length {args.max_sequence_length}")
    if position < 1 or position > len(sequence):
        raise SystemExit(f"Mutation position {position} outside sequence length {len(sequence)}")
    if sequence[position - 1] != wild_type:
        raise SystemExit(f"Sequence WT mismatch at {position}: sequence has {sequence[position - 1]}, mutation has {wild_type}")

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - server dependency
        raise SystemExit(f"Missing ESM2 runtime dependency: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    model.to(device)
    model.eval()

    masked_sequence = sequence[: position - 1] + tokenizer.mask_token + sequence[position:]
    encoded = tokenizer(masked_sequence, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    mask_positions = (encoded["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if len(mask_positions) != 1:
        raise SystemExit("Expected exactly one mask token in encoded sequence")
    mask_index = int(mask_positions[0, 1])
    with torch.no_grad():
        logits = model(**encoded).logits[0, mask_index]
        log_probs = torch.log_softmax(logits, dim=-1)
    wt_id = tokenizer.convert_tokens_to_ids(wild_type)
    mut_id = tokenizer.convert_tokens_to_ids(mutant)
    score = float((log_probs[mut_id] - log_probs[wt_id]).detach().cpu().item())
    result = {
        "dataset_name": payload.get("dataset_name", ""),
        "method": "esm2",
        "structure_source": "sequence_only",
        "mutation_record_id": payload.get("mutation_record_id", ""),
        "job_id": payload.get("job_id", ""),
        "mutation": payload.get("mutation", ""),
        "chain_id": payload.get("chain_id", ""),
        "sequence_id": payload.get("sequence_id", ""),
        "esm2_score": score,
        "esm2_score_unit": "log_probability_difference",
        "model_name": args.model_name,
        "scoring_mode": args.scoring_mode,
        "parser_status": "success",
        "error_message": "",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
