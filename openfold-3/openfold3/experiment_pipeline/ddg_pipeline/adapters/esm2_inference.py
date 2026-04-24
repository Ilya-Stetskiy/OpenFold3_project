from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _load_model():
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    return model, alphabet


def _pseudo_log_likelihood(model, alphabet, sequence: str) -> float:
    import torch

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("sequence", sequence)])
    tokens = tokens.to(next(model.parameters()).device)
    log_probability = 0.0
    with torch.no_grad():
        for position in range(1, len(sequence) + 1):
            masked_tokens = tokens.clone()
            masked_tokens[0, position] = alphabet.mask_idx
            logits = model(masked_tokens)["logits"][0, position]
            log_probs = torch.log_softmax(logits, dim=-1)
            residue_token = int(tokens[0, position].item())
            residue_log_prob = float(log_probs[residue_token].item())
            if math.isnan(residue_log_prob):
                raise ValueError(f"NaN residue log-probability at position {position}")
            log_probability += residue_log_prob
    if math.isnan(log_probability):
        raise ValueError("NaN sequence log-probability")
    return float(log_probability)


def _mode_check(output_path: Path) -> int:
    payload: dict[str, object] = {}
    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import esm

        payload["esm_import"] = True
        payload["esm_module"] = getattr(esm, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        payload["esm_import"] = False
        payload["esm_error"] = f"{type(exc).__name__}: {exc}"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _mode_infer(prepared_input_path: Path, output_path: Path) -> int:
    import torch

    payload = json.loads(prepared_input_path.read_text(encoding="utf-8"))
    model, alphabet = _load_model()
    first_param = next(model.parameters())
    wt_sequence = str(payload["wt_sequence"])
    mutant_sequence = str(payload["mutant_sequence"])
    wt_logp = _pseudo_log_likelihood(model, alphabet, wt_sequence)
    mutant_logp = _pseudo_log_likelihood(model, alphabet, mutant_sequence)
    ddg_raw = mutant_logp - wt_logp
    if math.isnan(ddg_raw):
        raise ValueError("NaN ESM2 ddG value")
    result = {
        "loaded": True,
        "device": str(first_param.device),
        "dtype": str(first_param.dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "alphabet_size": len(alphabet),
        "torch_version": torch.__version__,
        "wt_sequence": wt_sequence,
        "mutant_sequence": mutant_sequence,
        "wt_log_probability": wt_logp,
        "mutant_log_probability": mutant_logp,
        "ddg_raw": ddg_raw,
        "ddg": ddg_raw,
        "unit_raw": "log-probability",
        "unit": "log-probability",
        "sign_convention_raw": "mutant_minus_wildtype",
        "sign_convention": "mutant_minus_wildtype",
        "normalization_note": "ESM2 ddG is the pseudo-log-likelihood difference and is not converted to kcal/mol.",
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local ESM2 inference helper for ddg_pipeline.")
    parser.add_argument("--mode", choices=("check", "infer"), required=True)
    parser.add_argument("--prepared-input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-home", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.torch_home:
        os.environ["TORCH_HOME"] = args.torch_home
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        return _mode_check(args.output)
    if args.prepared_input is None:
        raise ValueError("--prepared-input is required for infer mode")
    return _mode_infer(args.prepared_input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
