from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPERS_DIR = PROJECT_ROOT / "helpers"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from of_notebook_lib import RuntimeConfig  # noqa: E402
from of_notebook_lib.workflows import run_server_end_to_end_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-json",
        type=Path,
        default=PROJECT_ROOT / "server_smoke" / "query_ubiquitin.json",
    )
    parser.add_argument("--query-id", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="ubiquitin_server_e2e")
    parser.add_argument("--mutation-chain-id", type=str, default="A")
    parser.add_argument("--position-1based", type=int, default=2)
    parser.add_argument("--amino-acids", type=str, default="AG")
    parser.add_argument("--runner-yaml", type=str, default=None)
    parser.add_argument("--inference-ckpt-path", type=str, default=None)
    parser.add_argument("--inference-ckpt-name", type=str, default=None)
    parser.add_argument("--repo-dir", type=str, default=None)
    parser.add_argument("--use-templates", action="store_true")
    parser.add_argument("--no-msa-server", action="store_true")
    parser.add_argument("--single-only", action="store_true")
    args = parser.parse_args()

    query_payload = json.loads(args.query_json.read_text(encoding="utf-8"))
    queries = query_payload["queries"]
    query_id = args.query_id or next(iter(queries))
    molecules = queries[query_id]["chains"]

    runtime = RuntimeConfig()
    result = run_server_end_to_end_case(
        runtime=runtime,
        experiment_name=args.experiment_name,
        molecules=molecules,
        mutation_chain_id=args.mutation_chain_id,
        position_1based=args.position_1based,
        amino_acids=args.amino_acids,
        use_templates=args.use_templates,
        use_msa_server=not args.no_msa_server,
        num_diffusion_samples=1,
        num_model_seeds=1,
        runner_yaml=args.runner_yaml,
        inference_ckpt_path=args.inference_ckpt_path,
        inference_ckpt_name=args.inference_ckpt_name,
        repo_dir=args.repo_dir,
        run_screening=not args.single_only,
    )
    print(f"summary_path={result.summary_path}")


if __name__ == "__main__":
    main()
