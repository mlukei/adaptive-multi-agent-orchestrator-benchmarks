# Adaptive Multi-Agent Orchestrator Benchmarks

Benchmark harness for evaluating an adaptive multi-agent orchestrator on OfficeBench and GAIA.

The orchestrator itself lives in [adaptive-multi-agent-orchestrator-core](https://github.com/mlukei/adaptive-multi-agent-orchestrator-core) and is installed as a dependency. This repository contains everything around the benchmark setup: environments, agent cards, configuration files, logging generated outputs, and analysis code for the paper experiments.

## Reproducibility

Runs depend on several moving parts: the orchestrator dependency revision, the task split seed, memory schema, model deployment names, configuration files, and the state of the Postgres memory database.

### 1. Install dependencies

1. Install host system dependencies used by GAIA tools (Tesseract for OCR and
   FFmpeg for Whisper audio transcription):

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg
```

2. Use Python 3.11:

```bash
uv sync --python 3.11
```

3. Configure Azure OpenAI credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
AZURE_OPENAI_ENDPOINT="https://..."
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT="..."
```

4. Start the local memory database for the adaptive orchestrator:

```bash
docker run --name orch-bench-postgres \
  -e POSTGRES_USER=orchestrator \
  -e POSTGRES_PASSWORD=orchestrator \
  -e POSTGRES_DB=orchestrator \
  -p 5433:5432 \
  -d pgvector/pgvector:pg16
```

If the container already exists, start it with:

```bash
docker start orch-bench-postgres
```

## Running Experiments

All runs go through `run.py`, which loads experiment configuration from YAML files:

```bash
uv run python run.py --benchmark officebench --config <config.yaml> [flags]
uv run python run.py --benchmark gaia        --config <config.yaml> [flags]
```

CLI flags override YAML defaults for the current invocation.

### YAML Configuration

See [configs/template.yaml](configs/template.yaml) for a commented configuration template.
Copy it into `configs/officebench/` or `configs/gaia/` and edit it per experiment.
Those experiment directories are gitignored, so local credentials and run-specific
settings are not published.

Key sections:

- **`llm`**: model deployments, Azure credentials, request timeout, pricing for cost logging.
- **`orchestrator`**: memory schema, feature toggles, agent card paths, noise agents, profiled capability bullets.
- **`docker`**: OfficeBench image/container settings (OfficeBench only).
- **`execution`**: task root directory, output mode, timeouts, task split configuration.

Results are written to `results/{orchestrator.variant}.csv`. For example, a config with:

```yaml
orchestrator:
  variant: "sparse/adaptive_train"
```

writes to:

```text
results/sparse/adaptive_train.csv
```

Postgres schemas use the variant with slashes converted to underscores (e.g., `sparse_adaptive_train`).

### Common CLI Flags

- `--start-index N --end-index M`: run task slice `[N, M)`.
- `--task-ids 0 5 9`: run specific task indices.
- `--task-split train|test`: override the split in the config.
- `--split-seed 42`: deterministic shuffle seed.
- `--mode default|force_new`: skip existing outputs or rerun.
- `--task-timeout-seconds 900`: hard timeout per task.
- `--retry-missing | --retry-timeouts | --retry-errors`: rerun only missing,
  timed-out, or failed tasks recorded in `results/{variant}.csv`.
- `--experiment-name name`: override the OfficeBench experiment name.

OfficeBench runs each task in an isolated Docker container. GAIA runs on the host.

## Repository Layout

```text
agents/
  officebench/           rich and sparse OfficeBench cards, noise cards, profiled bullets
  gaia/                  rich and sparse GAIA cards, noise cards, profiled bullets

analysis/
  loader.py              lightweight CSV loader for paper metrics
  metrics/               pure calculation modules
  *.ipynb                one notebook per paper metric family

annotations/             gold agent labels for OfficeBench and GAIA

apps/
  */                     OfficeBench app/tool implementations

benchmarks/
  officebench/           OfficeBench runner, policy, agents, environment, evaluation
  gaia/                  GAIA runner, policy, agents, evaluation
  shared/                shared policy, runner, card, registration helpers

configs/
  template.yaml          commented config template
  officebench/           local OfficeBench experiment configs (gitignored)
  gaia/                  local GAIA experiment configs (gitignored)

docker/
  Dockerfile             OfficeBench task container image

logger/
  *.py                   CSV logging and run-summary helpers

runtime/
  *.py                   config loading, CLI flags, task selection, subprocess execution

tasks/
  officebench/           OfficeBench task definitions and testbeds
  gaia/                  GAIA task definitions

output/
  officebench/           OfficeBench paper-analysis CSV snapshots
  gaia/                  GAIA paper-analysis CSV snapshots

results/
  */                     generated run CSVs and experiment summaries
```

## Analysis

The paper-metric analysis lives in `analysis/`:

```text
analysis/
  metrics/
    conditions.py
    overall.py
    retrieval.py
    errors.py
    agents.py
    judge.py
    paths.py
  overall_performance.ipynb
  retrieval.ipynb
  error_decomposition.ipynb
  agent_experiment.ipynb
  conversion.ipynb
  judge_agreement.ipynb
  selection_and_shell_usage.ipynb
  task_wise_comparison.ipynb
```

The paper analysis reads the curated OfficeBench and GAIA CSV snapshots under
`output/`.

## Acknowledgments

This work builds on several existing benchmarks, environments, and open-source projects:

- **GAIA** — Mialon et al., *GAIA: a benchmark for General AI Assistants* ([arXiv:2311.12983](https://arxiv.org/abs/2311.12983)). The GAIA tasks evaluated here come from this benchmark.
- **OfficeBench** — Wang et al., *OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation* ([arXiv:2407.19056](https://arxiv.org/abs/2407.19056), [zlwang-cs/OfficeBench](https://github.com/zlwang-cs/OfficeBench)). The OfficeBench tasks, apps, and testbeds are drawn from this benchmark.
- **InterCode** — Yang et al., *InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback* (NeurIPS 2023, [arXiv:2306.14898](https://arxiv.org/abs/2306.14898), [princeton-nlp/intercode](https://github.com/princeton-nlp/intercode)). OfficeBench's Docker-based execution environment builds on InterCode.
- **fisherman611/gaia-agent** ([github.com/fisherman611/gaia-agent](https://github.com/fisherman611/gaia-agent/tree/main)) — the starting point for the GAIA agent tool functions in [benchmarks/gaia/tools.py](benchmarks/gaia/tools.py) (see the attribution header in that file for the exact split of adopted vs. rewritten vs. added tools).
- **LEGOMem** — Han et al., *LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation* ([arXiv:2510.04851](https://arxiv.org/abs/2510.04851)). Inspiration for the multi-agent integration on OfficeBench.
