# Adaptive Multi-Agent Orchestrator Benchmarks

Benchmark harness for evaluating an adaptive multi-agent orchestrator on OfficeBench and GAIA.

The orchestrator itself lives in [adaptive-multi-agent-orchestrator-core](https://github.com/mlukei/adaptive-multi-agent-orchestrator-core) and is installed as a dependency. This repository contains everything around the benchmark setup: environments, agent cards, configuration files, logging generated outputs, and analysis code for the paper experiments.

## Reproducibility

Runs depend on several moving parts: the orchestrator dependency revision, the task split seed, memory schema, model deployment names, configuration files, and the state of the Postgres memory database.

### Prerequisites

- **uv** - manages the environment and the lockfile:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Docker** - required for the Postgres memory database (both benchmarks) and for
  the per-task OfficeBench containers. The OfficeBench image is built automatically
  on the first run.
- **Tesseract and FFmpeg** - host binaries the GAIA tools need for OCR and
  for Whisper audio transcription:

  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr ffmpeg
  ```

- **A Hugging Face account** with the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) terms accepted, if you want to run GAIA (see step 4).

### 1. Create the environment

```bash
uv sync
```

This installs every Python dependency from `uv.lock`, including the orchestrator
itself at its pinned revision.

`.python-version` pins Python 3.11.

### 2. Configure Azure OpenAI credentials

```bash
cp .env.example .env
```

Then fill in the endpoint, key and API version.

Credentials and deployment names are split:

- **`.env` holds the connection**: endpoint, API key and API version.
- **The YAML config names the deployments**, per role, under `llm`:
  `orchestrator`, `agents`, `curator`, `judge` and `embedder` each take a
  `model_name`, which is the Azure deployment to use. This is what lets one
  experiment run the orchestrator on a different model than its sub-agents.

### 3. Start the memory database

The adaptive orchestrator persists its memory in Postgres:

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

### 4. Provision GAIA attachment files (GAIA only)

GAIA attachment files must be downloaded from the Hugging Face dataset.

```bash
uv run python scripts/provision_gaia_testbeds.py
```


## Running Experiments

All runs go through `run.py`, which loads experiment configuration from YAML files:

```bash
uv run python run.py --benchmark officebench --config <config.yaml> 
uv run python run.py --benchmark gaia        --config <config.yaml>
```


### YAML Configuration

See [configs/template.yaml](configs/template.yaml) for a commented configuration template.
Copy it into `configs/officebench/` or `configs/gaia/` and edit it per experiment.

The exact YAML configuration files used for the experiments reported in the thesis are available in the [`Configs/` directory of the experimental artifacts](https://tubcloud.tu-berlin.de/s/Ne48mdXqCdQRjnL).

### Reproducing the paper splits

The published results are averaged over three splits. Each split is one
deterministic stratified random draw over the full task set (Monte Carlo
cross-validation). Splits 1, 2 and 3 use seeds
**42, 43 and 44**, with `train_fraction: 0.4` / `test_fraction: 0.6` (the values
in [configs/template.yaml](configs/template.yaml)).


Each split is a separate config file. Split 2's training config sets:

```yaml
execution:
  task_split:
    name: train
    train_fraction: 0.4
    test_fraction: 0.6
    seed: 43
```

and its testing config sets `name: test` with the same `seed: 43`, typically with
`memory_schema` pointed at the split-2 training variant. Then run each phase with
its own config file:

```bash
uv run python run.py --benchmark gaia --config configs/gaia/split2_train.yaml
uv run python run.py --benchmark gaia --config configs/gaia/split2_test.yaml
```


## Repository Layout

```text
agents/
  officebench/           rich and sparse OfficeBench agent cards
  gaia/                  rich and sparse GAIA agent cards
  noise/                 shared rich and sparse noise-agent cards

analysis/
  loader.py              lightweight CSV loader for paper metrics
  metrics/               pure calculation modules
  *.ipynb                one notebook per paper metric family

annotations/             gold agent labels for OfficeBench and GAIA

apps/
  */                     OfficeBench app/tool implementations

benchmarks/
  officebench/           batch runner, single-task worker, policy, agents, evaluation
  gaia/                  batch runner, single-task worker, policy, agents, evaluation
  shared/                base policy, card, and registration helpers

configs/
  template.yaml          commented config template
  officebench/           local OfficeBench experiment configs (gitignored)
  gaia/                  local GAIA experiment configs (gitignored)

docker/
  Dockerfile             OfficeBench task container image

scripts/
  provision_gaia_testbeds.py   fetches GAIA attachments into local testbeds

logger/
  run_logger.py          CSV run summaries and policy logging helpers

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

The analysis reads the curated result snapshots under `output/`.
`analysis/metrics/` holds the calculation modules, and one notebook per metric
family calls into them:

| Notebook | Produces | Metric module |
| --- | --- | --- |
| `overall_performance.ipynb` | Main results table: success rate per difficulty tier, delegations, tokens, cost, distractor share | `metrics/overall.py` |
| `task_wise_comparison.ipynb` | Paired per-task condition comparison with McNemar tests and pairwise p-values | `metrics/conditions.py` |
| `error_decomposition.ipynb` | Failure breakdown by error category | `metrics/errors.py` |
| `retrieval.ipynb` | Blueprint and playbook retrieval quality | `metrics/retrieval.py` |
| `judge_agreement.ipynb` | Agreement between the judge gate and task evaluation on the training splits | `metrics/judge.py` |
| `agent_experiment.ipynb` | Dynamic-pool experiment: stability, new-agent adoption, usage shift, playbook activity | `metrics/agents.py` |


## Acknowledgments

This work builds on several existing benchmarks, environments, and open-source projects:

- **GAIA** — Mialon et al., *GAIA: a benchmark for General AI Assistants* ([arXiv:2311.12983](https://arxiv.org/abs/2311.12983)). The GAIA tasks evaluated here come from this benchmark.
- **OfficeBench** — Wang et al., *OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation* ([arXiv:2407.19056](https://arxiv.org/abs/2407.19056), [zlwang-cs/OfficeBench](https://github.com/zlwang-cs/OfficeBench)). The OfficeBench tasks, apps, and testbeds are drawn from this benchmark.
- **InterCode** — Yang et al., *InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback* (NeurIPS 2023, [arXiv:2306.14898](https://arxiv.org/abs/2306.14898), [princeton-nlp/intercode](https://github.com/princeton-nlp/intercode)). OfficeBench's Docker-based execution environment builds on InterCode.
- **fisherman611/gaia-agent** ([github.com/fisherman611/gaia-agent](https://github.com/fisherman611/gaia-agent/tree/main)) — the starting point for the GAIA agent tool functions in [benchmarks/gaia/tools.py](benchmarks/gaia/tools.py) 
- **gaia-scorer** — Roucher, [`scripts/evaluation/gaia_scorer.py`](https://github.com/aymeric-roucher/GAIA/blob/main/scripts/evaluation/gaia_scorer.py). The official GAIA answer-matching logic used to score GAIA results in [benchmarks/gaia/evaluation.py](benchmarks/gaia/evaluation.py) 
- **LEGOMem** — Han et al., *LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation* ([arXiv:2510.04851](https://arxiv.org/abs/2510.04851)). Inspiration for the multi-agent integration on OfficeBench.
