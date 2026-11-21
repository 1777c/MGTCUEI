# MGTCUEI

**Built with Meta Llama 3**

[简体中文](README.md)

MGTCUEI predicts drug–disease associations. A graph encoder builds triple-typed memory, and the Controlled Upper-layer Evidence Injector (CUEI) supplies that memory to a frozen medical language model. The main model predicts with the frozen True and False token rows.

## Data and environment

The release includes entity mappings, node features, disease similarity, graph inputs and both frozen prompt styles. The fixed Train, Validation and Test files are in [data/benchmarks](data/benchmarks), with 10 folds each for Warm, Drug-Cold and Disease-Cold and 5 for Dual-Cold. [data/cold](data/cold) contains the development split used by the ablations.

Use Python 3.11 or later. The verified package versions are in [requirements-lock.txt](requirements-lock.txt). Evidence-model training requires a CUDA GPU with BF16 support.

```bash
python -m pip install -r requirements.txt
```

Fetch the model configuration, tokenizer and all seven weight shards from the [official MMed-Llama-3-8B release](https://huggingface.co/Henrychur/MMed-Llama-3-8B). The download is about 32.1 GB. The weight checksums are in [data/llm_contract](data/llm_contract). Set `LLM_ROOT` to the local model directory or pass `--llm-root` to the main training, evaluation and prediction scripts. The frozen semantic prompt file is distributed through Git LFS.

## Train, evaluate and predict

```bash
export LLM_ROOT=/path/to/MMed-Llama-3-8B
python scripts/train.py --protocol drug_cold_k10 --fold 0 --arm evidence_semantic
```

Use the run directory printed by training in place of `RUN_DIR`. A prediction CSV must contain `pair_id`, `drug_idx` and `disease_idx` for pairs in the frozen prompt cache.

```bash
python scripts/evaluate.py --run RUN_DIR --protocol drug_cold_k10 --fold 0 --block test
python scripts/predict.py --run RUN_DIR --protocol drug_cold_k10 --fold 0 --pairs pairs.csv --output scores.csv
```

The other formal protocol names are `warm_k10`, `disease_cold_k10` and `strict_both_k5`. Use `evidence_basic` for the basic prompt arm or `gnn_only` for the graph-only arm. The [configs](configs) directory records the model, seed, schedule and evaluation settings.

## Ablations

Both comparisons fit on development Train, select the checkpoint and threshold on Dev-A, and report on pooled Dev-B, Dev-C and Dev-D. For a component ablation, run `python scripts/train.py --protocol cold --arm evidence_semantic --reader non_query`. The other component options are `--cuei-span`, `--gate` and `--memory`.

**Graph encoders (Figure 5(e), Table S12)**

```bash
python scripts/s2_graph_executor.py --arms GE01,GE02R,GE03,GE04,GE05,GE06
```

The arms follow the order GCN-1L, GCN-2L matched replay, GCN-3L, GCN-4L, GAT-2L and GraphSAGE-2L. Stages `A,B,FINAL` run upstream graph fitting, joint fitting and pooled evaluation. The graph comparison uses `LLM_ROOT` for MMed-Llama.

**Language backbones (Figure 5(f), Table S13)**

Set `LLM_ROOT`, `META_LLAMA_ROOT` or `BIOBERT_ROOT` to the corresponding local model directory. The recorded sources and revisions for [Meta-Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B/tree/315b20096dc791d381d514deb5f8bd9c8d6d3061) and [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/67c9c25b46986521ca33df05d8540da1210b3256) are in [LANGUAGE_ARMS.json](configs/LANGUAGE_ARMS.json).

```bash
python scripts/run_language_arm.py --arm MMED --stage fit
python scripts/run_language_arm.py --arm MMED --stage dev-final
```

Use `META_LLAMA` or `BIOBERT` for the other arms, one arm per process. The Llama arms use frozen True/False rows. BioBERT uses its WordPiece tokenizer and a trained classifier on the final `[CLS]` state.

## Licensing

The source code of MGTCUEI is released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for information on third-party data, models, and software dependencies.
