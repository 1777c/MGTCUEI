# MGTCUEI

**Built with Meta Llama 3**

[English README](README_EN.md)

MGTCUEI 用于预测药物–疾病关联。图编码器构建三种类型槽位的记忆，受控上层证据注入器（CUEI）将该记忆注入冻结的医学语言模型。主模型使用冻结的 True 和 False 词元权重行进行预测。

## 数据和环境

发布包包含实体映射、节点特征、疾病相似度、图输入和两种冻结提示。固定的训练集、验证集和测试集位于 [data/benchmarks](data/benchmarks)。Warm、Drug-Cold 和 Disease-Cold 各有 10 折，Dual-Cold 有 5 折。[data/cold](data/cold) 包含消融实验使用的开发划分。

使用 Python 3.11 或更高版本。依赖版本见 [requirements-lock.txt](requirements-lock.txt)。训练证据模型需要支持 BF16 的 CUDA GPU。

```bash
python -m pip install -r requirements.txt
```

从 [MMed-Llama-3-8B 官方发布页](https://huggingface.co/Henrychur/MMed-Llama-3-8B) 获取模型配置、分词器和全部 7 个权重分片，下载量约为 32.1 GB。权重校验值见 [data/llm_contract](data/llm_contract)。将 `LLM_ROOT` 指向本地模型目录，也可以向主线训练、评估和预测脚本传入 `--llm-root`。冻结的语义提示文件通过 Git LFS 分发。

## 训练、评估和预测

```bash
export LLM_ROOT=/path/to/MMed-Llama-3-8B
python scripts/train.py --protocol drug_cold_k10 --fold 0 --arm evidence_semantic
```

将 `RUN_DIR` 替换为训练结束时输出的运行目录。用于预测的 CSV 必须包含 `pair_id`、`drug_idx` 和 `disease_idx`，且配对必须存在于冻结提示缓存中。

```bash
python scripts/evaluate.py --run RUN_DIR --protocol drug_cold_k10 --fold 0 --block test
python scripts/predict.py --run RUN_DIR --protocol drug_cold_k10 --fold 0 --pairs pairs.csv --output scores.csv
```

其他正式协议为 `warm_k10`、`disease_cold_k10` 和 `strict_both_k5`。基础提示组使用 `evidence_basic`，纯图模型组使用 `gnn_only`。[configs](configs) 记录模型、随机种子、训练计划和评估设置。

## 消融实验

两组对照均使用开发训练集拟合，在 Dev-A 上选择检查点和阈值，并在合并后的 Dev-B、Dev-C 和 Dev-D 上报告结果。组件消融可运行 `python scripts/train.py --protocol cold --arm evidence_semantic --reader non_query`。其他组件选项为 `--cuei-span`、`--gate` 和 `--memory`。

**图编码器（图 5(e)，表 S12）**

```bash
python scripts/s2_graph_executor.py --arms GE01,GE02R,GE03,GE04,GE05,GE06
```

实验组依次为 GCN-1L、GCN-2L 匹配重放、GCN-3L、GCN-4L、GAT-2L 和 GraphSAGE-2L。`A,B,FINAL` 分别执行上游图编码器拟合、联合拟合和合并评估。此对照使用 `LLM_ROOT` 指定 MMed-Llama 的路径。

**语言骨干（图 5(f)，表 S13）**

将 `LLM_ROOT`、`META_LLAMA_ROOT` 或 `BIOBERT_ROOT` 指向对应的本地模型目录。[Meta-Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B/tree/315b20096dc791d381d514deb5f8bd9c8d6d3061) 和 [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/67c9c25b46986521ca33df05d8540da1210b3256) 的来源和版本记录在 [LANGUAGE_ARMS.json](configs/LANGUAGE_ARMS.json)。

```bash
python scripts/run_language_arm.py --arm MMED --stage fit
python scripts/run_language_arm.py --arm MMED --stage dev-final
```

其他组使用 `META_LLAMA` 或 `BIOBERT`，每个进程运行一组。Llama 组使用冻结的 True/False 权重行。BioBERT 使用自己的 WordPiece 分词器，并在最终 `[CLS]` 状态上训练分类器。

## 许可

本项目源码采用 [MIT 许可证](LICENSE)发布。第三方数据、模型及软件依赖的相关说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
