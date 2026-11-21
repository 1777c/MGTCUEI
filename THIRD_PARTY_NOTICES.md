# 第三方说明

MGTCUEI 自有源码采用 [MIT 许可证](LICENSE)。第三方数据、模型及软件依赖遵循各自适用的许可证和使用条款。

## 数据来源

本项目使用的数据来源包括 ChEMBL、DrugCentral、DGIdb、BindingDB、CTD、FDA drugsfda、ClinicalTrials.gov、MONDO、Disease Ontology、ClinGen、Orphadata、Open Targets、GWAS Catalog、NCBI Gene 和 Reactome。

- **ChEMBL 37（EMBL-EBI）**。ChEMBL 数据采用 [CC BY-SA 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/) 许可证。本项目对原始记录进行了实体映射、配对标准化、划分及特征和提示构建。发布包中受该许可约束的 ChEMBL 内容及其改编部分按同一许可证提供。相关文件包括 `data/benchmarks/` 和 `data/cold/` 中的样本表及训练关联图、`data/entities/` 中的药物映射、`data/node_features.npy` 中的药物特征，以及 `data/prompts/semantic_compact_v1.parquet` 中的药物靶点信息。来源与署名见 [ChEMBL 官网](https://www.ebi.ac.uk/chembl/)、[ChEMBL 数据许可说明](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing)及 [Zdrazil 等，The ChEMBL Database in 2023](https://doi.org/10.1093/nar/gkad1004)。
- **Comparative Toxicogenomics Database（CTD）**。数据版权归 North Carolina State University。CTD 的证据用于冻结划分和训练关联图中的部分药物–疾病配对，以及 `data/prompts/semantic_compact_v1.parquet` 中的部分疾病基因信息。相关内容适用 [CTD 数据使用条款](https://ctdbase.org/about/legal.jsp)。参见 [CTD 官网](https://ctdbase.org/)、[Davis 等，Comparative Toxicogenomics Database’s 20th anniversary: update 2025](https://doi.org/10.1093/nar/gkae883)及 [CTD 商业许可](https://www.catalystresearch.io/products/ctd/licensing)。

## 模型

**Built with Meta Llama 3**

主模型使用 [MMed-Llama-3-8B](https://huggingface.co/Henrychur/MMed-Llama-3-8B)。[data/llm_contract](data/llm_contract) 中随附的分词器和上游模型接口文件遵循 [Meta Llama 3 Community License Agreement](LICENSE-META-LLAMA-3)，署名见 [NOTICE](NOTICE)。

语言骨干对照使用 [configs/LANGUAGE_ARMS.json](configs/LANGUAGE_ARMS.json) 所列版本的 Meta-Llama-3-8B 和 BioBERT。冻结语义特征由 [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) 生成。

## 软件依赖

Python 依赖列在 [requirements.txt](requirements.txt) 中，各依赖从各自的发行包安装。
