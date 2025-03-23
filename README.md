# 物流行业智能信息咨询系统

[![GitHub](https://img.shields.io/badge/GitHub-LogiRAG-blue?logo=github)](https://github.com/Adoubf/LogiRAG)
[![GitHub stars](https://img.shields.io/github/stars/Adoubf/LogiRAG?style=social)](https://github.com/Adoubf/LogiRAG/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Adoubf/LogiRAG?style=social)](https://github.com/Adoubf/LogiRAG/network/members)
[![Python](https://img.shields.io/badge/Python-3.9+-green?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-orange)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/github/license/Adoubf/LogiRAG)](https://github.com/Adoubf/LogiRAG/blob/main/LICENSE)

基于检索增强生成（RAG）技术的物流行业智能信息咨询系统，专为处理和检索物流领域专业知识而设计。能够从多种文档格式中提取、向量化信息，并提供精准的问答服务。

## 1. 系统概述

本系统采用现代RAG架构，通过LangChain框架将大型语言模型（LLM）的生成能力与向量检索的精准性相结合，为用户提供物流行业的专业知识问答服务。

### 核心特性

- **多格式文档处理**：支持PDF、Word、Excel、CSV、TXT等多种文档格式
- **向量数据增量更新**：智能识别文档变化，仅处理新增或修改的内容
- **向量数据管理**：支持按文件粒度删除向量数据，实现灵活管理
- **嵌入向量缓存**：缓存已计算的嵌入向量，大幅提升后续处理速度
- **断点续传**：支持在大规模处理中断后从断点继续，无需重新开始
- **并发处理**：多进程并行处理文档，充分利用计算资源
- **用户友好界面**：基于Streamlit的直观问答界面
- **日志完备**：详细的处理日志，便于监控和问题排查

### 架构设计

```
       ┌─────────────┐         ┌──────────────┐
       │ 多格式文档库 │────────▶│ 文档处理模块  │
       └─────────────┘         └──────┬───────┘
                                      │
                                      ▼
┌───────────────┐           ┌───────────────────┐
│ 嵌入模型      │◀──────────▶│ 文本分块与向量化  │
└───────────────┘           └──────┬────────────┘
                                   │
                                   ▼
┌───────────────┐           ┌───────────────────┐
│ 向量数据管理   │◀──────────▶│ 向量数据库存储   │
└───────────────┘           └──────┬────────────┘
                                   │
                                   ▼
┌───────────────┐           ┌───────────────────┐
│ 大语言模型     │◀──────────▶│ RAG检索与问答链  │
└───────────────┘           └──────┬────────────┘
                                   │
                                   ▼
                            ┌───────────────────┐
                            │ Streamlit用户界面  │
                            └───────────────────┘
```

## 2. 安装指南

### 2.1 基础环境要求

- Python 3.9+
- 50MB以上磁盘空间（不含向量库和模型）
- 推荐8GB以上内存

### 2.2 安装步骤

1. **克隆项目并进入目录**

```bash
git clone https://your-repository/logistics-rag.git
cd logistics-rag
```

2. **创建虚拟环境（推荐）**

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 或使用conda
conda create -n logistics-rag python=3.10
conda activate logistics-rag
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **安装特定文档处理依赖**

根据您需要处理的文档类型，可能需要安装额外依赖：

```bash
# 完整文档处理支持
pip install "unstructured[all-docs]"

# 或根据需要安装特定组件
pip install "unstructured[pdf]"    # PDF支持
pip install "unstructured[docx]"   # Word支持
pip install "unstructured[pptx]"   # PowerPoint支持
```

5. **安装Tesseract OCR（用于处理扫描文档，可选）**

- Windows: 从[此处](https://github.com/UB-Mannheim/tesseract/wiki)下载安装，并确保添加到系统PATH
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

6. **安装并配置Ollama**

从[Ollama官网](https://ollama.ai/download)下载并安装，然后拉取所需模型：

```bash
# 拉取嵌入模型
ollama pull mxbai-embed-large

# 拉取大语言模型
ollama pull llama2-chinese:7b
```

7. **数据准备**

在项目根目录创建`dataset`文件夹，并放入您的物流领域文档：

```bash
mkdir -p dataset
# 将您的PDF、Word、Excel等文档复制到dataset目录
```

## 3. 项目结构

```
03-物流RAG/
├── run.py           # 主入口文件
├── streamlit_app.py # Streamlit直接入口
├── app.py           # 兼容性入口
├── config/          # 配置模块
│   ├── __init__.py
│   └── settings.py  # 集中配置管理
├── data/            # 数据处理模块
│   ├── __init__.py
│   └── vector_store.py  # 向量存储访问
├── prompts/         # 提示词模板
│   ├── __init__.py
│   └── templates.py # 系统提示词
├── rag/             # RAG核心模块
│   ├── __init__.py
│   ├── embedding.py # 嵌入模型管理
│   └── chain.py     # 检索问答链
├── tools/           # 工具模块
│   ├── __init__.py
│   ├── vector_builder.py # 向量构建工具
│   └── vector_manager.py # 向量管理工具
├── ui/              # 用户界面
│   ├── __init__.py
│   └── streamlit_app.py  # Streamlit应用
├── utils/           # 工具函数
│   ├── __init__.py
│   ├── logging_utils.py  # 日志工具
│   └── cache_utils.py    # 缓存工具
├── compat/          # 兼容性模块
│   ├── __init__.py
│   └── app_compat.py     # 兼容性入口
├── requirements.txt # 项目依赖
└── README.md        # 项目文档
```

## 4. 使用指南

### 4.1 启动问答界面

推荐使用Streamlit直接启动：

```bash
streamlit run streamlit_app.py
```

或通过统一入口：

```bash
python run.py app
```

启动后，在浏览器中访问 `http://localhost:8501` 即可使用问答界面。

### 4.2 构建向量数据库

首次使用前，需要构建向量数据库：

```bash
# 基本构建（会覆盖现有向量库）
python run.py build

# 增量追加到现有向量库
python run.py build --append

# 从断点继续构建（中断后重启）
python run.py build --resume

# 清除嵌入缓存并重建
python run.py build --clear-cache
```

### 4.3 向量库管理

系统提供多种向量库管理命令：

```bash
# 查看向量库统计信息
python run.py manage stats

# 列出所有已处理的源文件
python run.py manage list

# 删除特定源文件的向量（支持多文件）
python run.py manage delete file1.pdf file2.docx
```

### 4.4 配置系统参数

主要配置参数位于 `config/settings.py`，可根据需要调整：

| 参数类别 | 参数名 | 说明 | 默认值 |
|---------|-------|------|--------|
| 数据目录 | DATA_DIR | 文档存放目录 | ./dataset |
| 向量库 | FAISS_SAVE_PATH | FAISS索引保存路径 | ./faiss/multi_format_store |
| Qdrant | QDRANT_HOST | Qdrant服务地址 | localhost |
| Qdrant | QDRANT_PORT | Qdrant端口 | 6333 |
| Qdrant | QDRANT_COLLECTION | Qdrant集合名称 | multi_format_vectors |
| 模型 | EMBEDDING_MODEL | 嵌入模型名称 | mxbai-embed-large |
| 模型 | LLM_MODEL | 大语言模型名称 | llama2-chinese:7b |
| 文本分割 | CHUNK_SIZE | 文本块大小 | 300 |
| 文本分割 | CHUNK_OVERLAP | 文本块重叠大小 | 50 |
| 并发 | MAX_WORKERS | 并发处理进程数 | 4 |
| 缓存 | EMBEDDING_CACHE_DIR | 嵌入缓存目录 | ./cache/embeddings |
| 其他 | BATCH_SIZE | 批处理大小 | 100 |

## 5. 高级功能

### 5.1 嵌入缓存优化

系统自动缓存计算过的嵌入向量，显著提高重复处理速度。缓存统计信息会在处理完成后显示在日志中：

```
[INFO] 嵌入缓存统计: 命中=153, 未命中=47, 命中率=76.50%
```

### 5.2 断点续传

对于大规模文档处理，系统支持中断后继续处理：

1. 按 `Ctrl+C` 可安全中断处理过程
2. 中断状态会自动保存
3. 使用 `python run.py build --resume` 从断点继续

### 5.3 并发控制

通过调整 `MAX_WORKERS` 参数可控制并发处理文档的进程数：

- 增加进程数可提高处理速度，但会占用更多资源
- 对于内存受限系统，建议设置为较小值（如2-4）

### 5.4 日志系统

系统提供详细日志，可在 `logs` 目录下查看：

- `app.log`: 应用运行日志
- `vector_build.log`: 向量构建日志
- `vector_manager.log`: 向量管理日志

## 6. 常见问题排查

### 6.1 文档解析错误

**问题**: 某些文档无法正确解析
**解决方案**:
- 确保已安装完整的unstructured依赖: `pip install "unstructured[all-docs]"`
- 对于复杂PDF，尝试安装额外OCR支持: `pip install pytesseract pdf2image`
- 确认文档编码正确，特别是非UTF-8编码文本文件

### 6.2 向量库错误

**问题**: 向量库操作失败
**解决方案**:
- 确保Qdrant服务运行: `docker run -p 6333:6333 qdrant/qdrant`
- 检查向量维度一致性，更改模型后需重建向量库
- FAISS加载错误可用 `--allow-dangerous` 参数处理

### 6.3 模型调用错误

**问题**: 嵌入或LLM模型调用失败
**解决方案**:
- 确认Ollama服务正常运行: `ollama serve`
- 验证模型已下载: `ollama list`
- 如需更换模型，同时更新 `config/settings.py` 中的模型名称

### 6.4 内存不足

**问题**: 处理大量文档时内存不足
**解决方案**:
- 减少 `MAX_WORKERS` 值，降低并发处理数
- 调整 `CHUNK_SIZE` 减小文本块大小
- 使用 `--append` 模式分批处理文档

## 7. 开发与扩展

### 7.1 添加新的文档处理器

在 `tools/vector_builder.py` 中的 `process_file` 函数中添加新的文档类型支持：

```python
elif filename.endswith(".新格式"):
    # 实现新格式处理逻辑
    docs = YourNewFormatLoader(path).load()
```

### 7.2 修改提示词模板

在 `prompts/templates.py` 中修改 `PROMPT_TEMPLATE` 可自定义系统提示词。

### 7.3 更换嵌入或LLM模型

1. 在 `config/settings.py` 中更新模型名称
2. 如使用非Ollama模型，需修改 `rag/embedding.py` 和 `rag/chain.py`

### 7.4 自定义UI界面

编辑 `ui/streamlit_app.py` 可自定义用户界面，添加新功能或改变外观。

## 8. 许可与致谢

### 许可证

本项目采用 [MIT许可证](LICENSE)

### 使用的开源项目

- [LangChain](https://github.com/langchain-ai/langchain): RAG框架与组件
- [FAISS](https://github.com/facebookresearch/faiss): 高效向量检索
- [Qdrant](https://github.com/qdrant/qdrant): 向量数据库
- [Streamlit](https://github.com/streamlit/streamlit): 用户界面
- [Ollama](https://github.com/ollama/ollama): 本地模型服务

## 9. 联系与支持

如有问题或建议，请通过以下方式联系：

- 项目讨论区: [链接到项目讨论区]
- 邮件: [您的联系邮箱]
