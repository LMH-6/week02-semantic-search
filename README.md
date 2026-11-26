# Week 02 - Local Semantic Search

一个基于 Python、SentenceTransformers 和 Streamlit 实现的轻量级本地 FAQ 检索系统。

项目实现了语义检索（Semantic Search）与关键词检索（Keyword Search）两种检索方式，并通过 Top-K 排序、相似度计算和无答案查询过滤，实现对本地 FAQ 知识库的检索与展示。

## 1. 项目功能

- FAQ 数据集管理
- 基于 SentenceTransformers 的文本向量编码
- 本地向量索引构建与加载
- 基于余弦相似度的语义检索
- 基于关键词交集的关键词检索
- Top-K 检索结果排序
- 无答案查询的初步识别
- 两种检索算法的对比实验
- Streamlit Web 交互界面

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python | 项目开发语言 |
| PyTorch | 张量计算及向量存储 |
| SentenceTransformers | 文本向量编码 |
| all-MiniLM-L6-v2 | 预训练文本嵌入模型 |
| Streamlit | Web 交互界面 |

## 3. 项目结构

```text
week02-semantic-search/
│
├── app.py                    # Streamlit 应用入口
├── build_index.py            # 构建 FAQ 向量索引
├── faq_data.py               # FAQ 数据集
├── search_faq.py             # 独立语义检索
├── search_keyword.py         # 关键词检索
├── compare_search.py         # 检索算法对比与评估
├── test_queries.py           # 测试查询数据
│
├── semantic_search_official.py  # 官方示例复现
├── official_run.txt          # 官方示例运行记录
│
├── faq_index.pt              # 本地 FAQ 向量索引
│
└── README.md
```

## 4. 环境配置

建议使用 Python 虚拟环境。

### 4.1 创建虚拟环境

```bash
python -m venv .venv
```

### 4.2 激活虚拟环境

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4.3 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

项目使用 CPU 进行向量编码和检索。

## 5. 运行项目

### 5.1 构建向量索引

```bash
python build_index.py
```

该脚本主要完成：

1. 加载 FAQ 数据。
2. 使用 SentenceTransformer 对 FAQ 问题进行向量编码。
3. 保存 FAQ 数据及对应向量。
4. 生成本地索引文件 `faq_index.pt`。

### 5.2 启动 Web 应用

```bash
streamlit run app.py
```

启动后，在浏览器中访问 Streamlit 提供的本地地址。

## 6. 核心原理

### 6.1 Semantic Search

语义检索通过预训练模型将文本转换为向量，再使用余弦相似度计算用户查询与 FAQ 问题之间的语义接近程度。

主要流程：

1. 用户输入查询。
2. 将查询编码为向量。
3. 计算查询向量与知识库向量之间的余弦相似度。
4. 使用 Top-K 获取相似度最高的结果。
5. 根据相似度阈值进行初步无答案判断。

### 6.2 Keyword Search

关键词检索采用简单的文本处理方式：

1. 将文本转换为小写。
2. 使用正则表达式提取关键词。
3. 通过停用词集合过滤常见词。
4. 计算查询与 FAQ 之间的共同关键词数量。
5. 按匹配数量从高到低排序。

### 6.3 检索算法对比

| 对比维度 | Semantic Search | Keyword Search |
|---|---|---|
| 匹配依据 | 语义向量 | 共同关键词 |
| 同义表达 | 能够识别一定程度的语义关联 | 较弱 |
| 精确词汇匹配 | 支持 | 支持 |
| 主要计算方式 | 余弦相似度 | 关键词交集 |

## 7. 实验与评估

使用自定义测试查询，对两种检索方式进行初步评估。

评估指标：

- Hit@3：正确答案是否出现在 Top-3 检索结果中。
- Top-K：返回相似度最高的 K 条结果。
- Semantic Similarity：查询与 FAQ 之间的余弦相似度。
- Keyword Score：查询与 FAQ 的共同关键词数量。

目前测试集规模较小，实验结果仅用于验证基础检索流程，不代表系统在真实场景中的泛化性能。

## 8. 项目运行效果

系统提供本地 Web 交互界面，用户可以：

- 输入自然语言查询。
- 选择语义检索或关键词检索。
- 自定义 Top-K 数量。
- 查看 FAQ 问题、答案和匹配分数。
- 对低相似度查询进行初步无答案提示。

## 9. 后续改进方向

- 扩充 FAQ 数据集。
- 优化关键词提取与文本预处理。
- 改进无答案识别及相似度阈值。
- 尝试更高质量的文本嵌入模型。
- 引入向量数据库。
- 探索检索增强生成（RAG）流程。