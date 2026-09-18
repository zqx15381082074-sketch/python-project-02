# LangChain 学习项目：RAG 问答应用与医学健康 Agent 智能客服

基于 LangChain + 通义千问（Qwen）构建的两个智能问答应用，以及一套 LangChain 基础学习笔记。

- 应用一：简易 RAG 问答机器人（根目录）—— LCEL 链式组装的检索增强问答
- 应用二：智康助手 · 医学健康 Agent 智能客服（`agent项目/`）—— 基于新版 `create_agent` 的自主 Agent，带工具调用与动态提示词切换
- 学习笔记（`langchainRAG基础知识/`）—— 21 个编号练习，覆盖模型调用 → 记忆 → 向量库 → 工具调用全流程

## 项目结构

```
02/
├── app_qa.py                  # 应用一：RAG 问答对话界面（Streamlit）
├── file_uploader.py           # 应用一：知识库文件上传界面
├── rag_service.py             # 应用一：RAG 核心链（LCEL）
├── knowledge_base_service.py  # 应用一：知识入库（MD5 去重 + 切片）
├── vector_store_service.py    # 应用一：Chroma 向量库访问层
├── history_store.py           # 应用一：文件版会话记忆
├── config_data.py             # 应用一：配置
│
├── agent项目/                  # 应用二：医学健康 Agent 智能客服
│   ├── app.py                 # 对话界面（Streamlit）
│   ├── agent/
│   │   ├── react_agent.py     # Agent 本体（create_agent + ReAct）
│   │   └── tools/
│   │       ├── agent_tools.py # 7 个工具（RAG 检索 / 天气 / 健康记录等）
│   │       └── middleware.py  # 中间件：日志监控 + 动态提示词切换
│   ├── model/factory.py       # 模型工厂（ChatTongyi + DashScopeEmbeddings）
│   ├── rag/                   # 检索服务（工程化 RAG，直接运行 vector_store.py 建库）
│   ├── prompts/               # 提示词文件（主客服 / RAG 总结 / 报告生成）
│   ├── config/                # yml 配置（模型、切片、路径、向量库）
│   ├── data/                  # 医学知识库文档 + 外部健康监测数据
│   └── utils/                 # 路径 / 配置 / 文件 / 提示词 / 日志工具
│
└── langchainRAG基础知识/       # LangChain 基础学习脚本（21 个）
```

## 功能特性

### 应用一：简易 RAG 问答机器人
- 上传 `.txt` 文档构建个人知识库（MD5 内容去重，防止重复入库）
- 向量检索（Chroma）+ 通义 qwen3-max 流式回答
- 基于文件的会话记忆，支持多轮对话

### 应用二：智康助手 · 医学健康 Agent
- **自主 Agent**：模型自行决定调用哪些工具、调用几次，而非固定执行链
- **7 个工具**：医学知识库检索、天气查询、用户身份 / 日期模拟、健康监测记录（CSV）读取等
- **动态提示词切换（亮点）**：通过 middleware 监控工具调用，当触发"生成报告"工具时，运行时自动把客服人设切换为"健康报告写手"
- **全链路日志** + 配置化（yml）+ 抽象工厂模式
- 医学主题知识库：常见病科普、慢性病管理、用药常识、体检指标解读、健康生活方式
- 回答附带就医提醒与 AI 免责声明

### 典型交互示例
| 提问 | Agent 行为 |
|---|---|
| "感冒发烧了该怎么办？" | 调用 RAG 工具检索知识库 → 组织专业回答 |
| "合肥今天天气怎么样，适合锻炼吗？" | 调用天气工具 → 结合建议回答 |
| "给我生成我的健康报告" | get_user_id → get_current_month → 切换报告提示词 → 读取健康记录 → 生成 Markdown 报告 |

## 快速开始

### 1. 环境要求
- Python 3.10+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

两个应用都使用阿里云百炼（DashScope）的模型，需要先在 [百炼控制台](https://bailian.console.aliyun.com/) 开通并创建 API Key，然后设置环境变量：

```bash
# Windows (PowerShell)
$env:DASHSCOPE_API_KEY = "sk-xxxx"

# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxx
```

### 4. 构建知识库并运行

**应用一（简易 RAG 问答）：**

```bash
streamlit run app_qa.py        # 对话界面
streamlit run file_uploader.py # 上传 txt 构建知识库
```

**应用二（医学健康 Agent）：**

```bash
cd agent项目

# 1) 先构建医学知识向量库（首次运行必须，扫描 data/ 目录并入库）
python -m rag.vector_store

# 2) 启动对话界面
streamlit run app.py
```

## 技术栈

| 组件 | 说明 |
|---|---|
| LangChain 1.x + LangGraph | Agent 框架（`create_agent`、middleware、ReAct） |
| LCEL | 应用一的链式编排（Runnable 组合） |
| Chroma | 向量数据库 |
| ChatTongyi (qwen3-max) | 对话模型 |
| DashScopeEmbeddings (text-embedding-v4) | 向量化模型 |
| Streamlit | Web 界面 |

## 两版架构对比

| 维度 | 应用一（根目录） | 应用二（agent项目） |
|---|---|---|
| 架构 | 固定执行链（LCEL 拼装） | 自主 Agent（模型决定工具调用） |
| 提示词 | 写死在代码里 | txt 文件配置化 + 运行时动态切换 |
| 配置 | 散落各处 | yml 统一管理 + 抽象工厂 |
| 去重 | 对文本内容算 MD5 | 对整个文件算 MD5 |
| 日志 | 无 | 全链路 logger |

## 免责声明

应用二的回答由 AI 生成，仅供健康科普参考，不能替代专业医疗建议。如有身体不适，请及时就医。
