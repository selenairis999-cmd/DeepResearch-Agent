# DeepResearch Agent · 多智能体深度研究报告生成系统

输入一个研究话题，五个 Agent 协作完成「拆题 → 并行检索 → 质量评审（可打回重检索）→ 人工确认 → 成稿 → 审校」，输出一份带可溯源引用的结构化研究报告。全程检查点持久化，支持断点续跑与历史分叉；顶到打回上限时停下来问人（HITL），Web 界面实时展示各 Agent 的执行状态与反馈回路。

## 架构与流程

```
用户话题
   │
   ▼
Planner ──拆成 3~5 个子问题（JSON 结构化输出）
   │
   ▼
Researcher × N ──Send API map-reduce 并行：Tavily 检索 → Jina 抓正文 → 摘要成证据卡
   │
   ▼
Critic ──按话题类型自适应评估（事实调研 vs 前瞻观点），打分 + 找缺口
   │
   ├─ 达标 → Writer → Editor → 报告 + trace 落盘
   ├─ 打回 → 带缺口描述回到 Planner 重拆题（最多 MAX_LOOPS 轮，防死循环）
   └─ 轮数用尽仍不达标 → human_gate（interrupt 挂起，等人工选择）
         ├─ force_write ── 用历史最高分轮证据直接出稿（回滚）
         └─ retry_once ── 再打回一轮；下轮仍不达标则硬性上限自动放行
```

## 核心设计

- **反馈回路**：Critic 评分低于及格线（默认 6 分）时，携带具体缺口描述触发重新规划与检索
- **人在回路（HITL）**：打回轮数用尽时 `interrupt()` 挂起整张图等人工裁决，而不是静默强制出稿或死循环；人工选择经 `Command(resume)` 注入继续。人工确认节点独立于 Critic 节点——`interrupt` 恢复后节点会从头重跑，打分逻辑混在里面会重复烧一次 LLM（测试实证 resume 后 Critic 调用数不变）
- **检查点持久化**：`SqliteSaver` 逐超步落盘（`checkpoints.sqlite`）；进程崩溃 / Ctrl+C 后 `--resume` 接着跑，`--history` 列检查点，`--fork` 从任意历史检查点分叉改写走向（回到过去）
- **最佳检查点回滚**：每轮刷新历史最高分并快照证据；强制放行时自动回滚到最高分轮次成稿，避免多轮迭代质量降级
- **反幻觉**：参考文献由代码从证据列表拼接（非 LLM 生成，URL 去重）；证据为空时 Writer 硬校验拒绝成稿
- **模型分级调度**：按调用模式分档——评审/拆题用推理型模型、检索摘要用高并发小模型、长文成稿用快速模型；支持 Ollama 本地与阿里云百炼 API 双后端按角色混合路由
- **可观测性**：每个节点的输入输出（含 Critic 原始响应、截断统计、回滚标记、人工裁决）落盘为 trace JSON；19 个 pytest 用例覆盖去重 / 检索重试 / 人工确认 / 持久化续跑

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 TAVILY_API_KEY；LLM 默认走本地 Ollama
python main.py         # CLI 模式
python web.py          # Web 演示界面 http://127.0.0.1:8000
pytest                 # 19 个测试（无网络依赖，全桩）
```

CLI 断点玩法：

```bash
python main.py --list                       # 历史任务
python main.py --resume research-xxxx-...   # 崩溃/中断后接着跑（含挂起中的人工确认）
python main.py --history research-xxxx-...  # 检查点序列（新→旧，含 checkpoint_id）
python main.py --fork research-xxxx-... <checkpoint_id>  # 回到过去某一步分叉续跑
```

Web 界面同样支持：任务挂在人工确认时弹确认框，选「直接出稿 / 再试一轮」后从断点续跑（`GET /api/research/resume`）。

本地模式需先安装 [Ollama](https://ollama.com) 并拉取模型（默认 `qwen3:8b` + `qwen3.5:4b`）。
`.env` 中 `LLM_BACKEND=dashscope` 可切换为阿里云百炼 API（支持 DeepSeek / Qwen 等模型）。

## 配置项（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_BACKEND` / `FAST_BACKEND` | 强/快模型后端：`ollama` 或 `dashscope` | ollama / 跟随强模型 |
| `MODEL_STRONG` / `MODEL_FAST` | 评审、拆题 / 检索摘要模型 | qwen3:8b / qwen3.5:4b |
| `MODEL_WRITER` / `MODEL_EDITOR` | 成稿、审校模型（推理型模型太慢可单独换） | 跟随 STRONG |
| `MAX_LOOPS` | Critic 打回重检索的最大轮数 | 3 |
| `CRITIC_PASS_SCORE` | Critic 及格线（0-10） | 6 |
| `HUMAN_IN_LOOP` | 打回上限后转人工确认（false=直接自动放行） | true |
| `CHECKPOINT_DB` | 检查点库路径 | checkpoints.sqlite |

## 项目结构

```
├── graph.py          # LangGraph 编排：Send 并行 fan-out + 反馈回路 + human_gate
├── state.py          # 共享 State（reducer 聚合并行证据、最佳快照、URL 去重）
├── config.py         # 双后端配置与模型分级路由 + HITL/检查点开关
├── persistence.py    # SqliteSaver 工厂 + 历史任务索引（runs.json）
├── agents/           # planner / researcher / critic / human_gate / writer / editor
├── tools/            # llm（重试+后端路由）/ search（Tavily，带重试退避）/ scrape（Jina）
├── web.py            # FastAPI + SSE 实时推送 + 人工确认断点续跑
├── static/           # 单页演示前端（KaTeX 公式渲染 + 人工确认弹窗）
├── tests/            # 19 个 pytest：去重 / 检索重试 / 人工确认 / 持久化
├── traces/           # 运行产物：报告 MD + 全链路 trace JSON
└── examples/         # 示例：一次真实运行的报告与 trace（GPT 自回归机制，Critic 一轮 7/10 放行）
```
