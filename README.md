# squilla_router_standalone

<p align="center">
  <a href="https://github.com/CatchMoa/squilla_router_standalone"><img src="https://img.shields.io/github/stars/CatchMoa/squilla_router_standalone?style=for-the-badge&logo=github" alt="Stars"></a>
  <a href="https://github.com/CatchMoa/squilla_router_standalone/releases"><img src="https://img.shields.io/github/v/release/CatchMoa/squilla_router_standalone?include_prereleases&style=for-the-badge&logo=github" alt="Release"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge&logo=python" alt="Python 3.12+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge" alt="Apache 2.0"></a>
  <a href="https://github.com/CatchMoa/squilla_router_standalone/actions"><img src="https://img.shields.io/github/actions/workflow/status/CatchMoa/squilla_router_standalone/ci.yml?style=for-the-badge&label=CI" alt="CI"></a>
</p>

OpenSquilla 模型路由引擎的独立抽离包。

一个自包含的 Python 包,完整复刻了 OpenSquilla 的 `squilla_router` 决策路径 ——
V4 Phase 3 ML 分类器 + 启发式降级 + 完整 8 道门控后策略管道,附带轻量 HTTP 代理层,
可作为 Claude Code 的前置动态路由网关使用。

---

## 快速开始

### 安装

```bash
# ML 路由(完整 V4 Phase 3,推荐)
pip install -e ".[ml]"

# 仅启发式降级(无 ML 依赖)
pip install -e .
```

### 作为 Claude Code 代理使用

```bash
# 交互式配置向导
python -m squilla_router_standalone --setup

# 启动代理
python -m squilla_router_standalone
```

然后在 Claude Code 的 `settings.json` 中配置:
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
    "ANTHROPIC_API_KEY": "sk-your-api-key"
  }
}
```

### 作为库使用

```python
import asyncio
from squilla_router_standalone import Router, SquillaRouterConfig

router = Router(SquillaRouterConfig())

async def main():
    result = await router.route(
        "请分析这段代码的复杂度",
        session_key="s1",
    )
    print(f"tier={result.tier}, model={result.model}, source={result.source}")
    # 输出示例: tier=c2, model=claude-sonnet-4-6, source=v4_phase3

asyncio.run(main())
```

---

## 架构

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────┐
│  V4 Phase 3 ML 分类器(LightGBM + ONNX)          │
│  └─ 提取特征(bge-onnx 嵌入 + 统计特征)           │
│  └─ 分类 → [c0, c1, c2, c3] 概率分布            │
│  └─ 降级: heuristic 规则(基于长度/附件)           │
└───────────────┬─────────────────────────────────┘
                │ proposed_tier, confidence
                ▼
┌─────────────────────────────────────────────────┐
│  8 道门控策略管道                                │
│                                                  │
│  1. confidence_gate    — 低置信度降级到默认 tier  │
│  2. complaint_upgrade  — 投诉/催促内容升级 tier   │
│  3. anti_downgrade     — 保护 KV-cache 连续性     │
│  4. capability_gate    — 按 vision/context 升 tier│
│  5. bind               — 绑定到有效 tier          │
│  6. large_context_floor— 大上下文强制最低 tier    │
│  7. budget_gate        — 会话预算上限             │
│  8. provider_mismatch  — provider 不可用回退      │
└───────────────┬─────────────────────────────────┘
                │ final_tier
                ▼
┌─────────────────────────────────────────────────┐
│  控制器(ThinkingController + PromptController)    │
│  └─ thinking_level: off | low | medium | high     │
│  └─ prompt_hint: 注入 [RESPONSE_POLICY: ...]      │
└───────────────┬─────────────────────────────────┘
                │ tier, model, thinking_level
                ▼
         转发到目标 API
```

---

## 配置参考

### 代理配置(`proxy` 节)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `host` | `127.0.0.1` | 监听地址 |
| `port` | `8080` | 监听端口 |
| `target_base_url` | `https://api.anthropic.com` | 目标 API 端点 |
| `active_provider` | `anthropic` | 提供商标识 |
| `auto_detect_settings` | `true` | 自动从 settings.json 检测 |
| `tool_result_projection_enabled` | `true` | 启用工具结果投影 |
| `tool_result_projection_max_chars` | `60000` | 投影阈值(字符) |
| `prompt_cache_enabled` | `true` | 启用 prompt cache 注入 |

### 路由配置(`squilla_router` 节)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `strategy` | `v4_phase3` | 路由策略: `auto` / `v4_phase3` / `heuristic` |
| `default_tier` | `c1` | 默认 tier |
| `confidence_threshold` | `0.5` | 置信度阈值(0.0-1.0) |
| `complaint_upgrade_enabled` | `true` | 启用投诉升级 |
| `kv_cache_anti_downgrade_window_seconds` | `600` | 反降级窗口(秒) |
| `calibration_enabled` | `true` | 启用校准 |
| `budget.max_session_cost_usd` | `null` | 会话预算上限(美元) |

Tier 配置示例:

```toml
[squilla_router.tiers.c0]
model = "claude-haiku-4-5-20251001"
context_window = 200000
supports_vision = false

[squilla_router.tiers.c3]
model = "claude-opus-4-8"
context_window = 200000
supports_vision = true
```

---

## 成本节省功能

### Tool Result Projection

在转发请求前,自动检测过大的工具结果(>60K 字符),将其替换为 `[tool_result_projection]`
摘要块(保留 head~65% + tail~35%),大幅减少输入 token 消耗。

```toml
[proxy]
tool_result_projection_enabled = true
tool_result_projection_max_chars = 60000
# 可选:存储完整结果到磁盘(供调试)
# tool_result_projection_store_dir = "/tmp/projected_results"
```

### Prompt Cache 注入

自动在 system prompt 的最后一个 text block 上注入 `cache_control: {type: "ephemeral"}`,
利用 Anthropic / OpenRouter 的 prompt caching 功能节省 50-90% 输入 token。

```toml
[proxy]
prompt_cache_enabled = true
```

### Context Compaction

当会话上下文超过预算时,自动压缩旧历史条目为摘要。目前为独立模块,需手动集成:

```python
from squilla_router_standalone.compaction import CompactionConfig, CompactionRequest, compact_context

result = await compact_context(CompactionRequest(
    session_id="s1",
    entries=history_entries,
    context_window_tokens=200_000,
    config=CompactionConfig(model="qwen3.6-flash", api_key="sk-..."),
))
# result.summary: 压缩后的摘要文本
# result.kept_entries: 保留的最近条目
```

---

## 项目结构

```
squilla_router_standalone/
├── __init__.py                 # 公共 API: Router, SquillaRouterConfig
├── pyproject.toml              # 包配置 + 依赖声明
├── README.md                   # 本文件
├── .gitignore
│
├── api.py                      # 高层入口: Router.route() → RoutingResult
├── config.py                   # SquillaRouterConfig(pydantic BaseSettings)
├── catalog.py                  # 配置驱动的 TierCapability 查询
├── pricing.py                  # 静态价格表
├── history.py                  # 路由历史存储(5 条 + 1800s 窗口)
├── decision_record.py          # 文件版 JSONL 决策记录持久化
├── router_tiers.py             # Tier 定义(逐字移植)
├── router_control.py           # 路由控制(逐字移植)
├── router_runtime_diagnostics.py  # 运行时诊断(逐字移植)
├── prompt_cache.py             # Prompt cache 指标(纯函数)
├── tool_result_projection.py   # 工具结果投影(纯函数)
├── compaction.py               # 上下文压缩(LLM + fallback)
├── compaction_state.py         # 结构化压缩状态
│
├── proxy.py                    # 薄 HTTP 代理层(Starlette)
├── __main__.py                 # python -m 入口
│
├── engine/
│   ├── __init__.py
│   ├── pipeline.py             # TurnContext dataclass
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── policy.py           # 8 门控策略引擎(逐字移植)
│   │   ├── policy_data.py      # 策略数据(投诉词等)
│   │   ├── heuristic.py        # 启发式规则
│   │   └── calibration.py      # 校准(逐字移植)
│   └── steps/
│       └── squilla_router.py   # 核心编排(1487 行)
│
├── squilla_router/
│   ├── __init__.py
│   ├── controller.py           # Thinking/Prompt 控制器(逐字移植)
│   └── v4_phase3.py            # V4 分类器(逐字移植)
│
├── models/
│   └── v4.2_phase3_inference/  # ML bundle(76M, 复制)
│
└── tests/
    └── test_router.py          # 20 个测试覆盖全部门控
```

---

## 范围说明

### 已移植 ✅
- V4 Phase 3 ML 分类器(LightGBM + ONNX 推理,76M bundle 自包含)
- 8 道门控策略管道(confidence/complaint/anti-downgrade/capability/bind/large-context/budget/provider-mismatch)
- 2 级控制器(ThinkingController + PromptController)
- 基于文件的路由决策记录持久化
- 路由历史存储(5 条上限 + 1800s 过期窗口)
- 校准(calibration) + 聚合
- 启发式降级(基于长度/附件)
- 交互式配置向导(SetupWizard)
- 模型自动发现(Anthropic/OpenAI/DashScope)
- Tool Result Projection(纯函数 + proxy 集成)
- Prompt Cache 注入(纯函数 + proxy 集成)
- Context Compaction(独立模块)

### 不在此包 ❌
- **Self-learning 离线训练器**:LightGBM 增量训练/promotion/rollback 不移植。
  V4 适配器保留了 `_train_features` 捕获 seam 和 `set_cache_invalidator` no-op 钩子,
  数据可流出,未来可接。
- **多 Provider 凭证切换**:proxy 层单一目标 API,不做凭证解析。
- **Gateway Web UI**:完整的 Vue 管理界面不移植。
- **Selector 多模型选择**:深度耦合 provider 系统,不移植。

---

## 开发

```bash
pip install -e ".[dev]"

# 运行测试
pytest -v

# 类型检查
mypy squilla_router_standalone/

# 代码检查
ruff check squilla_router_standalone/
```

---

## 许可

Apache-2.0 License。V4 Phase 3 模型 bundle 来自 [OpenSquilla](https://github.com/opensquilla/opensquilla)。