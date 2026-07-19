#!/usr/bin/env python3
"""OpenSquilla Router Standalone — 动态路由网关

将 OpenSquilla 的路由引擎（分类器 + 8 道门控策略 + 控制器）作为前置代理，
根据用户输入的复杂度动态选择模型，在保证质量的同时节省 Token 开销。

架构:
  Claude Code → 本代理 (http://localhost:8080) → 目标 API (Anthropic/OpenRouter/...)
                     │
                 路由引擎 (V4 ML 分类器 + 8 道门控)
                     │
                动态选择: c0 便宜模型 | c1 平衡模型 | c2 中等 | c3 最强模型

用法:
  python -m squilla_router_standalone --setup     # 交互式配置向导
  python -m squilla_router_standalone              # 启动代理

配置 Claude Code (.claude/settings.json):
  ANTHROPIC_BASE_URL = http://127.0.0.1:8080
  ANTHROPIC_API_KEY = <你的 API key>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import tomli_w
import tomllib
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from squilla_router_standalone.api import Router
from squilla_router_standalone.config import SquillaRouterConfig
from squilla_router_standalone.prompt_cache import cache_system_prompt
from squilla_router_standalone.router_control import RouterControlHoldStore
from squilla_router_standalone.router_tiers import TEXT_TIERS
from squilla_router_standalone.tool_result_projection import (
    ProjectionConfig,
    ProjectionResult,
    project_tool_result,
)

logger = logging.getLogger("claude-code-proxy")

# Proxy-specific: thinking_level -> Anthropic thinking.budget_tokens
_THINKING_BUDGET_TOKENS: dict[str, int] = {
    "low": 1024,
    "medium": 10000,
    "high": 20000,
    "xhigh": 32000,
}

# Text mode override: 用户消息中嵌入 @model:xxx 可跳过路由，直接使用指定模型
TEXT_MODE_PATTERN = r"@model:(\S+)"

# 默认模型映射
DEFAULT_TIER_CONFIGS: dict[str, dict[str, str]] = {
    "c0": {"model": "claude-haiku-4-5-20251001", "description": "简单任务：快/便宜"},
    "c1": {"model": "claude-sonnet-4-6", "description": "默认模型：平衡"},
    "c2": {"model": "claude-sonnet-4-6", "description": "中等复杂度：标准"},
    "c3": {"model": "claude-opus-4-8", "description": "最复杂任务：最强推理"},
}

# 常用 API 端点
KNOWN_API_ENDPOINTS = {
    "Anthropic": "https://api.anthropic.com",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Google Gemini": "https://generativelanguage.googleapis.com/v1beta",
    "Groq": "https://api.groq.com/openai/v1",
    "Together AI": "https://api.together.xyz/v1",
    "Fireworks AI": "https://api.fireworks.ai/inference/v1",
    "Mistral AI": "https://api.mistral.ai/v1",
    "Perplexity": "https://api.perplexity.ai",
}


# =========================================================================
# 配置系统
# =========================================================================


@dataclass
class ProxyConfig:
    """代理配置，支持 TOML 持久化。"""

    target_base_url: str = "https://api.anthropic.com"
    active_provider: str = "anthropic"
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    auto_detect_settings: bool = True
    router: SquillaRouterConfig = field(default_factory=SquillaRouterConfig)

    # 成本节省功能开关
    tool_result_projection_enabled: bool = True
    tool_result_projection_max_chars: int = 60_000
    tool_result_projection_store_dir: str | None = None
    prompt_cache_enabled: bool = True

    @classmethod
    def config_paths(cls) -> list[Path]:
        return [
            Path.cwd() / ".claude" / "claude_code_proxy.toml",
            Path.home() / ".claude" / "claude_code_proxy.toml",
        ]

    @classmethod
    def find_config(cls) -> Path | None:
        for path in cls.config_paths():
            if path.exists():
                return path
        return None

    @classmethod
    def default_config_path(cls) -> Path:
        return Path.home() / ".claude" / "claude_code_proxy.toml"

    def save(self, path: Path | None = None) -> Path:
        path = path or self.default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "proxy": {
                "host": self.host,
                "port": self.port,
                "target_base_url": self.target_base_url,
                "active_provider": self.active_provider,
                "auto_detect_settings": self.auto_detect_settings,
                "tool_result_projection_enabled": self.tool_result_projection_enabled,
                "tool_result_projection_max_chars": self.tool_result_projection_max_chars,
                "prompt_cache_enabled": self.prompt_cache_enabled,
            },
            "squilla_router": self.router.model_dump(mode="json", exclude_none=True),
        }
        if self.tool_result_projection_store_dir:
            data["proxy"]["tool_result_projection_store_dir"] = self.tool_result_projection_store_dir
        path.write_text(tomli_w.dumps(data), encoding="utf-8")
        logger.info("配置已保存到 %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> ProxyConfig:
        if path is None:
            found = cls.find_config()
            if found is None:
                logger.info("未找到配置文件，使用默认配置")
                return cls()
            path = found
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        cfg = cls()
        proxy_cfg = raw.get("proxy", {})
        cfg.target_base_url = str(proxy_cfg.get("target_base_url", cfg.target_base_url))
        cfg.active_provider = str(proxy_cfg.get("active_provider", cfg.active_provider))
        cfg.host = str(proxy_cfg.get("host", cfg.host))
        cfg.port = int(proxy_cfg.get("port", cfg.port))
        cfg.auto_detect_settings = bool(proxy_cfg.get("auto_detect_settings", cfg.auto_detect_settings))
        cfg.tool_result_projection_enabled = bool(proxy_cfg.get("tool_result_projection_enabled", cfg.tool_result_projection_enabled))
        cfg.tool_result_projection_max_chars = int(proxy_cfg.get("tool_result_projection_max_chars", cfg.tool_result_projection_max_chars))
        cfg.tool_result_projection_store_dir = proxy_cfg.get("tool_result_projection_store_dir") or cfg.tool_result_projection_store_dir
        cfg.prompt_cache_enabled = bool(proxy_cfg.get("prompt_cache_enabled", cfg.prompt_cache_enabled))
        router_section = raw.get("squilla_router", {})
        if isinstance(router_section, dict) and router_section:
            cfg.router = SquillaRouterConfig(**router_section)
        logger.info("配置已从 %s 加载", path)
        return cfg

    @classmethod
    def auto_detect_from_settings(cls) -> dict[str, Any]:
        """从 Claude Code 的 settings.json 中自动检测 base_url 和 API key。"""
        result: dict[str, Any] = {"base_url": None, "api_key": None}
        for settings_path in (
            Path.cwd() / ".claude" / "settings.json",
            Path.home() / ".claude" / "settings.json",
        ):
            if not settings_path.exists():
                continue
            try:
                raw = json.loads(settings_path.read_text(encoding="utf-8"))
                env = raw.get("env", {}) if isinstance(raw, dict) else {}
                base_url = str(env.get("ANTHROPIC_BASE_URL", "") or "").strip()
                api_key = str(env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
                if base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
                    continue
                if base_url:
                    result["base_url"] = base_url
                if api_key:
                    result["api_key"] = api_key
                logger.info("从 %s 检测到: base_url=%s", settings_path, result["base_url"])
                break
            except (json.JSONDecodeError, OSError):
                continue
        return result

    @classmethod
    def build_with_auto_detect(cls, config_path: Path | None = None) -> ProxyConfig:
        if config_path and config_path.exists():
            cfg = cls.load(config_path)
        elif cls.find_config():
            cfg = cls.load()
        else:
            cfg = cls()
        if cfg.auto_detect_settings:
            detected = cls.auto_detect_from_settings()
            if detected["base_url"]:
                cfg.target_base_url = detected["base_url"]
            if detected["api_key"]:
                cfg.api_key = detected["api_key"]
        return cfg


async def discover_models(base_url: str, api_key: str = "",
                          timeout: float = 10.0) -> list[str]:
    """从 API 端点发现可用模型列表。

    支持:
      - Anthropic API: GET /v1/models
      - OpenAI 兼容 API: GET /v1/models
      - OpenRouter: GET /v1/models
      - 阿里云百炼 DashScope: GET /api/v1/models（原生端点）
    """
    url_lower = base_url.lower()

    # ── 阿里云百炼 DashScope：使用原生 API 获取模型列表 ──
    if "dashscope" in url_lower or "aliyuncs" in url_lower or "maas" in url_lower:
        native_base = "https://dashscope.aliyuncs.com"
        models_url = f"{native_base}/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(models_url, headers=headers, params={"page_size": 100})
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    output = data.get("output", {})
                    model_list = output.get("models", []) if isinstance(output, dict) else []
                    for m in model_list:
                        mid = m.get("model") or m.get("name") or m.get("id") or ""
                        if mid:
                            models.append(str(mid))
                    if models:
                        return sorted(set(models))
        except Exception as e:
            logger.debug("DashScope 模型列表获取失败: %s", e)

    # ── 标准 Anthropic / OpenAI 兼容：GET /v1/models ──
    models_url = f"{base_url.rstrip('/')}/v1/models"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                models_raw = data.get("data", [])
                if isinstance(models_raw, list):
                    model_ids = []
                    for m in models_raw:
                        mid = m.get("id") if isinstance(m, dict) else str(m)
                        if mid:
                            model_ids.append(str(mid))
                    return sorted(model_ids)
                return []
            elif resp.status_code == 401:
                logger.warning("模型发现失败: API key 无效 (HTTP 401)")
                return []
            else:
                logger.warning("模型发现失败: HTTP %s", resp.status_code)
                return []
    except httpx.ConnectError:
        logger.warning("模型发现失败: 无法连接到 %s", models_url)
        return []
    except httpx.TimeoutException:
        logger.warning("模型发现失败: 连接超时 %s", models_url)
        return []
    except Exception as e:
        logger.warning("模型发现失败: %s", e)
        return []


# =========================================================================
# 交互式配置向导
# =========================================================================


class SetupWizard:
    """交互式配置向导，提供类似 OpenSquilla 的 onboarding 体验。"""

    @staticmethod
    def _print_banner():
        print()
        print("  ╔══════════════════════════════════════════════╗")
        print("  ║   OpenSquilla Router Standalone 配置向导    ║")
        print("  ║   动态路由网关 · 智能模型选择                ║")
        print("  ╚══════════════════════════════════════════════╝")
        print()

    @staticmethod
    def _print_step(step: int, total: int, title: str):
        print(f"\n  [{step}/{total}] {title}")
        print(f"  {'─' * 40}")

    @staticmethod
    def _suggest_models(base_url: str) -> list[str]:
        url_lower = base_url.lower()

        if "dashscope" in url_lower or "aliyuncs" in url_lower or "maas" in url_lower:
            return [
                "qwen3.7-plus", "qwen3.6-plus", "qwen3-coder-plus",
                "qwen3.5-plus", "qwen-max", "qwen-plus",
                "qwen-turbo-latest", "glm-5.2", "glm-5.1",
                "deepseek-v4-flash", "deepseek-v4-pro",
            ]

        if "anthropic" in url_lower:
            return [
                "claude-sonnet-4-6", "claude-opus-4-8",
                "claude-haiku-4-5-20251001",
            ]

        if "openrouter" in url_lower:
            return [
                "anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-8",
                "anthropic/claude-haiku-4-5-20251001",
            ]

        if "openai" in url_lower:
            return ["gpt-5.5", "gpt-5.4-mini", "gpt-5.3", "o5-mini", "o5-pro"]

        if "deepseek" in url_lower:
            return ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"]

        return [
            "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
            "claude-opus-4-8", "gpt-5.5", "gpt-5.4-mini",
        ]

    def run(self) -> ProxyConfig:
        import questionary
        from questionary import Choice

        self._print_banner()
        total_steps = 5

        cfg = ProxyConfig()

        # ── 步骤 1: 检测 settings.json ──
        self._print_step(1, total_steps, "检测 Claude Code 配置")
        detected = ProxyConfig.auto_detect_from_settings()
        if detected["base_url"]:
            print(f"  ✅ 检测到 settings.json:")
            print(f"     API 端点: {detected['base_url']}")
            if detected.get("api_key"):
                print(f"     API Key:   {'*' * 8 + detected['api_key'][-4:]}")
            cfg.target_base_url = detected["base_url"]
            if detected.get("api_key"):
                cfg.api_key = detected["api_key"]
        else:
            print("  ℹ️  未检测到 settings.json，使用默认配置")

        # ── 步骤 2: 配置 API 端点 ──
        self._print_step(2, total_steps, "配置 API 端点")

        endpoint_choices = [Choice(title=f"{name} ({url})", value=url)
                           for name, url in KNOWN_API_ENDPOINTS.items()]

        detected_url = cfg.target_base_url
        default_choice = endpoint_choices[0]
        matched = False
        for choice in endpoint_choices:
            if choice.value == detected_url:
                default_choice = choice
                matched = True
                break

        if not matched:
            endpoint_choices.insert(0, Choice(
                title=f"✨ 检测到的端点 ({detected_url})", value=detected_url))
            default_choice = endpoint_choices[0]

        endpoint_choices.append(Choice(title="手动输入其他端点", value="__custom__"))

        selected = questionary.select(
            "选择 API 提供商:",
            choices=endpoint_choices,
            default=default_choice,
            use_search_filter=True,
            use_jk_keys=False,
        ).ask()

        if selected is None:
            print("  ⚠️  未选择，使用检测到的端点")
        elif selected == "__custom__":
            cfg.target_base_url = questionary.text(
                "输入 API 端点 URL:",
                default=detected_url,
                validate=lambda v: v.startswith("http") or "必须以 http:// 或 https:// 开头",
            ).ask()
        else:
            cfg.target_base_url = selected

        # ── 步骤 3: 发现模型 ──
        self._print_step(3, total_steps, "发现可用模型")

        cfg.api_key = questionary.password(
            "输入 API Key（可选，留空则从请求头读取）:",
            default=cfg.api_key,
        ).ask() or ""

        print("  🔍 正在从 API 拉取模型列表...")
        discovered = asyncio.run(discover_models(cfg.target_base_url, cfg.api_key))
        if discovered:
            print(f"  ✅ 发现 {len(discovered)} 个模型:")
            for m in discovered[:10]:
                print(f"     - {m}")
            if len(discovered) > 10:
                print(f"     ... 以及另外 {len(discovered) - 10} 个")
        else:
            print("  ℹ️  无法自动发现模型（API 不支持 /v1/models 端点）")
            print("     将使用常见模型列表供选择")

        # ── 步骤 4: 配置 Tier 模型映射 ──
        self._print_step(4, total_steps, "配置模型映射")

        tier_descriptions = {
            "c0": "简单任务（快速问答、文件读取）",
            "c1": "默认任务（常规问答、代码生成）",
            "c2": "中等复杂度（多步推理、代码 Debug）",
            "c3": "最复杂任务（架构设计、深度分析）",
        }

        if discovered:
            model_choices = [Choice(title=m, value=m) for m in discovered]
        else:
            model_choices = [Choice(title=m, value=m) for m in self._suggest_models(cfg.target_base_url)]

        for tier in TEXT_TIERS:
            print(f"\n     配置 {tier} — {tier_descriptions.get(tier, '')}")
            current = cfg.router.tiers.get(tier, {}).get("model", "")

            chosen = questionary.select(
                f"  为 {tier} 选择模型:",
                choices=model_choices + [Choice(title=f"自定义 (当前: {current})", value="__custom__")],
                default=current if current in [c.value for c in model_choices] else None,
                use_search_filter=True,
                use_jk_keys=False,
            ).ask()

            if chosen == "__custom__":
                chosen = questionary.text(
                    f"  输入 {tier} 的模型 ID:",
                    default=current,
                ).ask()

            if chosen:
                cfg.router.tiers.setdefault(tier, {})["model"] = chosen
            ctx_win = questionary.text(
                f"  {tier} context_window（用于 capability_gate，留空=未知）:",
                default=str(cfg.router.tiers.get(tier, {}).get("context_window", "")),
            ).ask()
            if ctx_win and ctx_win.strip():
                cfg.router.tiers.setdefault(tier, {})["context_window"] = int(ctx_win.strip())

        # ── 步骤 5: 路由策略配置 ──
        self._print_step(5, total_steps, "路由策略配置")

        cfg.router.default_tier = questionary.select(
            "默认 tier:",
            choices=[Choice(title=f"{t} — {cfg.router.tiers[t]['model']}", value=t) for t in TEXT_TIERS],
            default=cfg.router.default_tier,
        ).ask() or cfg.router.default_tier

        cfg.router.strategy = questionary.select(
            "路由策略:",
            choices=["auto(ML+启发式降级)", "v4_phase3", "heuristic"],
            default="auto(ML+启发式降级)",
        ).ask() or "auto"
        if cfg.router.strategy == "auto":
            cfg.router.strategy = "v4_phase3"

        cfg.router.confidence_threshold = float(questionary.text(
            "置信度阈值（0.0-1.0）:",
            default=str(cfg.router.confidence_threshold),
            validate=lambda v: 0 <= float(v) <= 1 or "请输入 0.0 到 1.0 之间的值",
        ).ask())

        cfg.router.complaint_upgrade_enabled = questionary.confirm(
            "启用投诉升级?", default=cfg.router.complaint_upgrade_enabled
        ).ask()

        # ── 保存 ──
        print()
        print("  ══════════════════════════════════════════════")
        print("  📝 配置摘要:")
        print(f"     目标 API:  {cfg.target_base_url}")
        print(f"     默认 Tier: {cfg.router.default_tier}")
        for tier in TEXT_TIERS:
            tcfg = cfg.router.tiers.get(tier, {})
            print(f"     {tier}: {tcfg.get('model')}  (ctx={tcfg.get('context_window')})")
        print(f"     路由策略:   {cfg.router.strategy}")
        print(f"     置信度阈值: {cfg.router.confidence_threshold}")
        print(f"     投诉升级:   {'启用' if cfg.router.complaint_upgrade_enabled else '禁用'}")
        print(f"  ══════════════════════════════════════════════")

        save_path = cfg.save()
        print(f"\n  ✅ 配置已保存到 {save_path}")
        print(f"  🚀 运行: python -m squilla_router_standalone")

        return cfg


# =========================================================================
# 代理服务
# =========================================================================


class ClaudeCodeProxy:
    """OpenSquilla Router 作为 Claude Code 的前置代理。"""

    def __init__(self, config: ProxyConfig):
        cfg = config
        self.config = cfg
        self.target_base_url = cfg.target_base_url.rstrip("/")
        self._api_key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""

        # 路由策略
        if cfg.router.strategy == "auto":
            cfg.router.strategy = "v4_phase3"
        self.router = Router(
            cfg.router,
            active_provider=cfg.active_provider,
            persist_decisions=True,
        )
        self._hold_store = RouterControlHoldStore()
        self._session_spend: dict[str, float] = {}
        # 文本模式覆盖的短时缓存：{session_key: (model, expiry_monotonic)}
        # 用于在单次用户请求的多轮工具调用中保持模型一致
        self._text_mode_hold: dict[str, tuple[str, float]] = {}
        self._stats: dict[str, Any] = {
            "total_requests": 0, "routes": {}, "sources": {}, "start_time": time.time(),
        }

        strategy_name = self.router.runtime_status().get("strategy", "heuristic")
        logger.info("代理初始化: target=%s, strategy=%s", self.target_base_url, strategy_name)

    # ---- 路由决策 ----

    def _extract_user_message(self, body: dict) -> tuple[str, list[dict]]:
        """提取用户最新一条消息文本 + 附件。"""
        messages = body.get("messages", [])
        attachments: list[dict] = []
        text = ""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "text":
                        texts.append(c.get("text", ""))
                    elif c.get("type") == "image" or (
                        isinstance(c.get("source"), dict)
                        and str(c["source"].get("media_type", "")).startswith("image/")
                    ):
                        attachments.append({"type": "image/png"})
                if texts:
                    text = texts[-1]
                    break
            elif isinstance(content, str) and content.strip():
                text = content.strip()
                break
        return text, attachments

    def _session_key_from_body(self, body: dict) -> str:
        """从 messages 前 2 条内容计算稳定 session key(同一对话内不变)。"""
        import hashlib
        messages = body.get("messages", [])
        fingerprint = ""
        for msg in messages[:2]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                content = " ".join(texts)
            elif not isinstance(content, str):
                content = str(content)
            fingerprint += f"{role}:{content[:200]}|"
        return "s-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

    def _estimate_input_tokens(self, body: dict) -> int:
        total = 0
        for msg in body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        total += len(c.get("text", ""))
        sys_text = body.get("system", "")
        if isinstance(sys_text, str):
            total += len(sys_text)
        elif isinstance(sys_text, list):
            for s in sys_text:
                if isinstance(s, dict):
                    total += len(s.get("text", ""))
        return max(total // 4, 1)

    def _project_tool_results(self, body: dict) -> dict:
        """扫描 messages 中的 tool result,投影大结果以节省 token。

        对每个 role="tool" 的消息,若 content 超过阈值,替换为
        ``[tool_result_projection]`` 摘要块(保留 head~65% + tail~35%)。
        """
        if not self.config.tool_result_projection_enabled:
            return body

        modified = dict(body)
        messages = list(modified.get("messages", []))
        proj_config = ProjectionConfig(
            max_preview_chars=self.config.tool_result_projection_max_chars,
            store_dir=self.config.tool_result_projection_store_dir,
        )
        projected_count = 0
        saved_chars = 0

        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) <= proj_config.max_preview_chars:
                continue

            tool_use_id = msg.get("tool_call_id", f"tool_{i}")
            # 尽量从相邻 assistant 消息中找 tool_name
            tool_name = "tool"
            if i > 0 and isinstance(messages[i - 1], dict):
                prev = messages[i - 1]
                if prev.get("role") == "assistant":
                    tc = prev.get("tool_calls") or []
                    if isinstance(tc, list) and len(tc) > 0:
                        for t in tc:
                            if isinstance(t, dict):
                                tid = t.get("id") or t.get("tool_use_id") or ""
                                if tid == tool_use_id:
                                    fn = t.get("function", {}) if isinstance(t, dict) else {}
                                    tool_name = fn.get("name", "tool") if isinstance(fn, dict) else "tool"
                                    break

            result = project_tool_result(
                content,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                config=proj_config,
                store_full=self.config.tool_result_projection_store_dir is not None,
            )
            messages[i] = dict(msg)
            messages[i]["content"] = result.block
            projected_count += 1
            saved_chars += result.omitted_chars

        modified["messages"] = messages
        if projected_count:
            logger.info(
                "tool_result_projection | projected=%d | saved_chars=%d",
                projected_count, saved_chars,
            )
        return modified

    def _apply_prompt_cache(self, body: dict) -> dict:
        """向 system prompt 注入 Anthropic prompt cache breakpoint。

        将 system prompt 转为 list-of-blocks 格式,在最后一个 block 上添加
        ``cache_control: {type: "ephemeral"}``,标记 Anthropic/OpenRouter 缓存。
        """
        if not self.config.prompt_cache_enabled:
            return body

        system = body.get("system")
        if not system:
            return body

        # 记录缓存指标(纯观测)
        meta = cache_system_prompt(
            system if isinstance(system, str) else str(system),
            enabled=True,
        )
        if meta.get("cache_enabled"):
            logger.debug(
                "prompt_cache | base_chars=%s | hash=%s",
                meta.get("cache_base_chars"), meta.get("cache_base_hash"),
            )

        # 注入 cache_control breakpoint
        if isinstance(system, str):
            modified = dict(body)
            modified["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
            return modified

        if isinstance(system, list):
            # 在最后一个 text block 上加 breakpoint
            blocks = list(system)
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                if isinstance(b, dict) and b.get("type") == "text":
                    b = dict(b)
                    b["cache_control"] = {"type": "ephemeral"}
                    blocks[i] = b
                    break
            modified = dict(body)
            modified["system"] = blocks
            return modified

        return body

    # ---- Text mode override (跳过路由，直接使用指定模型) ----

    def _extract_text_mode_override(self, body: dict) -> tuple[str | None, dict]:
        """扫描最新一条用户消息，提取 ``@model:xxx`` 标记。

        仅在最新一条 ``role=user`` 的消息中匹配，避免历史消息中的标记
        影响当前轮次。返回 (model_name, cleaned_body)，cleaned_body 中
        的标记已被移除，不会传递给目标 API。
        """
        import re
        messages = body.get("messages", [])
        if not messages:
            return None, body

        # 找到最新一条 role=user 的消息
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx < 0:
            return None, body

        msg = messages[last_user_idx]
        content = msg.get("content", "")

        if isinstance(content, str):
            match = re.search(TEXT_MODE_PATTERN, content)
            if match:
                model_spec = match.group(1)
                cleaned = re.sub(TEXT_MODE_PATTERN, "", content).strip()
                # 清理开头/结尾可能残留的空格、逗号等
                cleaned = cleaned.lstrip(",;:. \t\n\r").rstrip(",;:. \t\n\r")
                resolved = self._resolve_text_mode_model(model_spec)
                if resolved:
                    new_messages = list(messages)
                    new_messages[last_user_idx] = dict(msg)
                    new_messages[last_user_idx]["content"] = cleaned
                    modified_body = dict(body)
                    modified_body["messages"] = new_messages
                    return resolved, modified_body

        elif isinstance(content, list):
            # list 格式：遍历所有 text block
            for block in reversed(content):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text = block.get("text", "")
                match = re.search(TEXT_MODE_PATTERN, text)
                if match:
                    model_spec = match.group(1)
                    cleaned = re.sub(TEXT_MODE_PATTERN, "", text).strip()
                    cleaned = cleaned.lstrip(",;:. \t\n\r").rstrip(",;:. \t\n\r")
                    resolved = self._resolve_text_mode_model(model_spec)
                    if resolved:
                        new_messages = list(messages)
                        new_blocks = list(content)
                        for j, b in enumerate(content):
                            if b is block:
                                new_blocks[j] = dict(block)
                                new_blocks[j]["text"] = cleaned
                                break
                        new_messages[last_user_idx] = dict(msg)
                        new_messages[last_user_idx]["content"] = new_blocks
                        modified_body = dict(body)
                        modified_body["messages"] = new_messages
                        return resolved, modified_body

        return None, body

    def _resolve_text_mode_model(self, model_spec: str) -> str | None:
        """解析模型标识：如果是 tier（c0/c1/c2/c3），映射到配置的模型；否则直接用。

        ``@model:c3`` → 读取 config.tiers[c3].model 返回具体模型 ID
        ``@model:deepseek-v4-flash`` → 直接返回 ``deepseek-v4-flash``
        """
        from squilla_router_standalone.router_tiers import normalize_text_tier

        spec = model_spec.strip()
        if not spec:
            return None

        # 检查是否是 tier 名称
        tier = normalize_text_tier(spec)
        if tier:
            entry = self.router.config.tiers.get(tier, {})
            model = entry.get("model", "") if isinstance(entry, dict) else ""
            if model:
                return str(model)
            logger.warning("text_mode: tier %s has no configured model, using raw spec %s", tier, spec)
            return spec

        return spec

    async def _route(self, message: str, session_key: str, body: dict) -> dict[str, Any]:
        attachments = self._extract_user_message(body)[1]
        input_tokens = self._estimate_input_tokens(body)
        spend = self._session_spend.get(session_key, 0.0)

        result = await self.router.route(
            message,
            session_key=session_key,
            attachments=attachments,
            base_model=body.get("model", ""),
            session_spend_usd=spend if spend > 0 else None,
            session_cost_source="estimate" if spend > 0 else "unknown",
            material_estimated_tokens=input_tokens,
            hold_store=self._hold_store,
        )

        # 累计预估花费
        from squilla_router_standalone.pricing import lookup_price
        routed_price = lookup_price(result.model).input_per_m
        self._session_spend[session_key] = spend + (input_tokens / 1_000_000.0) * routed_price

        self._stats["total_requests"] += 1
        self._stats["routes"][result.tier] = self._stats["routes"].get(result.tier, 0) + 1
        self._stats["sources"][result.source] = self._stats["sources"].get(result.source, 0) + 1

        return {
            "tier": result.tier,
            "model": result.model,
            "confidence": result.confidence,
            "thinking_mode": result.thinking_mode,
            "thinking_level": result.thinking_level,
            "prompt_policy": result.prompt_policy,
            "prompt_hint": result.prompt_hint,
            "source": result.source,
            "routing_extra": result.routing_extra,
            "metadata": result.metadata,
        }

    # ---- HTTP 请求处理 ----

    def _resolve_api_key(self, request: Request) -> str | None:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return self._api_key or None

    def _apply_route_to_body(self, body: dict, route: dict[str, Any]) -> dict:
        modified = dict(body)
        modified["model"] = route["model"]

        # 注入 prompt hint
        hint = route.get("prompt_hint")
        if hint:
            messages = modified.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = f"{content}\n\n---\n[RESPONSE_POLICY: {hint}]"
                    break
                if isinstance(content, list):
                    for c in reversed(content):
                        if isinstance(c, dict) and c.get("type") == "text":
                            c["text"] = f"{c.get('text', '')}\n\n---\n[RESPONSE_POLICY: {hint}]"
                            break
                    break
            modified["messages"] = messages

        # 设置 thinking budget
        level = route.get("thinking_level")
        budget = _THINKING_BUDGET_TOKENS.get(level or "")
        if budget is not None:
            modified["thinking"] = {"type": "enabled", "budget_tokens": budget}
            modified.pop("temperature", None)
        elif route.get("tier") == "c0" and "thinking" in modified:
            modified.pop("thinking", None)

        return modified

    async def handle_messages(self, request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return Response(json.dumps({"error": {"message": "Invalid JSON body"}}),
                            status_code=400, media_type="application/json")

        user_message, _ = self._extract_user_message(body)
        # 用请求头或对话指纹推算 session key。
        # Claude Code 不发 x-session-id,所以用 messages 前 2 条内容 hash 做稳定标识。
        session_key = (
            request.headers.get("x-session-id")
            or request.headers.get("x-request-id")
            or self._session_key_from_body(body)
        )
        api_key = self._resolve_api_key(request)
        if not api_key:
            return Response(json.dumps({"error": {"message": "No API key provided"}}),
                            status_code=401, media_type="application/json")

        # ★ 检查文本模式覆盖（@model:xxx），命中则跳过路由，直接使用指定模型
        text_mode_model, body = self._extract_text_mode_override(body)
        if text_mode_model:
            # 新用户消息中检测到 @model:xxx → 缓存，供后续工具调用子请求使用
            self._text_mode_hold[session_key] = (text_mode_model, time.monotonic())
            route = {
                "tier": "text_mode",
                "model": text_mode_model,
                "confidence": 1.0,
                "thinking_mode": None,
                "thinking_level": None,
                "prompt_policy": None,
                "prompt_hint": None,
                "source": "text_mode_override",
                "routing_extra": {},
                "metadata": {},
            }
            logger.info("text_mode_override | session=%s | model=%s | msg_len=%d | msg=%.60s",
                        session_key, text_mode_model, len(user_message), user_message)
            # 从 body 中提取最新用户消息（标记已被移除），用于后续流程
            user_message, _ = self._extract_user_message(body)
        else:
            # 未检测到 @model:xxx → 检查是否是工具调用子请求
            messages = body.get("messages", [])
            latest_role = messages[-1].get("role", "") if messages else ""
            if latest_role == "tool" and session_key in self._text_mode_hold:
                # 工具调用子请求 → 使用缓存中的模型
                cached_model, _cached_ts = self._text_mode_hold[session_key]
                route = {
                    "tier": "text_mode",
                    "model": cached_model,
                    "confidence": 1.0,
                    "thinking_mode": None,
                    "thinking_level": None,
                    "prompt_policy": None,
                    "prompt_hint": None,
                    "source": "text_mode_override_cached",
                    "routing_extra": {},
                    "metadata": {},
                }
                logger.info("text_mode_override_cached | session=%s | model=%s | msg_len=%d",
                            session_key, cached_model, len(user_message))
            else:
                # 新用户消息或缓存不存在 → 清除缓存，走正常路由
                self._text_mode_hold.pop(session_key, None)
                route = await self._route(user_message, session_key, body)
                logger.info("路由 | session=%s | tier=%s | model=%s | conf=%.2f | source=%s | msg_len=%d | msg=%.60s",
                            session_key, route["tier"], route["model"], route["confidence"],
                            route["source"], len(user_message), user_message)

        # 1. 应用路由(替换 model/thinking/prompt_hint)
        modified_body = self._apply_route_to_body(body, route)
        # 2. 投影大工具结果(节省 token)
        modified_body = self._project_tool_results(modified_body)
        # 3. 注入 prompt cache breakpoint(节省输入 token)
        modified_body = self._apply_prompt_cache(modified_body)
        is_stream = modified_body.get("stream", True)
        target_url = f"{self.target_base_url}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
            "content-type": "application/json",
        }

        if is_stream:
            return await self._forward_stream(target_url, modified_body, headers, route)
        else:
            return await self._forward_sync(target_url, modified_body, headers, route)

    async def _forward_stream(self, url, body, headers, route):
        client = httpx.AsyncClient(timeout=300.0)
        try:
            req = client.build_request("POST", url, json=body, headers=headers)
            response = await client.send(req, stream=True)

            if response.status_code >= 400:
                error_body = await response.aread()
                return Response(error_body, status_code=response.status_code,
                                headers=dict(response.headers), media_type="application/json")

            async def generate():
                try:
                    route_event = f"event: squilla_route\ndata: {json.dumps(route, ensure_ascii=False)}\n\n"
                    yield route_event.encode("utf-8")
                    async for line in response.aiter_lines():
                        yield (line + "\n").encode("utf-8")
                except Exception as e:
                    logger.error("流式转发错误: %s", e)
                finally:
                    await client.aclose()

            resp_headers = {}
            skip_headers = {"transfer-encoding", "content-length", "content-encoding", "content-type"}
            for key, value in response.headers.items():
                if key.lower() not in skip_headers:
                    resp_headers[key] = value
            resp_headers.update({
                "x-squilla-tier": route["tier"],
                "x-squilla-model": route["model"],
                "x-squilla-confidence": f"{route['confidence']:.2f}",
                "x-squilla-thinking-mode": route.get("thinking_mode") or "",
            })
            return StreamingResponse(generate(), status_code=response.status_code,
                                     headers=resp_headers, media_type="text/event-stream")
        except httpx.ConnectError as e:
            logger.error("流式转发连接失败: %s", e)
            await client.aclose()
            return Response(json.dumps({"error": {"message": f"Cannot connect to {url}: {e}"}}),
                            status_code=502, media_type="application/json")
        except httpx.TimeoutException as e:
            logger.error("流式转发超时: %s", e)
            await client.aclose()
            return Response(json.dumps({"error": {"message": "Request timed out"}}),
                            status_code=504, media_type="application/json")
        except Exception as e:
            logger.error("流式转发异常: %s", e)
            await client.aclose()
            return Response(json.dumps({"error": {"message": f"Stream error: {e}"}}),
                            status_code=500, media_type="application/json")

    async def _forward_sync(self, url, body, headers, route):
        client = httpx.AsyncClient(timeout=300.0)
        try:
            response = await client.post(url, json=body, headers=headers)
            resp_data = response.json()
            resp_data["_squilla_route"] = {
                "tier": route["tier"], "model": route["model"],
                "confidence": route["confidence"], "thinking_mode": route.get("thinking_mode"),
            }
            resp_headers = {}
            skip_headers = {"transfer-encoding", "content-length", "content-encoding"}
            for key, value in response.headers.items():
                if key.lower() not in skip_headers:
                    resp_headers[key] = value
            resp_headers.update({
                "x-squilla-tier": route["tier"],
                "x-squilla-model": route["model"],
                "x-squilla-confidence": f"{route['confidence']:.2f}",
            })
            return Response(json.dumps(resp_data, ensure_ascii=False),
                            status_code=response.status_code, headers=resp_headers,
                            media_type="application/json")
        except httpx.ConnectError as e:
            logger.error("无法连接到目标 API: %s", e)
            return Response(json.dumps({"error": {"message": f"Cannot connect to {url}: {e}"}}),
                            status_code=502, media_type="application/json")
        except httpx.TimeoutException as e:
            logger.error("请求超时: %s", e)
            return Response(json.dumps({"error": {"message": "Request timed out"}}),
                            status_code=504, media_type="application/json")
        finally:
            await client.aclose()

    # ---- 管理端点 ----

    async def handle_status(self, request: Request) -> Response:
        uptime = time.time() - self._stats["start_time"]
        return Response(json.dumps({
            "service": "OpenSquilla Router Standalone",
            "version": "0.1.0",
            "runtime": self.router.runtime_status(),
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._stats["total_requests"],
            "routes": self._stats["routes"],
            "sources": self._stats["sources"],
            "active_sessions": len(self._session_spend),
            "tier_config": {t: {"model": c.get("model"), "context_window": c.get("context_window")}
                           for t, c in self.router.config.tiers.items() if t in TEXT_TIERS},
            "target_api": self.target_base_url,
        }, ensure_ascii=False, indent=2), media_type="application/json",
            headers={"access-control-allow-origin": "*"})

    async def handle_health(self, request: Request) -> Response:
        return Response(json.dumps({"status": "ok", "requests": self._stats["total_requests"]}),
                        media_type="application/json")

    async def handle_count_tokens(self, request: Request) -> Response:
        """Proxy /v1/messages/count_tokens to the target API (Claude Code needs it)."""
        api_key = self._resolve_api_key(request)
        if not api_key:
            return Response(json.dumps({"error": {"message": "No API key"}}), status_code=401)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return Response(json.dumps({"error": "Invalid JSON"}), status_code=400)
        target_url = f"{self.target_base_url}/v1/messages/count_tokens"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(target_url, json=body, headers=headers)
                return Response(await resp.aread(), status_code=resp.status_code,
                                headers=dict(resp.headers), media_type="application/json")
        except httpx.ConnectError:
            return Response(json.dumps({"error": f"Cannot connect to {target_url}"}), status_code=502)
        except httpx.TimeoutException:
            return Response(json.dumps({"error": "Request timed out"}), status_code=504)


# =========================================================================
# 应用工厂 + 入口
# =========================================================================


def create_app(proxy: ClaudeCodeProxy | None = None) -> Starlette:
    if proxy is None:
        proxy = ClaudeCodeProxy(ProxyConfig.build_with_auto_detect())
    return Starlette(routes=[
        Route("/v1/messages", proxy.handle_messages, methods=["POST"]),
        Route("/v1/messages/count_tokens", proxy.handle_count_tokens, methods=["POST"]),
        Route("/health", proxy.handle_health, methods=["GET"]),
        Route("/router-status", proxy.handle_status, methods=["GET"]),
    ])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenSquilla Router Standalone — 动态路由网关",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式配置向导
  python -m squilla_router_standalone --setup

  # 启动代理（自动读取 ~/.claude/claude_code_proxy.toml）
  python -m squilla_router_standalone

  # 查看路由统计
  curl http://127.0.0.1:8080/router-status

Claude Code 配置 (.claude/settings.json):
  {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
           "ANTHROPIC_API_KEY": "sk-...",
           "ANTHROPIC_MODEL": "claude-sonnet-4-6",
           "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-5-20251001"}}
        """,
    )

    parser.add_argument("--setup", action="store_true", help="运行交互式配置向导")
    parser.add_argument("--config", help="配置文件路径（默认 ~/.claude/claude_code_proxy.toml）")
    parser.add_argument("--host", help="监听地址（覆盖配置文件）")
    parser.add_argument("--port", type=int, help="监听端口（覆盖配置文件）")
    parser.add_argument("--target", help="目标 API 地址（覆盖配置文件）")
    parser.add_argument("--api-key", help="API key（覆盖配置文件）")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "heuristic", "v4_phase3"],
                        help="路由策略: auto=自动选择, heuristic=启发式, v4_phase3=ML 模型（默认 auto）")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="日志级别（默认 info）")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    import uvicorn

    # ── 交互式配置向导 ──
    if args.setup:
        SetupWizard().run()
        return

    # ── 正常启动 ──
    config_path = Path(args.config) if args.config else None
    cfg = ProxyConfig.build_with_auto_detect(config_path)

    # 命令行参数覆盖
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.target:
        cfg.target_base_url = args.target
    if args.api_key:
        cfg.api_key = args.api_key
    if args.strategy == "heuristic":
        cfg.router.strategy = "heuristic"
    elif args.strategy in ("auto", "v4_phase3"):
        cfg.router.strategy = "v4_phase3"

    proxy = ClaudeCodeProxy(cfg)
    app = create_app(proxy)

    api_key_display = "***" if cfg.api_key else "从请求头读取"
    logger.info("═" * 54)
    logger.info("  🚀 OpenSquilla Router Standalone")
    logger.info("═" * 54)
    logger.info(f"  监听地址:  http://{cfg.host}:{cfg.port}")
    logger.info(f"  目标 API:  {cfg.target_base_url}")
    logger.info(f"  API Key:   {api_key_display}")
    logger.info("")
    logger.info("  📊 Tier 配置:")
    for tier in TEXT_TIERS:
        tcfg = cfg.router.tiers.get(tier, {})
        logger.info(f"     {tier} → {tcfg.get('model')}  (ctx={tcfg.get('context_window')})")
    logger.info("")
    strategy_info = proxy.router.runtime_status()
    logger.info("  🧠 路由策略: %s", strategy_info.get("strategy"))
    logger.info("  🧠 门控策略:")
    logger.info(f"     置信度阈值: {cfg.router.confidence_threshold}")
    logger.info(f"     反降级窗口: {cfg.router.kv_cache_anti_downgrade_window_seconds}s")
    logger.info(f"     投诉升级:   {'启用' if cfg.router.complaint_upgrade_enabled else '禁用'}")
    logger.info(f"     默认 Tier:  {cfg.router.default_tier}")
    logger.info("")
    logger.info("  📋 Claude Code 配置:")
    logger.info(f"     ANTHROPIC_BASE_URL = http://{cfg.host}:{cfg.port}")
    logger.info("     ANTHROPIC_API_KEY = <你的 API key>")
    logger.info("     ANTHROPIC_MODEL = claude-sonnet-4-6")
    logger.info("     ANTHROPIC_SMALL_FAST_MODEL = claude-haiku-4-5-20251001")
    logger.info("")
    logger.info(f"  📈 路由状态: http://{cfg.host}:{cfg.port}/router-status")
    logger.info("  ⚙️  配置文件: 运行 --setup 重新配置")
    logger.info("═" * 54)

    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()