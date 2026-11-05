"""LLM 供应商抽象层：统一 OpenAI 兼容 / Responses / Anthropic 三种调用协议。

模块约定：

- 导入本模块不会发起任何网络请求；
- 所有失败（配置缺失、网络异常、响应不可解析）统一抛出 :class:`ProviderError`，
  调用方可据此降级到离线规则实现；
- API Key 只保存在 :class:`ProviderConfig.api_key`（``repr=False``），
  :meth:`ProviderConfig.public`、异常文本与日志中都不会出现明文 Key。

配置来源优先级（:func:`load_provider`）：本地 ``.spacefl_ai_config.json`` →
CC Switch 供应商 → 环境变量（``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CONFIG_FILE_NAME = ".spacefl_ai_config.json"
DEFAULT_TIMEOUT = 90.0
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
ERROR_BODY_LIMIT = 300
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.IGNORECASE | re.DOTALL)

#: ``wire_api`` 别名，兼容不同前端/旧配置的写法。
WIRE_API_ALIASES: dict[str, str] = {
    "chat": "chat",
    "chat_completions": "chat",
    "openai": "chat",
    "responses": "responses",
    "response": "responses",
    "anthropic": "anthropic",
    "messages": "anthropic",
    "claude": "anthropic",
}


class ProviderError(RuntimeError):
    """LLM 供应商调用失败（配置缺失、网络错误、响应不可解析等）。"""


class _HttpStatusError(ProviderError):
    """内部使用：携带 HTTP 状态码，便于实现 404 时的端点回退。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


@dataclass
class ProviderConfig:
    """一条可用的（或缺失的）LLM 供应商配置。

    ``api_key`` 使用 ``repr=False``，保证 ``repr(config)`` / 日志不会泄漏密钥。
    """

    configured: bool = False
    source: str = "none"  # environment | local | cc_switch | none
    provider: str = ""  # 展示名，如 DeepSeek / iCode
    model: str = ""
    base_url: str = ""
    wire_api: str = "chat"  # chat | responses | anthropic
    api_key: str = field(default="", repr=False)
    timeout: float = 90.0
    # 兼容 web/server.py 中 Responses 协议的扩展字段
    reasoning_effort: str = ""
    disable_response_storage: bool = False

    def public(self) -> dict[str, Any]:
        """返回给前端的配置（不含 api_key，配置了则 masked_api_key='sk-***abcd'）。"""
        info: dict[str, Any] = {
            "configured": bool(self.configured),
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "wire_api": self.wire_api,
            "timeout": float(self.timeout),
        }
        masked = _mask_api_key(self.api_key)
        if masked:
            info["masked_api_key"] = masked
        return info


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """返回 UTC ISO 时间字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mask_api_key(api_key: str) -> str:
    """把 Key 掩码成 ``sk-***abcd`` 形式；空 Key 返回空串。"""
    key = str(api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return f"{key[:3]}***{key[-4:]}"


def _redact(text: str, api_key: str) -> str:
    """把文本中出现的明文 Key 替换成 ``***``。"""
    key = str(api_key or "").strip()
    message = str(text or "")
    if key and key in message:
        return message.replace(key, "***")
    return message


def _coerce_timeout(value: Any) -> float:
    """把配置里的超时值转成正整数秒；非法值回落到默认超时。"""
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    return timeout if timeout > 0 else DEFAULT_TIMEOUT


def _read_json_file(path: Path) -> dict[str, Any]:
    """读取 JSON 对象文件；不存在或非法时返回空字典（不打印、不抛异常）。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 JSON 文件（先写临时文件再 replace）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def _normalize_wire_api(value: Any) -> str:
    """归一化 ``wire_api``，未知写法原样（小写）返回，由 :func:`chat` 报错。"""
    key = str(value or "").strip().lower()
    return WIRE_API_ALIASES.get(key, key or "chat")


def _env_get(env: Mapping[str, str] | None, name: str) -> str:
    """读取环境变量值；``env`` 为 ``None`` 时回落到 ``os.environ``。"""
    source: Mapping[str, str] = os.environ if env is None else env
    return str(source.get(name) or "").strip()


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------


def _local_config(project_dir: Path) -> ProviderConfig | None:
    """解析 ``<project>/.spacefl_ai_config.json``；``source`` 不是 local 时返回 None。"""
    data = _read_json_file(Path(project_dir) / CONFIG_FILE_NAME)
    if data.get("source") != "local":
        return None
    api_key = str(data.get("api_key") or data.get("OPENAI_API_KEY") or "").strip()
    return ProviderConfig(
        configured=bool(api_key),
        source="local",
        provider=str(data.get("provider") or data.get("name") or "iCode").strip(),
        model=str(data.get("model") or "gpt-5.5").strip(),
        base_url=str(data.get("base_url") or DEFAULT_OPENAI_BASE_URL).strip(),
        wire_api=_normalize_wire_api(data.get("wire_api") or "responses"),
        api_key=api_key,
        timeout=_coerce_timeout(data.get("timeout")),
        reasoning_effort=str(data.get("model_reasoning_effort") or "").strip(),
        disable_response_storage=bool(data.get("disable_response_storage")),
    )


def _cc_switch_config(raw: Mapping[str, Any] | None) -> ProviderConfig | None:
    """解析 Web 层传入的 CC Switch 供应商；没有令牌时返回 None。"""
    if not isinstance(raw, Mapping):
        return None
    token = str(raw.get("token") or raw.get("api_key") or "").strip()
    if not token:
        return None
    return ProviderConfig(
        configured=True,
        source="cc_switch",
        provider=str(raw.get("name") or "CC Switch").strip(),
        model=str(raw.get("model") or "gpt-5.5").strip(),
        base_url=str(raw.get("base_url") or DEFAULT_OPENAI_BASE_URL).strip(),
        wire_api=_normalize_wire_api(raw.get("protocol") or "responses"),
        api_key=token,
    )


def _environment_config(env: Mapping[str, str] | None) -> ProviderConfig | None:
    """解析环境变量供应商：DeepSeek 优先，其次 OpenAI。"""
    deepseek_key = _env_get(env, "DEEPSEEK_API_KEY")
    if deepseek_key:
        return ProviderConfig(
            configured=True,
            source="environment",
            provider="DeepSeek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            wire_api="chat",
            api_key=deepseek_key,
        )
    openai_key = _env_get(env, "OPENAI_API_KEY")
    if openai_key:
        return ProviderConfig(
            configured=True,
            source="environment",
            provider="OpenAI",
            model="gpt-4o-mini",
            base_url=DEFAULT_OPENAI_BASE_URL,
            wire_api="chat",
            api_key=openai_key,
        )
    return None


def load_provider(
    project_dir: Path,
    *,
    cc_switch_provider: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> ProviderConfig:
    """按顺序解析供应商配置。

    顺序为：

    1. ``<project>/.spacefl_ai_config.json`` 且 ``source == "local"``；
    2. ``cc_switch_provider``（由 Web 层解析后传入的
       ``{"name","model","base_url","token","protocol"}``）；
    3. 环境变量 ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``。

    都没有（或本地配置缺少 Key 且无其它来源）时返回 ``configured=False`` 的配置；
    本地配置存在但缺 Key 时仍保留 ``source="local"``，方便前端提示用户补 Key。

    Parameters
    ----------
    project_dir : Path
        项目根目录，用于定位 ``.spacefl_ai_config.json``。
    cc_switch_provider : dict | None
        Web 层解析好的 CC Switch 当前供应商。
    env : dict | None
        显式环境变量映射，便于测试；``None`` 时读取 ``os.environ``。

    Returns
    -------
    ProviderConfig
        解析结果，``configured`` 表示是否可以直接发起调用。
    """
    local = _local_config(Path(project_dir))
    if local is not None and local.configured:
        return local
    remote = _cc_switch_config(cc_switch_provider)
    if remote is not None:
        return remote
    from_env = _environment_config(env)
    if from_env is not None:
        return from_env
    if local is not None:
        return local
    return ProviderConfig()


def save_provider_config(config_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """保存前端提交的供应商配置并返回 public 配置（不含明文 Key）。

    写入字段：``provider`` / ``model`` / ``base_url`` / ``api_key`` / ``wire_api``，
    并标记 ``source="local"``。``api_key`` 为空表示不修改已有 Key。
    """
    path = Path(config_path)
    existing = _read_json_file(path)
    submitted = data if isinstance(data, Mapping) else {}

    api_key = str(submitted.get("api_key") or "").strip()
    if not api_key:
        api_key = str(existing.get("api_key") or existing.get("OPENAI_API_KEY") or "").strip()
    reasoning_effort = str(
        submitted.get("model_reasoning_effort") or existing.get("model_reasoning_effort") or ""
    ).strip()
    disable_storage = bool(
        submitted.get("disable_response_storage", existing.get("disable_response_storage"))
    )
    config = ProviderConfig(
        configured=bool(api_key),
        source="local",
        provider=str(
            submitted.get("provider") or existing.get("provider") or existing.get("name") or "iCode"
        ).strip(),
        model=str(submitted.get("model") or existing.get("model") or "gpt-5.5").strip(),
        base_url=str(
            submitted.get("base_url") or existing.get("base_url") or DEFAULT_OPENAI_BASE_URL
        ).strip(),
        wire_api=_normalize_wire_api(
            submitted.get("wire_api") or existing.get("wire_api") or "chat"
        ),
        api_key=api_key,
        timeout=_coerce_timeout(submitted.get("timeout") or existing.get("timeout")),
        reasoning_effort=reasoning_effort,
        disable_response_storage=disable_storage,
    )

    payload: dict[str, Any] = {
        "source": "local",
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "wire_api": config.wire_api,
        "api_key": config.api_key,
    }
    if config.reasoning_effort:
        payload["model_reasoning_effort"] = config.reasoning_effort
    if config.disable_response_storage:
        payload["disable_response_storage"] = True
    _atomic_write_json(path, payload)
    return config.public()


# ---------------------------------------------------------------------------
# HTTP 传输
# ---------------------------------------------------------------------------


def _read_error_body(exc: HTTPError) -> str:
    """读取 HTTPError 响应体的前 300 字（失败时返回空串）。"""
    try:
        raw = exc.read()
    except Exception:
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")[:ERROR_BODY_LIMIT]


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST JSON 并解析 JSON 响应，所有传输/解析错误统一转成 ProviderError。"""
    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    merged_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    merged_headers.update({str(key): str(value) for key, value in headers.items() if value})
    request = Request(url, data=body, headers=merged_headers, method="POST")

    try:
        with urlopen(request, timeout=float(timeout)) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = _read_error_body(exc)
        message = f"请求失败：HTTP {exc.code} {exc.reason}"
        if detail:
            message = f"{message}；{detail}"
        raise _HttpStatusError(exc.code, message) from None
    except URLError as exc:
        raise ProviderError(f"网络连接失败：{getattr(exc, 'reason', exc)}") from None
    except OSError as exc:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            raise ProviderError(f"请求超时（{float(timeout):.0f} 秒）") from exc
        raise ProviderError(f"网络请求异常：{exc}") from exc

    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ProviderError(f"响应不是合法 JSON：{text[:ERROR_BODY_LIMIT]}") from exc
    if not isinstance(data, dict):
        raise ProviderError("供应商响应不是 JSON 对象")
    return data


# ---------------------------------------------------------------------------
# 协议实现
# ---------------------------------------------------------------------------


def _require_config(config: ProviderConfig) -> None:
    """校验配置可用于发起调用。"""
    if not (config.configured or str(config.api_key or "").strip()):
        raise ProviderError("LLM 供应商尚未配置，请先填写 API Key")
    if not str(config.model or "").strip():
        raise ProviderError("LLM 供应商缺少模型名称")
    if not str(config.base_url or "").strip():
        raise ProviderError("LLM 供应商缺少 base_url")


def _base_url(config: ProviderConfig) -> str:
    """去掉尾部斜杠的 base_url。"""
    return str(config.base_url or "").strip().rstrip("/")


def _auth_headers(config: ProviderConfig) -> dict[str, str]:
    """构造 Bearer 认证头（无 Key 时为空字典，便于对接本地网关）。"""
    key = str(config.api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _content_to_text(content: Any) -> str:
    """把 chat 协议的 ``message.content``（字符串或分段列表）拼成文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = [
            str(block.get("text") or "") for block in content if isinstance(block, Mapping)
        ]
        return "".join(chunks).strip()
    return ""


def _complete_chat(
    config: ProviderConfig,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """OpenAI 兼容 ``/chat/completions``，404 时回退到 ``/v1/chat/completions``。"""
    root = _base_url(config)
    payload = {
        "model": config.model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = _auth_headers(config)
    try:
        data = _post_json(f"{root}/chat/completions", payload, headers, config.timeout)
    except _HttpStatusError as exc:
        if exc.status != 404 or root.endswith("/v1"):
            raise
        data = _post_json(f"{root}/v1/chat/completions", payload, headers, config.timeout)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            content = message.get("content") if isinstance(message, Mapping) else message
            text = _content_to_text(content)
            if not text:
                text = str(first.get("text") or "").strip()
            if text:
                return text
    raise ProviderError("模型响应格式无法识别（缺少 choices[0].message.content）")


def _complete_responses(
    config: ProviderConfig,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    """OpenAI Responses 协议：解析 ``output_text`` 或 ``output[].content[].text``。"""
    payload: dict[str, Any] = {
        "model": config.model,
        "instructions": system,
        "input": user,
        "max_output_tokens": int(max_tokens),
    }
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    if config.disable_response_storage:
        payload["store"] = False
    data = _post_json(
        f"{_base_url(config)}/responses",
        payload,
        _auth_headers(config),
        config.timeout,
    )

    text = data.get("output_text")
    result = text.strip() if isinstance(text, str) else ""
    if not result:
        chunks: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    if block.get("type") in {"output_text", "text"}:
                        chunks.append(str(block.get("text") or ""))
        result = "".join(chunks).strip()
    if not result:
        raise ProviderError("模型响应格式无法识别（缺少 output_text）")
    return result


def _complete_anthropic(
    config: ProviderConfig,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Anthropic Messages 协议：解析 ``content[].text``。"""
    root = _base_url(config)
    endpoint = f"{root}/messages" if root.endswith("/v1") else f"{root}/v1/messages"
    payload = {
        "model": config.model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {"x-api-key": str(config.api_key or "").strip(), "anthropic-version": "2023-06-01"}
    # Claude Code 中转服务有的只认 Authorization，这里两个头都带上（与 web/server.py 一致）
    headers.update(_auth_headers(config))
    data = _post_json(endpoint, payload, headers, config.timeout)

    blocks = data.get("content")
    if isinstance(blocks, list):
        chunks = [
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, Mapping) and block.get("text")
        ]
        text = "".join(chunks).strip()
        if text:
            return text
    raise ProviderError("模型响应格式无法识别（缺少 content[].text）")


def chat(
    config: ProviderConfig,
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> str:
    """调用大模型并返回文本；任何失败都抛 :class:`ProviderError`。"""
    _require_config(config)
    wire_api = _normalize_wire_api(config.wire_api)
    try:
        if wire_api == "chat":
            text = _complete_chat(config, system, user, temperature, max_tokens)
        elif wire_api == "responses":
            text = _complete_responses(config, system, user, max_tokens)
        elif wire_api == "anthropic":
            text = _complete_anthropic(config, system, user, temperature, max_tokens)
        else:
            raise ProviderError(f"不支持的 wire_api：{config.wire_api or '（空）'}")
    except _HttpStatusError as exc:
        raise _HttpStatusError(exc.status, _redact(str(exc), config.api_key)) from None
    except ProviderError as exc:
        raise ProviderError(_redact(str(exc), config.api_key)) from None
    if not text.strip():
        raise ProviderError("模型返回了空内容")
    return text.strip()


def _find_json_block(text: str) -> str:
    """从模型回复中截取 JSON 片段（优先 ```json 代码块，其次首尾花括号）。"""
    raw = str(text or "").strip()
    match = JSON_FENCE_PATTERN.search(raw)
    if match:
        raw = match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw


def chat_json(
    config: ProviderConfig,
    system: str,
    user: str,
    *,
    schema_hint: str = "",
) -> dict[str, Any]:
    """要求模型输出 JSON 并解析成字典；解析失败抛 :class:`ProviderError`。"""
    instruction = "请只输出一个 JSON 对象，不要输出任何解释文字。"
    if schema_hint:
        instruction = f"{instruction}期望字段：{schema_hint}"
    text = chat(config, system, f"{user}\n\n{instruction}")
    candidate = _find_json_block(text)
    try:
        data = json.loads(candidate)
    except ValueError as exc:
        raise ProviderError("模型返回内容不是合法 JSON") from exc
    if not isinstance(data, dict):
        raise ProviderError("模型返回的 JSON 不是对象")
    return data


def test_provider(config: ProviderConfig) -> dict[str, Any]:
    """轻量连通性测试，返回 ``{"ok", "latency_ms", "error"}``，永不抛异常。"""
    if not (config.configured or str(config.api_key or "").strip()):
        return {"ok": False, "latency_ms": 0, "error": "尚未配置 LLM 供应商"}
    started = time.monotonic()
    try:
        reply = chat(
            config,
            "你是连通性检查助手，只回复 OK。",
            "请回复 OK",
            temperature=0.0,
            max_tokens=64,
        )
    except ProviderError as exc:
        return {
            "ok": False,
            "latency_ms": _elapsed_ms(started),
            "error": _redact(str(exc), config.api_key),
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": _elapsed_ms(started),
            "error": _redact(f"未知错误：{exc}", config.api_key),
        }
    latency_ms = _elapsed_ms(started)
    if not reply.strip():
        return {"ok": False, "latency_ms": latency_ms, "error": "模型未返回内容"}
    return {"ok": True, "latency_ms": latency_ms, "error": ""}


def _elapsed_ms(started: float) -> int:
    """返回自 ``started``（time.monotonic）以来的毫秒数。"""
    return round((time.monotonic() - started) * 1000)
