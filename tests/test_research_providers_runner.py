"""科研智能平台 v1：LLM 供应商层与后台批量执行器的离线单元测试。

覆盖：

- ``fl_space/research/providers.py``：配置来源优先级、保存/读回、三种协议解析、
  错误路径（全部离线，网络层被 monkeypatch）；
- ``scripts/run_research_plan.py``：计划校验、参数映射、结果提取、汇总、
  ``--dry-run`` 端到端以及失败/取消的退出码。

测试不加载 PyTorch，也不产生任何真实训练或网络请求。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_research_plan.py"
LOCAL_KEY = "sk-secret-1234567890"


def _import_providers() -> Any:
    """导入供应商模块。

    常规路径是 ``fl_space.research.providers``；但 ``fl_space/research/__init__.py``
    会立即导入 service 层，若并行开发中的同层模块尚未落地就会 ImportError，
    此时退化为按文件路径加载，保证供应商测试与其它的研究层模块解耦。
    """
    try:
        return importlib.import_module("fl_space.research.providers")
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "spacefl_research_providers",
            PROJECT_ROOT / "fl_space" / "research" / "providers.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # dataclass 需要能从 sys.modules 找到自身模块，按路径加载时必须先注册
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


providers_module = _import_providers()
ProviderConfig = providers_module.ProviderConfig
ProviderError = providers_module.ProviderError
chat = providers_module.chat
chat_json = providers_module.chat_json
load_provider = providers_module.load_provider
save_provider_config = providers_module.save_provider_config
# 注意：不能把 providers.test_provider 绑定成模块级名字，否则 pytest 会把它当成测试用例
run_probe = providers_module.test_provider

LOCAL_CONFIG = {
    "source": "local",
    "provider": "DeepSeek",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "wire_api": "chat",
    "api_key": LOCAL_KEY,
}
CC_SWITCH_PROVIDER = {
    "name": "iCode",
    "model": "icode-model",
    "base_url": "https://cc.test/v1",
    "token": "cc-switch-token",
    "protocol": "anthropic",
}

SAMPLE_PLAN: dict[str, Any] = {
    "id": "plan_20261105_120000",
    "title": "星簇规模与卸载收益",
    "goal": "研究星地链路受限条件下的聚合与卸载策略",
    "backend": "fedleo",
    "conditions": [
        {
            "id": "c1",
            "name": "3 轨面 × 2 星",
            "params": {
                "planes": 3,
                "sats_per_plane": 2,
                "gs": 3,
                "rounds": 3,
                "local_epochs": 1,
                "batch_size": 64,
                "lr": 0.02,
                "seed": 42,
                "offload_every": 2,
                "max_offload_iter": 2,
                "max_samples_per_client": 120,
                "dataset": "mnist",
                "device": "cpu",
            },
        },
        {"id": "c2", "name": "3 轨面 × 3 星", "params": {"sats_per_plane": 3}},
    ],
}


# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------


def _write_local_config(project_dir: Path, payload: dict[str, Any]) -> Path:
    """在项目目录写入 ``.spacefl_ai_config.json``。"""
    path = project_dir / ".spacefl_ai_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _chat_config(**overrides: Any) -> ProviderConfig:
    """构造一份可用的 chat 协议配置。"""
    fields: dict[str, Any] = {
        "configured": True,
        "source": "local",
        "provider": "Test",
        "model": "test-model",
        "base_url": "https://api.test/v1",
        "wire_api": "chat",
        "api_key": LOCAL_KEY,
    }
    fields.update(overrides)
    return ProviderConfig(**fields)


@pytest.fixture(scope="module")
def runner() -> Any:
    """按文件路径加载批处理执行器（scripts 目录不是包）。"""
    spec = importlib.util.spec_from_file_location("run_research_plan_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_plan(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """把计划写进临时目录。"""
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# providers：配置解析
# ---------------------------------------------------------------------------


def test_load_provider_unconfigured(tmp_path: Path) -> None:
    """没有任何来源时返回 configured=False / source='none'。"""
    config = load_provider(tmp_path, env={})
    assert config.configured is False
    assert config.source == "none"
    public = config.public()
    assert public["configured"] is False
    assert "api_key" not in public
    assert "masked_api_key" not in public


def test_load_provider_reads_local_config(tmp_path: Path) -> None:
    """本地配置优先，且 public()/repr 不泄漏 Key。"""
    _write_local_config(tmp_path, LOCAL_CONFIG)
    config = load_provider(tmp_path, env={})
    assert config.configured is True
    assert config.source == "local"
    assert config.provider == "DeepSeek"
    assert config.model == "deepseek-chat"
    assert config.base_url == "https://api.deepseek.com"
    assert config.wire_api == "chat"
    assert config.api_key == LOCAL_KEY

    public = config.public()
    assert "api_key" not in public
    assert public["masked_api_key"] == "sk-***7890"
    assert LOCAL_KEY not in json.dumps(public, ensure_ascii=False)
    assert LOCAL_KEY not in repr(config)


def test_load_provider_local_legacy_defaults(tmp_path: Path) -> None:
    """旧配置：OPENAI_API_KEY 字段 + 无 wire_api 时按 responses 处理。"""
    _write_local_config(
        tmp_path,
        {"source": "local", "OPENAI_API_KEY": "sk-legacy-abcdefgh", "model": "gpt-5.5"},
    )
    config = load_provider(tmp_path, env={})
    assert config.configured is True
    assert config.api_key == "sk-legacy-abcdefgh"
    assert config.wire_api == "responses"
    assert config.base_url == "https://api.openai.com/v1"


def test_load_provider_priority(tmp_path: Path) -> None:
    """优先级：本地配置 > CC Switch > 环境变量。"""
    project = tmp_path / "project"
    project.mkdir()
    env = {"DEEPSEEK_API_KEY": "sk-env-deepseek"}

    from_env = load_provider(project, env=env)
    assert from_env.source == "environment"
    assert from_env.provider == "DeepSeek"
    assert from_env.model == "deepseek-chat"
    assert from_env.base_url == "https://api.deepseek.com"

    from_cc = load_provider(project, cc_switch_provider=CC_SWITCH_PROVIDER, env=env)
    assert from_cc.source == "cc_switch"
    assert from_cc.provider == "iCode"
    assert from_cc.wire_api == "anthropic"
    assert from_cc.api_key == "cc-switch-token"

    _write_local_config(project, LOCAL_CONFIG)
    from_local = load_provider(project, cc_switch_provider=CC_SWITCH_PROVIDER, env=env)
    assert from_local.source == "local"
    assert from_local.provider == "DeepSeek"


def test_load_provider_environment_falls_back_to_openai(tmp_path: Path) -> None:
    """只有 OPENAI_API_KEY 时使用 OpenAI 默认值。"""
    config = load_provider(tmp_path, env={"OPENAI_API_KEY": "sk-openai-env"})
    assert config.source == "environment"
    assert config.provider == "OpenAI"
    assert config.model == "gpt-4o-mini"
    assert config.wire_api == "chat"


def test_load_provider_reads_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env 参数为 None 时读取真实环境变量（由 monkeypatch 提供）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-monkeypatched")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_provider(tmp_path)
    assert config.configured is True
    assert config.source == "environment"
    assert config.provider == "DeepSeek"
    assert config.api_key == "sk-env-monkeypatched"
    assert config.public()["masked_api_key"] == "sk-***ched"


def test_load_provider_ignores_non_local_config_marker(tmp_path: Path) -> None:
    """配置文件标记为 cc_switch 且没有传入 CC Switch 供应商时不算已配置。"""
    _write_local_config(tmp_path, {"source": "cc_switch", "app_type": "codex"})
    config = load_provider(tmp_path, env={})
    assert config.configured is False
    assert config.source == "none"


def test_load_provider_local_without_key_keeps_local_source(tmp_path: Path) -> None:
    """本地配置缺 Key 时仍保留 source='local'，方便前端提示补 Key。"""
    _write_local_config(
        tmp_path,
        {"source": "local", "provider": "iCode", "model": "gpt-5.5", "api_key": ""},
    )
    config = load_provider(tmp_path, env={})
    assert config.configured is False
    assert config.source == "local"
    assert config.provider == "iCode"


# ---------------------------------------------------------------------------
# providers：保存配置
# ---------------------------------------------------------------------------


def test_save_provider_config_roundtrip(tmp_path: Path) -> None:
    """保存后可被 load_provider 读回，public() 不含明文 Key。"""
    config_path = tmp_path / ".spacefl_ai_config.json"
    public = save_provider_config(
        config_path,
        {
            "provider": "DeepSeek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key": LOCAL_KEY,
            "wire_api": "chat",
        },
    )
    assert public["configured"] is True
    assert "api_key" not in public
    assert public["masked_api_key"] == "sk-***7890"

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["source"] == "local"
    assert raw["api_key"] == LOCAL_KEY
    assert raw["wire_api"] == "chat"

    loaded = load_provider(tmp_path, env={})
    assert loaded.configured is True
    assert loaded.api_key == LOCAL_KEY
    assert loaded.model == "deepseek-chat"
    assert not list(tmp_path.glob("*.tmp"))


def test_save_provider_config_keeps_previous_key(tmp_path: Path) -> None:
    """空 api_key 表示不修改已有 Key。"""
    config_path = tmp_path / ".spacefl_ai_config.json"
    save_provider_config(
        config_path,
        {"provider": "DeepSeek", "model": "deepseek-chat", "api_key": LOCAL_KEY},
    )
    public = save_provider_config(
        config_path,
        {
            "provider": "DeepSeek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key": "",
            "wire_api": "chat",
        },
    )
    assert public["masked_api_key"] == "sk-***7890"
    loaded = load_provider(tmp_path, env={})
    assert loaded.api_key == LOCAL_KEY
    assert loaded.base_url == "https://api.deepseek.com"


# ---------------------------------------------------------------------------
# providers：三种协议解析
# ---------------------------------------------------------------------------


def test_chat_parses_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat 协议解析 choices[0].message.content。"""
    calls: list[dict[str, Any]] = []

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        return {"choices": [{"message": {"content": "  你好世界  "}}]}

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    text = chat(_chat_config(), "系统提示", "用户问题")
    assert text == "你好世界"
    assert calls[0]["url"] == "https://api.test/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == f"Bearer {LOCAL_KEY}"
    assert calls[0]["payload"]["messages"][0] == {"role": "system", "content": "系统提示"}
    assert calls[0]["timeout"] == 90.0


def test_chat_falls_back_to_v1_endpoint_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_url 不含 /v1 时，首次 404 回退到 /v1/chat/completions。"""
    urls: list[str] = []

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        urls.append(url)
        if len(urls) == 1:
            raise providers_module._HttpStatusError(404, "请求失败：HTTP 404 Not Found")
        return {"choices": [{"text": "回退成功"}]}

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    assert chat(_chat_config(base_url="https://api.test"), "s", "u") == "回退成功"
    assert urls == [
        "https://api.test/chat/completions",
        "https://api.test/v1/chat/completions",
    ]


def test_chat_does_not_retry_when_base_url_has_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_url 已含 /v1 时 404 直接失败，不再拼第二段 /v1。"""
    urls: list[str] = []

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        urls.append(url)
        raise providers_module._HttpStatusError(404, "请求失败：HTTP 404 Not Found")

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    with pytest.raises(ProviderError):
        chat(_chat_config(), "s", "u")
    assert len(urls) == 1


def test_chat_parses_responses_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses 协议解析 output[].content[].text。"""
    seen: dict[str, Any] = {}

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        seen["url"] = url
        seen["payload"] = payload
        return {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "第一段"},
                        {"type": "text", "text": "第二段"},
                        {"type": "tool_call", "text": "忽略"},
                    ]
                }
            ]
        }

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    text = chat(_chat_config(wire_api="responses", base_url="https://api.test/v1"), "指令", "输入")
    assert text == "第一段第二段"
    assert seen["url"] == "https://api.test/v1/responses"
    assert seen["payload"]["instructions"] == "指令"
    assert seen["payload"]["input"] == "输入"
    assert seen["payload"]["max_output_tokens"] == 1800


def test_chat_responses_prefers_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses 协议优先使用顶层 output_text。"""
    monkeypatch.setattr(
        providers_module,
        "_post_json",
        lambda *args, **kwargs: {"output_text": "直出文本", "output": []},
    )
    assert chat(_chat_config(wire_api="responses"), "s", "u") == "直出文本"


def test_chat_parses_anthropic_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """anthropic 协议解析 content[].text，并带上必要请求头。"""
    seen: dict[str, Any] = {}

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = payload
        return {"content": [{"type": "text", "text": "你好"}, {"type": "text", "text": "，Claude"}]}

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    config = _chat_config(wire_api="anthropic", base_url="https://api.anthropic.com")
    assert chat(config, "sys", "user") == "你好，Claude"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == LOCAL_KEY
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["payload"]["system"] == "sys"


# ---------------------------------------------------------------------------
# providers：错误路径
# ---------------------------------------------------------------------------


def test_chat_requires_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配置时直接抛 ProviderError，且不发起请求。"""

    def fake_post(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("不应发起网络请求")

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    with pytest.raises(ProviderError):
        chat(ProviderConfig(), "s", "u")


def test_chat_rejects_unknown_wire_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知 wire_api 抛 ProviderError。"""

    def fake_post(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("不应发起网络请求")

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    with pytest.raises(ProviderError):
        chat(_chat_config(wire_api="grpc"), "s", "u")


def test_chat_redacts_api_key_in_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """错误信息带 HTTP 细节但不泄漏 Key。"""

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        raise ProviderError(f"请求失败：HTTP 401；bad key {LOCAL_KEY}")

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    with pytest.raises(ProviderError) as excinfo:
        chat(_chat_config(), "s", "u")
    message = str(excinfo.value)
    assert "HTTP 401" in message
    assert LOCAL_KEY not in message


def test_chat_raises_on_unparsable_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """响应结构不认识时抛 ProviderError。"""
    monkeypatch.setattr(providers_module, "_post_json", lambda *args, **kwargs: {"unexpected": []})
    with pytest.raises(ProviderError):
        chat(_chat_config(), "s", "u")


def test_chat_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """空内容是错误，不能静默返回空串。"""
    monkeypatch.setattr(
        providers_module,
        "_post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": "   "}}]},
    )
    with pytest.raises(ProviderError):
        chat(_chat_config(), "s", "u")


# ---------------------------------------------------------------------------
# providers：chat_json 与连通性测试
# ---------------------------------------------------------------------------


def test_chat_json_extracts_fenced_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """优先解析 ```json 代码块。"""
    content = '思考如下：\n```json\n{"score": 88, "items": [1, 2]}\n```\n完成'
    monkeypatch.setattr(
        providers_module,
        "_post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": content}}]},
    )
    assert chat_json(_chat_config(), "sys", "prompt") == {"score": 88, "items": [1, 2]}


def test_chat_json_extracts_braces_without_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有代码块时截取首尾花括号之间的内容。"""
    content = '结果：{"ok": true, "note": "含 } 的字符串"} 以上'
    monkeypatch.setattr(
        providers_module,
        "_post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": content}}]},
    )
    parsed = chat_json(_chat_config(), "sys", "prompt")
    assert parsed["ok"] is True
    assert parsed["note"] == "含 } 的字符串"


def test_chat_json_sends_schema_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """schema_hint 会拼进 user 提示。"""
    seen: dict[str, Any] = {}

    def fake_post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 90.0
    ) -> dict[str, Any]:
        seen["payload"] = payload
        return {"choices": [{"message": {"content": '{"a": 1}'}}]}

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    chat_json(_chat_config(), "sys", "请分析实验", schema_hint="summary: string")
    user_message = seen["payload"]["messages"][1]["content"]
    assert "请分析实验" in user_message
    assert "summary: string" in user_message


def test_chat_json_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 JSON 回复抛 ProviderError，交由调用方降级。"""
    monkeypatch.setattr(
        providers_module,
        "_post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": "这里没有 JSON"}}]},
    )
    with pytest.raises(ProviderError):
        chat_json(_chat_config(), "sys", "prompt")


def test_test_provider_reports_missing_config() -> None:
    """未配置时返回 ok=False（不抛异常）。"""
    result = run_probe(ProviderConfig())
    assert result["ok"] is False
    assert result["error"]


def test_test_provider_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """连通性测试成功时返回 ok=True 与毫秒耗时。"""
    monkeypatch.setattr(providers_module, "chat", lambda *args, **kwargs: "OK")
    result = run_probe(_chat_config())
    assert result["ok"] is True
    assert result["error"] == ""
    assert isinstance(result["latency_ms"], int)


def test_test_provider_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """连通性测试失败时返回结构化错误且隐藏 Key。"""

    def boom(*args: Any, **kwargs: Any) -> str:
        raise ProviderError(f"请求失败：HTTP 401 {LOCAL_KEY}")

    monkeypatch.setattr(providers_module, "chat", boom)
    result = run_probe(_chat_config())
    assert result["ok"] is False
    assert "HTTP 401" in result["error"]
    assert LOCAL_KEY not in result["error"]


# ---------------------------------------------------------------------------
# runner：计划校验与参数映射
# ---------------------------------------------------------------------------


def test_load_plan_valid(runner: Any, tmp_path: Path) -> None:
    """合法计划返回规范化结构（保留未知字段、补齐 name/params）。"""
    plan_path = _write_plan(tmp_path, SAMPLE_PLAN)
    plan = runner.load_plan(plan_path)
    assert plan["id"] == SAMPLE_PLAN["id"]
    assert plan["title"] == SAMPLE_PLAN["title"]
    assert plan["backend"] == "fedleo"
    assert [item["id"] for item in plan["conditions"]] == ["c1", "c2"]
    assert plan["conditions"][1]["name"] == "3 轨面 × 3 星"
    assert plan["goal"] == SAMPLE_PLAN["goal"]


def test_load_plan_defaults(runner: Any, tmp_path: Path) -> None:
    """缺省的 id/title/backend/params 会被补齐。"""
    payload = {"conditions": [{"id": "only", "params": None}]}
    plan = runner.load_plan(_write_plan(tmp_path, payload))
    assert plan["backend"] == "fedleo"
    assert plan["title"] == "未命名研究计划"
    assert plan["conditions"][0]["name"] == "条件 only"
    assert plan["conditions"][0]["params"] == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "p", "conditions": []},
        {"id": "p"},
        {"id": "p", "conditions": "c1"},
        {"id": "p", "conditions": [{"name": "缺少 id"}]},
        {"id": "p", "conditions": [{"id": "c1"}, {"id": "c1"}]},
        {"id": "p", "conditions": [{"id": "c1", "params": []}]},
        {"id": "p", "conditions": ["不是对象"]},
    ],
    ids=[
        "empty-conditions",
        "missing-conditions",
        "conditions-not-list",
        "missing-id",
        "duplicate-id",
        "params-not-object",
        "condition-not-object",
    ],
)
def test_load_plan_rejects_invalid(runner: Any, tmp_path: Path, payload: dict[str, Any]) -> None:
    """非法计划统一抛 ValueError。"""
    with pytest.raises(ValueError):
        runner.load_plan(_write_plan(tmp_path, payload))


def test_load_plan_rejects_non_object_and_missing_file(runner: Any, tmp_path: Path) -> None:
    """顶层不是对象、文件不存在都抛 ValueError。"""
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_plan(list_path)
    with pytest.raises(ValueError):
        runner.load_plan(tmp_path / "missing.json")


def test_build_run_kwargs_defaults(runner: Any, tmp_path: Path) -> None:
    """缺省参数完全按规格映射。"""
    condition_dir = tmp_path / "c9"
    kwargs = runner.build_run_kwargs({"id": "c9", "name": "默认", "params": {}}, condition_dir)
    assert kwargs == {
        "num_planes": 3,
        "sats_per_plane": 2,
        "num_ground_stations": 3,
        "num_rounds": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.02,
        "dataset": "mnist",
        "data_dir": "./data",
        "device": "cpu",
        "offload_every_n_rounds": 2,
        "max_offload_iter": 2,
        "max_samples_per_client": 120,
        "seed": 42,
        "verbose": False,
        "output_dir": str(condition_dir),
        "run_baseline": True,
    }


def test_build_run_kwargs_maps_params(runner: Any, tmp_path: Path) -> None:
    """计划字段逐项映射到 run_fedleo_vs_baseline 形参。"""
    params = {
        "planes": 4,
        "sats_per_plane": 3,
        "gs": 5,
        "rounds": 7,
        "local_epochs": 2,
        "batch_size": 32,
        "lr": 0.05,
        "seed": 2026,
        "offload_every": 3,
        "max_offload_iter": 4,
        "max_samples_per_client": 200,
        "dataset": "CIFAR10",
        "device": "CPU",
    }
    kwargs = runner.build_run_kwargs({"id": "c1", "params": params}, tmp_path / "c1")
    assert kwargs["num_planes"] == 4
    assert kwargs["sats_per_plane"] == 3
    assert kwargs["num_ground_stations"] == 5
    assert kwargs["num_rounds"] == 7
    assert kwargs["local_epochs"] == 2
    assert kwargs["batch_size"] == 32
    assert kwargs["learning_rate"] == 0.05
    assert kwargs["seed"] == 2026
    assert kwargs["offload_every_n_rounds"] == 3
    assert kwargs["max_offload_iter"] == 4
    assert kwargs["max_samples_per_client"] == 200
    assert kwargs["dataset"] == "cifar10"
    assert kwargs["device"] == "cpu"
    assert kwargs["data_dir"] == "./data"
    assert kwargs["run_baseline"] is True
    assert kwargs["verbose"] is False


def test_build_run_kwargs_accepts_experiment_names(runner: Any, tmp_path: Path) -> None:
    """直接写 run_fedleo_vs_baseline 形参名的计划也能映射。"""
    kwargs = runner.build_run_kwargs(
        {"id": "c1", "params": {"num_rounds": 5, "learning_rate": 0.5, "num_planes": 2}},
        tmp_path / "c1",
    )
    assert kwargs["num_rounds"] == 5
    assert kwargs["learning_rate"] == 0.5
    assert kwargs["num_planes"] == 2


def test_build_run_kwargs_rejects_invalid_number(runner: Any, tmp_path: Path) -> None:
    """非数值参数抛 ValueError，便于把条件标记为 FAILED。"""
    with pytest.raises(ValueError):
        runner.build_run_kwargs({"id": "c1", "params": {"rounds": "abc"}}, tmp_path / "c1")
    with pytest.raises(ValueError):
        runner.build_run_kwargs({"id": "c1", "params": {"rounds": 1.5}}, tmp_path / "c1")


# ---------------------------------------------------------------------------
# runner：结果提取与汇总
# ---------------------------------------------------------------------------


def test_extract_condition_result_from_runner_summary(runner: Any) -> None:
    """真实 comparison_summary.json 命名（fedleo/baseline）可正确提取。"""
    summary = {
        "config": {"num_planes": 3, "num_rounds": 3, "verbose": False},
        "fedleo": {
            "final_accuracy": 0.83,
            "peak_accuracy": 0.86,
            "total_offloaded": 120,
            "elapsed_sec": 40.5,
            "history": [
                {"total_delay": 12.0, "total_offloaded_samples": 60},
                {"total_delay": 15.0, "total_offloaded_samples": 60},
            ],
            "extra": {"data_balance": 1.0},
        },
        "baseline": {
            "final_accuracy": 0.8,
            "peak_accuracy": 0.82,
            "elapsed_sec": 30.0,
            "history": [{"accuracy": 0.8}],
        },
    }
    result = runner.extract_condition_result(
        {"id": "c1", "name": "3 轨面 × 2 星"}, summary, 75.25
    )
    assert result["id"] == "c1"
    assert result["name"] == "3 轨面 × 2 星"
    assert result["status"] == "DONE"
    assert result["elapsed_sec"] == 75.25
    assert result["final_accuracy"] == 0.83
    assert result["peak_accuracy"] == 0.86
    assert result["baseline_accuracy"] == 0.8
    assert result["accuracy_delta"] == pytest.approx(0.03)
    assert result["total_offloaded"] == 120
    assert result["total_delay_slots"] == 27.0
    assert result["passed"] is True
    assert "history" not in result["summary"]["offload_on"]
    assert result["summary"]["offload_off"]["rounds"] == 1
    assert result["summary"]["config"]["num_rounds"] == 3


def test_extract_condition_result_supports_validation_summary(runner: Any) -> None:
    """offload_on/offload_off + gates 的校验脚本产物也能提取。"""
    summary = {
        "offload_on": {
            "final_accuracy": 0.5,
            "peak_accuracy": 0.55,
            "total_offloaded": 7,
            "total_delay_slots": 99.0,
        },
        "offload_off": {"final_accuracy": 0.6},
        "deltas_on_minus_off": {"final_accuracy": -0.1},
        "gates": {"samples_conserved": True, "improved": False},
        "passed": False,
    }
    result = runner.extract_condition_result({"id": "c2"}, summary, 1.0)
    assert result["accuracy_delta"] == pytest.approx(-0.1)
    assert result["total_delay_slots"] == 99.0
    assert result["passed"] is False
    assert result["summary"]["gates"] == {"samples_conserved": True, "improved": False}
    assert result["summary"]["passed"] is False


def test_extract_condition_result_tolerates_empty_summary(runner: Any) -> None:
    """空/异常 summary 不抛异常，指标为 0 且 passed=False。"""
    result = runner.extract_condition_result({"id": "c3"}, {}, 0.5)
    assert result["status"] == "DONE"
    assert result["final_accuracy"] == 0.0
    assert result["passed"] is False


def test_summarize_conditions(runner: Any) -> None:
    """汇总只统计 DONE 且有数值结果的条件。"""
    results = [
        {"id": "c1", "name": "A", "status": "DONE", "final_accuracy": 0.8, "accuracy_delta": 0.02},
        {"id": "c2", "name": "B", "status": "DONE", "final_accuracy": 0.9, "accuracy_delta": 0.06},
        {"id": "c3", "name": "C", "status": "FAILED", "final_accuracy": None},
    ]
    summary = runner.summarize_conditions(results, 12.5)
    assert summary["mean_final_accuracy"] == pytest.approx(0.85)
    assert summary["max_final_accuracy"] == pytest.approx(0.9)
    assert summary["mean_delta_accuracy"] == pytest.approx(0.04)
    assert summary["best_condition"] == "c2"
    assert summary["best_condition_name"] == "B"
    assert summary["total_elapsed_sec"] == 12.5


def test_summarize_conditions_empty(runner: Any) -> None:
    """空结果集返回零值。"""
    summary = runner.summarize_conditions([], 0.0)
    assert summary["mean_final_accuracy"] == 0.0
    assert summary["max_final_accuracy"] == 0.0
    assert summary["best_condition"] == ""


# ---------------------------------------------------------------------------
# runner：端到端（dry-run / 失败 / 取消）
# ---------------------------------------------------------------------------


def test_dry_run_writes_progress_and_result(runner: Any, tmp_path: Path) -> None:
    """--dry-run 不训练，但产出完整的 progress.json 与 research_result.json。"""
    plan_path = _write_plan(tmp_path, SAMPLE_PLAN)
    output_dir = tmp_path / "job"
    exit_code = runner.main(["--plan", str(plan_path), "--output", str(output_dir), "--dry-run"])
    assert exit_code == 0

    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "DONE"
    assert progress["dry_run"] is True
    assert progress["total"] == 2
    assert progress["progress"] == 1.0
    assert progress["current"] == "c2"
    assert [item["status"] for item in progress["conditions"]] == ["DONE", "DONE"]
    assert [item["id"] for item in progress["conditions"]] == ["c1", "c2"]
    assert isinstance(progress["logs"], list)
    assert progress["updated_at"]
    assert not list(output_dir.glob("*.tmp"))

    result = json.loads((output_dir / "research_result.json").read_text(encoding="utf-8"))
    assert set(result) >= {
        "plan_id",
        "title",
        "backend",
        "generated_at",
        "conditions",
        "summary",
    }
    assert result["plan_id"] == SAMPLE_PLAN["id"]
    assert result["title"] == SAMPLE_PLAN["title"]
    assert result["backend"] == "fedleo"
    assert result["conditions"][0]["summary"]["kwargs"]["num_rounds"] == 3
    assert set(result["summary"]) >= {
        "mean_final_accuracy",
        "max_final_accuracy",
        "mean_delta_accuracy",
        "best_condition",
        "total_elapsed_sec",
    }


def _fake_done(condition: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """构造一个成功的条件执行结果。"""
    return (
        {
            "id": condition["id"],
            "name": condition["name"],
            "status": "DONE",
            "elapsed_sec": 1.0,
            "final_accuracy": 0.7,
            "peak_accuracy": 0.75,
            "baseline_accuracy": 0.6,
            "accuracy_delta": 0.1,
            "total_offloaded": 5,
            "total_delay_slots": 10.0,
            "passed": True,
            "summary": {},
        },
        "训练日志一行",
    )


def test_partial_failure_continues_and_exits_zero(
    runner: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单个条件失败只记 FAILED，继续下一个；有成功条件时退出码 0。"""
    plan_path = _write_plan(tmp_path, SAMPLE_PLAN)
    output_dir = tmp_path / "job"
    executed: list[str] = []

    def fake_execute(condition: dict[str, Any], condition_dir: Path) -> tuple[dict[str, Any], str]:
        executed.append(condition["id"])
        if condition["id"] == "c1":
            raise RuntimeError("模拟训练失败")
        return _fake_done(condition)

    monkeypatch.setattr(runner, "_execute_condition", fake_execute)
    exit_code = runner.main(["--plan", str(plan_path), "--output", str(output_dir)])
    assert exit_code == 0
    assert executed == ["c1", "c2"]

    result = json.loads((output_dir / "research_result.json").read_text(encoding="utf-8"))
    statuses = [item["status"] for item in result["conditions"]]
    assert statuses == ["FAILED", "DONE"]
    assert "模拟训练失败" in result["conditions"][0]["error"]
    assert result["conditions"][1]["final_accuracy"] == 0.7
    assert result["summary"]["best_condition"] == "c2"

    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "DONE"
    assert "训练日志一行" in progress["logs"]


def test_all_failed_exits_one(
    runner: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部条件失败时写 FAILED 且退出码 1。"""
    plan_path = _write_plan(tmp_path, SAMPLE_PLAN)
    output_dir = tmp_path / "job"

    def always_fail(condition: dict[str, Any], condition_dir: Path) -> tuple[dict[str, Any], str]:
        raise RuntimeError("训练异常")

    monkeypatch.setattr(runner, "_execute_condition", always_fail)
    assert runner.main(["--plan", str(plan_path), "--output", str(output_dir)]) == 1

    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "FAILED"
    result = json.loads((output_dir / "research_result.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in result["conditions"]] == ["FAILED", "FAILED"]
    assert result["summary"]["mean_final_accuracy"] == 0.0


def test_keyboard_interrupt_writes_cancelled(
    runner: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyboardInterrupt 写 CANCELLED 并退出码 2。"""
    plan_path = _write_plan(tmp_path, SAMPLE_PLAN)
    output_dir = tmp_path / "job"

    def interrupt(condition: dict[str, Any], condition_dir: Path) -> tuple[dict[str, Any], str]:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_execute_condition", interrupt)
    assert runner.main(["--plan", str(plan_path), "--output", str(output_dir)]) == 2

    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "CANCELLED"
    assert not (output_dir / "research_result.json").exists()


def test_invalid_plan_writes_failed_progress(runner: Any, tmp_path: Path) -> None:
    """计划非法时也写 progress.json（FAILED），返回退出码 1。"""
    plan_path = _write_plan(tmp_path, {"id": "p", "conditions": []})
    output_dir = tmp_path / "job"
    assert runner.main(["--plan", str(plan_path), "--output", str(output_dir)]) == 1
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "FAILED"
    assert "condition" in progress["error"]


def test_condition_timeout_is_only_recorded(runner: Any, tmp_path: Path) -> None:
    """--condition-timeout-sec 只记录到 progress.json，不改变执行结果。"""
    plan_path = _write_plan(tmp_path, SAMPLE_PLAN)
    output_dir = tmp_path / "job"
    exit_code = runner.main(
        [
            "--plan",
            str(plan_path),
            "--output",
            str(output_dir),
            "--dry-run",
            "--condition-timeout-sec",
            "120",
        ]
    )
    assert exit_code == 0
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["condition_timeout_sec"] == 120
