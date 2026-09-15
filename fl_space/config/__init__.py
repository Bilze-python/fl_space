"""
配置模块 — JSON/YAML 双格式配置加载。

提供：
    - schemas.py — Pydantic v2 配置契约（可选依赖）
    - yaml_loader.py — YAML 配置加载器
    - defaults.py — 实验预设（已存在）

用法::

    # JSON (默认，零依赖)
    from fl_space.config import load_json_config

    # YAML (可选，需 pip install pyyaml pydantic)
    from fl_space.config.yaml_loader import load_yaml_config
"""
