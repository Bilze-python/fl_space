# Space Satellite Engineering Routing

完整规范维护在 `docs/references/engineering-standards.md` 和根目录
`CODING_STANDARDS.md`。

执行新功能时遵守：

- `mission` 可以调用 `environment` / `orbit`，反向依赖禁止；
- 数据对象使用 dataclass，并提供 `to_dict()` / `from_dict()`；
- 外观和物理属性分离；
- 外部网络服务只能在 integrations 适配器中；
- 自定义卫星、任务模式和经营规则优先使用注册表/插件；
- 公开 API 通过 `__init__.py` 和 `__all__` 导出；
- 新物理公式写清单位、假设和适用范围；
- 至少测试序列化、量级、非法输入和最小集成。
