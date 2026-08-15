"""
文献管理功能增强
添加本地文献导入、Markdown渲染、默认路径设置
"""
from pathlib import Path
from typing import List, Dict, Any
import json

class LiteratureManager:
    """文献管理器"""

    def __init__(self, config_path: Path = Path(".literature_config.json")):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """加载配置"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding='utf-8'))
        return {
            "default_paths": ["文献"],
            "custom_paths": [],
            "recent_imports": []
        }

    def save_config(self):
        """保存配置"""
        self.config_path.write_text(
            json.dumps(self.config, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def add_custom_path(self, path: str):
        """添加自定义文献路径"""
        if path not in self.config["custom_paths"]:
            self.config["custom_paths"].append(path)
            self.save_config()

    def scan_literature(self) -> List[Dict[str, Any]]:
        """扫描所有文献"""
        literature = []

        # 扫描所有配置的路径
        all_paths = self.config["default_paths"] + self.config["custom_paths"]

        for base_path in all_paths:
            path = Path(base_path)
            if not path.exists():
                continue

            # 支持的文件格式
            for pattern in ['*.pdf', '*.md', '*.markdown', '*.txt']:
                for file in path.rglob(pattern):
                    literature.append({
                        "name": file.stem,
                        "path": str(file),
                        "type": file.suffix[1:],
                        "size": file.stat().st_size,
                        "modified": file.stat().st_mtime
                    })

        return literature

if __name__ == '__main__':
    manager = LiteratureManager()
    lit = manager.scan_literature()
    print(f'Found {len(lit)} literature files')
