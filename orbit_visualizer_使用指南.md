# SpaceFL 轨道可视化独立工具 - 使用指南

## 功能说明

这是一个独立的轨道可视化工具，比网页版更专业和清晰。

## 依赖安装

```bash
# 基础依赖（必须）
pip install matplotlib numpy

# 高级地图支持（推荐）
pip install cartopy
```

## 使用步骤

### 1. 生成轨道数据
```bash
python -m fl_space.cli run simulate --sats 10 --stations 5 --hours 2 --output orbit_data.json
```

### 2. 运行可视化工具
```bash
python orbit_visualizer.py
```

然后选择：
- `1` - 显示静态图（单个时隙）
- `2` - 显示动画（所有时隙）
- `3` - 保存为GIF动画

## 特性

### ✓ 已实现
- 专业的地球投影显示
- 卫星轨道动画
- 地面站标记
- 卫星-地面站连接线
- ISL星间链路显示
- 时间和统计信息

### 对比网页版
- ✓ 更清晰的视觉效果
- ✓ 专业的地理投影
- ✓ 更好的性能
- ✓ 可导出高质量图片/动画

## 下一步

根据你的测试结果，我可以：
1. 继续优化这个独立工具
2. 打包成exe可执行文件
3. 或集成回网页（如果这个版本效果好）

请先测试这个版本的效果！
