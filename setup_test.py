"""
快速测试脚本 - 验证Web平台效果
"""
import sys
import json
sys.path.insert(0, '.')

from fl_space.cli import load_session, save_session

print("=== 配置测试参数 ===")
session = load_session()
session['tune']['rounds'] = 10
session['tune']['epochs'] = 2
session['tune']['batch_size'] = 64
session['tune']['dataset'] = 'mnist'
session['mount']['algo'] = 'fedavg'
session['mount']['sats'] = 3
session['mount']['stations'] = 2
save_session(session)

print("轮次:", session['tune']['rounds'])
print("算法:", session['mount']['algo'])
print("卫星:", session['mount']['sats'])
print("数据集:", session['tune']['dataset'])
print("\n测试配置已保存")
print("访问: http://127.0.0.1:8700")
