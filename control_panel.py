#!/usr/bin/env python3
"""
SpaceFL 中控面板 — 一站式操作控制台
用法: python control_panel.py [--lang en|zh]
"""
import os
import json
import subprocess
import sys
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
TEMPLATE_DIR = os.path.join(PROJECT_DIR, "config_templates")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)

# ── 语言 ──────────────────────────────────────────────────────
LANG = "en"
for i, arg in enumerate(sys.argv):
    if arg == "--lang" and i + 1 < len(sys.argv):
        LANG = sys.argv[i + 1]

T = {
    "title": {"en": "SpaceFL Control Panel", "zh": "SpaceFL 中控面板"},
    "lang_label": {"en": "Language: EN", "zh": "语言: 中文"},
    "select": {"en": "Select", "zh": "请选择"},
    "back": {"en": "Back to main menu", "zh": "返回主菜单"},
    "press_enter": {"en": "Press Enter to continue...", "zh": "按回车继续..."},
    "invalid": {"en": "Invalid choice!", "zh": "无效选项!"},
    "exit_msg": {"en": "SpaceFL Control Panel closed. Goodbye!", "zh": "SpaceFL 中控面板已关闭，感谢使用!"},
    "confirm": {"en": "Confirm? (y/n)", "zh": "确认? (y/n)"},
    "running": {"en": "Running...", "zh": "正在运行..."},
    "done": {"en": "Done!", "zh": "完成!"},
    "error": {"en": "Error occurred", "zh": "执行出错"},
    "warning_long": {"en": "WARNING: This may take a long time!", "zh": "警告: 可能需要较长时间!"},
    "output_dir": {"en": "Output", "zh": "输出目录"},
}

def t(key):
    return T.get(key, {}).get(LANG, key)

def cls():
    os.system("cls" if os.name == "nt" else "clear")

def pause():
    input(f"\n  {t('press_enter')}")

def run(cmd, **kwargs):
    """运行命令并返回是否成功"""
    print(f"\n  {t('running')}")
    print(f"  > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print()
    result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=PROJECT_DIR, **kwargs)
    if result.returncode != 0:
        print(f"\n  [{t('error')}: code={result.returncode}]")
    return result.returncode


def run_feedback(cmd, success="执行成功"):
    """Run a command and explain success/failure in the terminal."""
    code = run(cmd)
    if code == 0:
        print(f"\n  [成功] {success}")
    else:
        print("\n  [失败] 命令未完成。请根据上方错误信息修正参数；常见原因：")
        print("        类型不匹配、值超出允许范围、依赖未安装或输入文件不存在。")
    return code

def ask(prompt, default=""):
    if default:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default="n"):
    value = ask(f"{prompt} (y/n)", default).lower()
    return value in ("y", "yes", "是")


def save_template(name):
    """Save current session as a reusable JSON template."""
    from fl_space.cli import load_session
    clean = name.strip() or datetime.now().strftime("template_%Y%m%d_%H%M%S")
    path = os.path.join(TEMPLATE_DIR, clean if clean.endswith(".json") else clean + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(load_session(), f, ensure_ascii=False, indent=2)
    print(f"  [成功] 参数模板已保存: {path}")
    return path


def list_templates():
    return sorted(x for x in os.listdir(TEMPLATE_DIR) if x.endswith(".json"))


def menu_templates():
    while True:
        c = menu("Parameter Templates / 参数模板", [
            ("a", "Import a template (load only, do not run)", "导入模板（只加载，不直接运行）"),
            ("b", "Save current session as template", "保存当前参数为模板"),
            ("c", "List saved templates", "查看历史模板"),
        ])
        if c == "a":
            names = list_templates()
            if not names:
                print("  [提示] 暂无模板。请先保存当前参数。")
            else:
                for i, name in enumerate(names, 1):
                    print(f"    {i}. {name}")
                raw = ask("输入模板编号或文件名")
                try:
                    name = names[int(raw) - 1] if raw.isdigit() else raw
                    path = os.path.join(TEMPLATE_DIR, name)
                    run_feedback([sys.executable, "-m", "fl_space.cli", "mount", "config", path],
                                 "模板已导入；当前仅加载配置，请到 FL 训练菜单确认后运行")
                except (IndexError, ValueError):
                    print("  [失败] 模板编号无效")
            pause()
        elif c == "b":
            save_template(ask("模板名称（如 mnist_fedavg）"))
            pause()
        elif c == "c":
            names = list_templates()
            print("  已保存模板:" if names else "  [提示] 暂无模板")
            for name in names:
                print(f"    {name}")
            pause()
        elif c in ("x", ""):
            return

# ── 菜单渲染 ──────────────────────────────────────────────────
def header():
    cls()
    print(f"""
  ============================================================
    {t('title')}   [{t('lang_label')}]
  ============================================================""")

def menu(title, items):
    """items: list of (key, label_en, label_zh)"""
    header()
    print(f"  --- {title} ---")
    print()
    for key, en, zh in items:
        label = zh if LANG == "zh" else en
        print(f"    {key}. {label}")
    print()
    print(f"    x. {t('back')}" if LANG == "zh" else f"    x. {t('back')}")
    print()
    choice = input(f"  {t('select')}: ").strip().lower()
    return choice

# ═══════════════════════════════════════════════════════════════
#  各子菜单
# ═══════════════════════════════════════════════════════════════

def menu_demos():
    while True:
        c = menu("Quick Demos / 快速演示", [
            ("a", "Constellation Viz (5 scenes)", "星座可视化 (5个预设场景)"),
            ("b", "Environment Demo (bodies/orbits/mars...)", "环境模拟 (天体/轨道/火星等)"),
            ("c", "Run All Demos", "全部运行"),
            ("d", "Basic Demo (orbit+heatmap+GS map+accuracy)", "基础演示 (一键: 轨道+热力图+地图+准确率)"),
        ])
        if c == "a":
            run(f"python examples/demo_satellites.py --lang {LANG}")
            pause()
        elif c == "b":
            run("python examples/demo_environment.py")
            pause()
        elif c == "c":
            run(f"python examples/demo_satellites.py --lang {LANG}")
            run("python examples/demo_environment.py")
            pause()
        elif c == "d":
            run(f"python _run_demo.py --lang {LANG}")
            pause()
        elif c == "x" or c == "":
            return

def menu_sim():
    while True:
        c = menu("Orbit Environment Tuning / 轨道环境调参", [
            ("a", "Preview current orbit environment", "预览当前轨道环境"),
            ("b", "Edit orbit environment parameters", "编辑轨道环境参数（不立即运行）"),
            ("c", "One-off simulation override", "临时覆盖参数运行模拟"),
            ("d", "View current configuration", "查看当前配置"),
        ])
        if c == "a":
            run("python -m fl_space.cli run simulate")
            pause()
        elif c == "b":
            sats = ask("Satellites", "5")
            gs = ask("Ground Stations", "3")
            hrs = ask("Sim Hours", "3")
            slot = ask("Slot Min", "1.0")
            backend = ask("Backend (kepler/skyfield)", "kepler")
            run(f"python -m fl_space.cli run simulate --sats {sats} --stations {gs} --hours {hrs} --slot-min {slot} --backend {backend}")
            pause()
        elif c == "c":
            run("python -m fl_space.cli run show")
            pause()
        elif c == "x" or c == "":
            return

def menu_viz():
    while True:
        c = menu("Visualization / 可视化工具", [
            ("a", "Mount visualization outputs for the next run", "挂载本次运行要输出的可视化结果"),
            ("b", "Generate charts from an existing result folder", "从已有结果文件夹补生成图表"),
            ("c", "Start 3D orbit server (opens in browser)", "启动3D轨道服务器（浏览器打开）"),
        ])
        if c == "a":
            print("\n  可选结果: accuracy=准确率, time=时间分解, summary=实验摘要")
            print("  输入逗号分隔，例如: accuracy,time,summary")
            plots = ask("本次运行输出哪些图", "accuracy,summary")
            path = os.path.join(PROJECT_DIR, ".fls_visuals.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"plots": plots}, f, ensure_ascii=False, indent=2)
            print(f"  [成功] 已挂载可视化选择: {plots}")
            pause()
        elif c == "b":
            menu_config()
        elif c == "c":
            result = ask("结果 JSON 或结果文件夹路径")
            plots = ask("生成哪些图 (accuracy,time,summary)", "accuracy,time,summary")
            run_feedback([sys.executable, "scripts/generate_result_visuals.py", result, "--plots", plots],
                         "历史结果图表已生成")
            pause()
        elif c == "d":
            port = ask("端口", "8700")
            print(f"  [提示] 服务启动后，请在浏览器打开 http://127.0.0.1:{port}")
            print("  [提示] 保持此窗口运行；按 Ctrl+C 停止服务器。")
            run_feedback([sys.executable, "-m", "fl_space.cli", "run", "serve", "--host", "127.0.0.1", "--port", port],
                         "3D服务器已退出")
            pause()
        elif c == "x" or c == "":
            return

def menu_fl():
    while True:
        c = menu("FL Training / 联邦学习训练", [
            ("a", "FedProxSat Quick Test (adaptive mu)", "FedProxSat 快速测试 (自适应mu)"),
            ("b", "FedAvg Standard Training", "FedAvg 标准训练"),
            ("c", "FedProx Training", "FedProx 训练"),
            ("d", "FedBuff Training", "FedBuff 训练"),
            ("e", "Full Custom Training", "完全自定义训练"),
        ])
        if c == "a":
            gs = ask("Ground Stations", "5")
            rds = ask("Rounds", "300")
            mu = ask("Base mu", "0.01")
            eps = ask("Epochs", "2")
            adapt = ask("Adaptive mu? (y/n)", "y")
            ad_flag = "" if adapt.lower() == "y" else "--no-adaptive"
            if confirm("确认启动 FedProxSat 快速测试"):
                run_feedback(f"python examples/quick_test.py --gs {gs} --rounds {rds} --mu {mu} --epochs {eps} {ad_flag} --lang {LANG} --output {OUTPUT_DIR}/quick_test", "快速测试完成")
                after_experiment()
            pause()
        elif c == "b":
            run("python -m fl_space.cli mount algo fedavg")
            confirm_and_train()
            pause()
        elif c == "c":
            mu = ask("mu value", "0.01")
            run("python -m fl_space.cli mount algo fedprox")
            run(f"python -m fl_space.cli tune mu {mu}")
            confirm_and_train()
            pause()
        elif c == "d":
            buf = ask("Buffer size", "5")
            run("python -m fl_space.cli mount algo fedbuff")
            run(f"python -m fl_space.cli tune buffer-size {buf}")
            confirm_and_train()
            pause()
        elif c == "e":
            algo = ask("Algorithm (fedavg/fedprox/fedbuff)", "fedavg")
            sats = ask("Satellites", "5")
            gs = ask("Ground Stations", "3")
            rds = ask("Rounds", "300")
            lr = ask("Learning Rate", "0.01")
            eps = ask("Local Epochs", "2")
            bs = ask("Batch Size", "32")
            ds = ask("Dataset (mnist/cifar10)", "mnist")
            dev = ask("Device (cpu/cuda)", "cpu")
            run(f"python -m fl_space.cli mount algo {algo}")
            run(f"python -m fl_space.cli mount sats {sats}")
            run(f"python -m fl_space.cli mount stations {gs}")
            for val, cmd in [(rds, "rounds"), (lr, "lr"), (eps, "epochs"), (bs, "batch"), (ds, "dataset"), (dev, "device")]:
                run(f"python -m fl_space.cli tune {cmd} {val}")
            confirm_and_train()
            pause()
        elif c == "x" or c == "":
            return


def confirm_and_train():
    """Show the effective session once, then run the final FL entry point."""
    print("\n  ===== 实验前最终确认 =====")
    run([sys.executable, "-m", "fl_space.cli", "run", "show"])
    if not confirm("以上参数将用于 FL 联邦学习训练，确认启动"):
        print("  [提示] 已取消运行，参数保持不变。")
        return
    output = ask("结果文件路径", os.path.join(OUTPUT_DIR, "fl_result.json"))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    code = run_feedback([sys.executable, "-m", "fl_space.cli", "run", "train", "--output", output], "FL 训练完成")
    if code == 0:
        after_experiment(output)


def after_experiment(result_path=None):
    """Offer post-run visualization/template save and restore defaults."""
    if result_path and os.path.exists(result_path):
        if confirm("是否根据本次结果生成已挂载的图表", "y"):
            plots = "accuracy,summary"
            visual_path = os.path.join(PROJECT_DIR, ".fls_visuals.json")
            if os.path.exists(visual_path):
                with open(visual_path, encoding="utf-8") as f:
                    plots = json.load(f).get("plots", plots)
            run_feedback([sys.executable, "scripts/generate_result_visuals.py", result_path, "--plots", plots], "结果图表生成完成")
    if confirm("是否另存一份当前参数模板"):
        save_template(ask("模板名称"))
    if confirm("实验结束后恢复默认参数", "y"):
        run_feedback([sys.executable, "-m", "fl_space.cli", "reset"], "已恢复默认参数")

def menu_exp():
    while True:
        c = menu("Full Experiment / 完整实验套件", [
            ("a", "Standard Grid [GS=3,5,7,10]x[SAT=3,5,7,10] (16 runs)", "标准网格搜索 16组"),
            ("b", "Small Quick [GS=1,3,5]x[SAT=3,5] (6 runs)", "小规模快速 6组"),
            ("c", "Single Experiment (specify GS+SAT)", "单组实验 (指定GS+SAT)"),
            ("d", "FedProx Suite (hetero orbits)", "FedProx 实验套件"),
            ("e", "Full Custom Grid", "完全自定义网格"),
        ])
        if c == "a":
            print(f"\n  {t('warning_long')}")
            if ask(t("confirm"), "n").lower() == "y":
                run(f"python examples/standard_experiment.py --gs 3 5 7 10 --sats 3 5 7 10 --rounds 300 --lang {LANG} --output {OUTPUT_DIR}/full_grid")
            pause()
        elif c == "b":
            run(f"python examples/standard_experiment.py --gs 1 3 5 --sats 3 5 --rounds 100 --lang {LANG} --output {OUTPUT_DIR}/quick_grid")
            pause()
        elif c == "c":
            gs = ask("Ground Stations", "5")
            sats = ask("Satellites", "7")
            rds = ask("Rounds", "300")
            eps = ask("Epochs", "2")
            run(f"python examples/standard_experiment.py --gs {gs} --sats {sats} --rounds {rds} --epochs {eps} --lang {LANG} --output {OUTPUT_DIR}/single_exp")
            pause()
        elif c == "d":
            gs_list = ask("GS counts (space separated)", "1 3 5")
            rds = ask("Rounds", "300")
            run(f"python examples/run_spacefl_experiment.py --gs-counts {gs_list} --rounds {rds} --lang {LANG} --output {OUTPUT_DIR}/fedprox_suite")
            pause()
        elif c == "e":
            gs_list = ask("GS list (space separated)", "3 5 7")
            sat_list = ask("SAT list (space separated)", "3 5 7")
            rds = ask("Rounds", "300")
            eps = ask("Epochs", "2")
            lr = ask("Learning Rate", "0.01")
            ds = ask("Dataset", "mnist")
            alt = ask("Altitude (km)", "500")
            incl = ask("Inclination (deg)", "53")
            hrs = ask("Sim Hours", "3")
            run(f"python examples/standard_experiment.py --gs {gs_list} --sats {sat_list} --rounds {rds} --epochs {eps} --lr {lr} --dataset {ds} --altitude {alt} --inclination {incl} --sim-hours {hrs} --lang {LANG} --output {OUTPUT_DIR}/custom_exp")
            pause()
        elif c == "x" or c == "":
            return

def menu_tune():
    specs = {
        "a": ("lr", "float", "0.000001~10", "例如 0.01"),
        "b": ("rounds", "int", ">=1", "例如 30"),
        "c": ("epochs", "int", ">=1", "例如 2"),
        "d": ("batch", "int", ">=1", "例如 32"),
        "e": ("mu", "float", ">=0", "例如 0.01"),
        "f": ("seed", "int", "整数", "例如 42"),
        "g": ("dataset", "enum", "mnist|fashion_mnist|cifar10", "输入其中一个名称"),
        "h": ("scale", "enum", "small|medium|large", "输入其中一个名称"),
        "i": ("early-stop", "float", "0~1", "例如 0.90"),
        "j": ("workers", "int", ">=1", "例如 1"),
        "k": ("non-iid", "enum", "on|off", "输入 on 或 off"),
        "l": ("alpha", "float", ">0", "例如 0.5"),
        "m": ("device", "enum", "cpu|cuda", "输入 cpu 或 cuda"),
        "n": ("buffer-size", "int", ">=1", "例如 5"),
    }
    while True:
        c = menu("Tune Params / 调参面板", [
            ("a", "Learning rate [float, 0.000001~10]", "学习率 [浮点数, 0.000001~10]"),
            ("b", "Rounds [int, >=1]", "轮次 [整数, >=1]"),
            ("c", "Local epochs [int, >=1]", "本地Epoch [整数, >=1]"),
            ("d", "Batch size [int, >=1]", "Batch大小 [整数, >=1]"),
            ("e", "FedProx mu [float, >=0]", "FedProx mu [浮点数, >=0]"),
            ("f", "Random seed [int]", "随机种子 [整数]"),
            ("g", "Dataset [enum]", "数据集 [枚举]"),
            ("h", "Scale [small|medium|large]", "规模 [small|medium|large]"),
            ("i", "Early stop [float, 0~1]", "早停阈值 [浮点数, 0~1]"),
            ("j", "Workers [int, >=1]", "训练线程 [整数, >=1]"),
            ("k", "Non-IID [on|off]", "Non-IID [on|off]"),
            ("l", "Dirichlet alpha [float, >0]", "Dirichlet alpha [浮点数, >0]"),
            ("m", "Device [cpu|cuda]", "设备 [cpu|cuda]"),
            ("n", "FedBuff buffer [int, >=1]", "FedBuff缓冲 [整数, >=1]"),
            ("s", "Show Current Values", "查看当前值"),
            ("r", "Reset to Defaults", "恢复默认"),
        ])
        if c in specs:
            key, kind, rule, example = specs[c]
            print(f"\n  参数: {key} | 类型: {kind} | 允许范围: {rule} | 输入示例: {example}")
            val = ask("请输入新值")
            code = run([sys.executable, "-m", "fl_space.cli", "tune", key, val])
            if code == 0:
                print(f"  [成功] {key} 已更新。失败时请检查类型/范围/可选值。")
            else:
                print(f"  [失败] {key} 未更新：输入不符合 {kind} / {rule}，或 CLI 返回了具体错误。")
        elif c == "s":
            run("python -m fl_space.cli tune show")
            pause()
        elif c == "r":
            run("python -m fl_space.cli tune reset")
            print(f"  {t('done')}")
            pause()
        elif c == "x" or c == "":
            return

def menu_config():
    specs = {
        "a": ("algo", "enum", "fedavg|fedprox|fedbuff", "fedavg"),
        "b": ("isl", "enum", "disabled|wgs84", "disabled"),
        "c": ("backend", "enum", "kepler|skyfield", "kepler"),
        "d": ("body", "enum", "earth|mars|moon|jupiter|saturn|venus", "earth"),
        "e": ("sats", "int", ">=1", "5"),
        "f": ("stations", "int", ">=1", "3"),
        "g": ("altitude", "float", ">0 km", "500"),
        "h": ("inclination", "float", "0~180 deg", "53"),
        "i": ("sim-hours", "float", ">0", "3"),
        "j": ("timeslot-min", "float", ">0", "1"),
        "k": ("distribution", "enum", "uniform|walker|cluster", "uniform"),
        "l": ("time-model", "enum", "slot|physics", "slot"),
        "m": ("staleness", "enum", "on|off", "off"),
        "n": ("isl-buffer", "float", ">=0 km", "0"),
    }
    while True:
        c = menu("Config Panel / 配置面板", [
            ("a", "FL Algorithm (fedavg/fedprox/fedbuff)", "FL算法"),
            ("b", "ISL Link (disabled/wgs84)", "ISL星间链路"),
            ("c", "Orbit Backend (kepler/skyfield)", "轨道后端"),
            ("d", "Celestial Body (earth/mars)", "天体选择"),
            ("e", "Number of Satellites", "卫星数量"),
            ("f", "Number of Ground Stations", "地面站数量"),
            ("g", "Orbit Altitude (km)", "轨道高度"),
            ("h", "Inclination (deg)", "轨道倾角"),
            ("i", "Simulation Hours", "模拟时长"),
            ("j", "Timeslot Minutes", "时隙时长"),
            ("k", "Distribution (uniform/walker)", "分布方式"),
            ("l", "Time Model (slot/physics)", "时间模型"),
            ("m", "Staleness (on/off)", "陈旧度加权"),
            ("n", "ISL Buffer (km)", "ISL缓冲高度"),
            ("s", "Show Current Config", "查看当前配置"),
            ("r", "Reset to Defaults", "恢复默认"),
        ])
        if c in specs:
            key, kind, rule, example = specs[c]
            print(f"\n  参数: {key} | 类型: {kind} | 允许范围: {rule} | 输入示例: {example}")
            val = ask("请输入新值")
            code = run([sys.executable, "-m", "fl_space.cli", "mount", key, val])
            if code == 0:
                print(f"  [成功] {key} 已更新。")
            else:
                print(f"  [失败] {key} 未更新：请检查类型、范围或枚举值。")
        elif c == "s":
            run("python -m fl_space.cli mount show")
            pause()
        elif c == "r":
            run("python -m fl_space.cli reset")
            print(f"  {t('done')}")
            pause()
        elif c == "x" or c == "":
            return

def menu_3d():
    while True:
        c = menu("3D Web Server / 3D可视化", [
            ("a", "Start Server (port 8080)", "启动服务器 (端口8080)"),
            ("b", "Custom Port", "自定义端口"),
        ])
        if c == "a":
            print("\n  Starting at http://localhost:8080 (Ctrl+C to stop)\n")
            run("python -m fl_space.cli run serve --port 8080")
        elif c == "b":
            port = ask("Port", "8080")
            print(f"\n  Starting at http://localhost:{port} (Ctrl+C to stop)\n")
            run(f"python -m fl_space.cli run serve --port {port}")
        elif c == "x" or c == "":
            return

def menu_info():
    header()
    print("  ====== SpaceFL System Info ======\n")
    run("python -m fl_space.cli info")
    print(f"\n  Python: ", end="")
    run("python --version")
    print(f"\n  Project Dir: {PROJECT_DIR}")
    print(f"  Output Dir:  {OUTPUT_DIR}")
    if os.path.exists(os.path.join(PROJECT_DIR, ".fls_session.json")):
        print("  Session:     .fls_session.json [exists]")
    else:
        print("  Session:     .fls_session.json [not created]")
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        if files:
            print(f"\n  {t('output_dir')}:")
            for f in sorted(files):
                full = os.path.join(OUTPUT_DIR, f)
                size = os.path.getsize(full)
                print(f"    {f} ({size:,} bytes)")
    pause()

# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════
def main():
    global LANG
    while True:
        cls()
        print(f"""
  ============================================================
    {t('title')}   [{t('lang_label')}]
  ============================================================
    {'1. 快速演示          2. 轨道模拟' if LANG == 'zh' else '1. Quick Demos       2. Orbit Simulation'}
    {'3. 可视化工具        4. FL联邦学习训练' if LANG == 'zh' else '3. Visualization     4. FL Training'}
    {'5. 完整实验套件      6. 调参面板' if LANG == 'zh' else '5. Full Experiment   6. Tune Params'}
    {'7. 配置面板          8. 可视化与3D' if LANG == 'zh' else '7. Config Panel      8. Visualization'}
    {'9. 参数模板          10. 系统信息' if LANG == 'zh' else '9. Templates        10. System Info'}
    {'11. 切换语言' if LANG == 'zh' else '11. Switch Language'}
    {'0. 退出' if LANG == 'zh' else '0. Exit'}
  ============================================================
    {'推荐流程: 9导入模板 -> 6调参 -> 7轨道环境 -> 3挂载图表 -> 4确认并训练' if LANG == 'zh' else 'Flow: 9 import -> 6 tune -> 7 orbit env -> 3 mount plots -> 4 confirm/train'}
""")
        c = input(f"  {t('select')} [0-11]: ").strip()

        if c == "0":
            cls()
            print(f"\n  {t('exit_msg')}\n")
            break
        elif c == "1": menu_demos()
        elif c == "2": menu_sim()
        elif c == "3": menu_viz()
        elif c == "4": menu_fl()
        elif c == "5": menu_exp()
        elif c == "6": menu_tune()
        elif c == "7": menu_config()
        elif c == "8": menu_viz()
        elif c == "9": menu_templates()
        elif c == "10": menu_info()
        elif c == "11":
            LANG = "zh" if LANG == "en" else "en"
        else:
            print(f"  {t('invalid')}")
            import time; time.sleep(0.5)

if __name__ == "__main__":
    main()
