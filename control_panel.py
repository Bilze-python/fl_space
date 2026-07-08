#!/usr/bin/env python3
"""
SpaceFL 中控面板 — 一站式操作控制台
用法: python control_panel.py [--lang en|zh]
"""
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
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

def ask(prompt, default=""):
    if default:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt}: ").strip()

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
        c = menu("Orbit Simulation / 轨道模拟", [
            ("a", "Quick Sim (session defaults)", "快速模拟 (当前session参数)"),
            ("b", "Custom Sim", "自定义参数模拟"),
            ("c", "View Session Status", "查看当前session状态"),
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
            ("a", "Dashboard (map+heatmap+stats)", "一键仪表盘 (地图+热力图+统计)"),
            ("b", "Contact Heatmap", "接触热力图"),
            ("c", "Ground Station Map", "地面站地图"),
            ("d", "Accuracy Curve", "准确率曲线"),
            ("e", "Time Breakdown Chart", "时间分解图"),
            ("f", "Generate All Charts", "全部生成"),
        ])
        if c == "a":
            run(f"python _run_demo.py --lang {LANG}")
            pause()
        elif c == "b":
            sats = ask("Satellites", "5")
            gs = ask("Ground Stations", "3")
            hrs = ask("Sim Hours", "24")
            code = (
                f"import matplotlib; matplotlib.use('Agg');"
                f"from fl_space.environment import *; from fl_space.orbit import *;"
                f"from fl_space.simulator import *; from fl_space.utils.viz import plot_contact_heatmap;"
                f"earth=CelestialBody.earth();"
                f"orbits=[create_circular_orbit(500,53,0,i*360/{sats},earth) for i in range({sats})];"
                f"gss=create_default_network({gs});"
                f"sim=OrbitSimulator(body=earth,orbits=orbits,ground_station_network=gss,num_timeslots=int({hrs}*60),backend='kepler',verbose=False);"
                f"plot_contact_heatmap(sim,'{OUTPUT_DIR}/heatmap.png',lang='{LANG}');"
                f"print('Saved: {OUTPUT_DIR}/heatmap.png')"
            )
            run(f'python -c "{code}"')
            pause()
        elif c == "c":
            gs = ask("Ground Stations", "7")
            code = (
                f"import matplotlib; matplotlib.use('Agg');"
                f"from fl_space.environment import *; from fl_space.orbit import *;"
                f"from fl_space.simulator import *; from fl_space.utils.viz import plot_ground_station_map;"
                f"earth=CelestialBody.earth();"
                f"orbits=[create_circular_orbit(500,53,0,0,earth)];"
                f"gss=create_default_network({gs});"
                f"sim=OrbitSimulator(body=earth,orbits=orbits,ground_station_network=gss,num_timeslots=1440,backend='kepler',verbose=False);"
                f"plot_ground_station_map(sim,'{OUTPUT_DIR}/gs_map.png',lang='{LANG}',show_tracks=True);"
                f"print('Saved: {OUTPUT_DIR}/gs_map.png')"
            )
            run(f'python -c "{code}"')
            pause()
        elif c == "d":
            code = (
                f"import matplotlib; matplotlib.use('Agg');"
                f"from fl_space.utils.viz import plot_accuracy_comparison;"
                f"import numpy as np; np.random.seed(42);"
                f"hist=[{{'round':i,'accuracy':0.1+0.4*np.tanh(i*0.02)+np.random.normal(0,0.02),'timeslot':i*10}} for i in range(100)];"
                f"plot_accuracy_comparison(hist,output_path='{OUTPUT_DIR}/accuracy.png',lang='{LANG}');"
                f"print('Saved: {OUTPUT_DIR}/accuracy.png')"
            )
            run(f'python -c "{code}"')
            pause()
        elif c == "e":
            code = (
                f"import matplotlib; matplotlib.use('Agg');"
                f"from fl_space.utils.viz import plot_time_breakdown;"
                f"import numpy as np; np.random.seed(42);"
                f"hist=[{{'round':i,'time_breakdown':{{'wait_distribution':np.random.exponential(3),'download':np.random.exponential(2),'train':np.random.exponential(5),'wait_return':np.random.exponential(2),'upload':np.random.exponential(1)}}}} for i in range(30)];"
                f"plot_time_breakdown(hist,output_path='{OUTPUT_DIR}/time_breakdown.png',lang='{LANG}');"
                f"print('Saved: {OUTPUT_DIR}/time_breakdown.png')"
            )
            run(f'python -c "{code}"')
            pause()
        elif c == "f":
            print("\n  ====== Generating All Charts ======")
            # Heatmap
            run(f'python -c "import matplotlib; matplotlib.use(\"Agg\"); from fl_space.environment import *; from fl_space.orbit import *; from fl_space.simulator import *; from fl_space.utils.viz import plot_contact_heatmap; earth=CelestialBody.earth(); orbits=[create_circular_orbit(500,53,0,i*72,earth) for i in range(5)]; gss=create_default_network(5); sim=OrbitSimulator(body=earth,orbits=orbits,ground_station_network=gss,num_timeslots=1440,backend=\"kepler\",verbose=False); plot_contact_heatmap(sim,\"{OUTPUT_DIR}/heatmap.png\",lang=\"{LANG}\"); print(\"heatmap.png OK\")"')
            # GS Map
            run(f'python -c "import matplotlib; matplotlib.use(\"Agg\"); from fl_space.environment import *; from fl_space.orbit import *; from fl_space.simulator import *; from fl_space.utils.viz import plot_ground_station_map; earth=CelestialBody.earth(); orbits=[create_circular_orbit(500,53,0,0,earth)]; gss=create_default_network(7); sim=OrbitSimulator(body=earth,orbits=orbits,ground_station_network=gss,num_timeslots=1440,backend=\"kepler\",verbose=False); plot_ground_station_map(sim,\"{OUTPUT_DIR}/gs_map.png\",lang=\"{LANG}\",show_tracks=True); print(\"gs_map.png OK\")"')
            # Accuracy
            run(f'python -c "import matplotlib; matplotlib.use(\"Agg\"); from fl_space.utils.viz import plot_accuracy_comparison; import numpy as np; np.random.seed(42); hist=[{{\"round\":i,\"accuracy\":0.1+0.4*np.tanh(i*0.02)+np.random.normal(0,0.02),\"timeslot\":i*10}} for i in range(100)]; plot_accuracy_comparison(hist,output_path=\"{OUTPUT_DIR}/accuracy.png\",lang=\"{LANG}\"); print(\"accuracy.png OK\")"')
            # Time breakdown
            run(f'python -c "import matplotlib; matplotlib.use(\"Agg\"); from fl_space.utils.viz import plot_time_breakdown; import numpy as np; np.random.seed(42); hist=[{{\"round\":i,\"time_breakdown\":{{\"wait_distribution\":np.random.exponential(3),\"download\":np.random.exponential(2),\"train\":np.random.exponential(5),\"wait_return\":np.random.exponential(2),\"upload\":np.random.exponential(1)}}}} for i in range(30)]; plot_time_breakdown(hist,output_path=\"{OUTPUT_DIR}/time_breakdown.png\",lang=\"{LANG}\"); print(\"time_breakdown.png OK\")"')
            print(f"\n  {t('done')} {t('output_dir')}: {OUTPUT_DIR}")
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
            run(f"python examples/quick_test.py --gs {gs} --rounds {rds} --mu {mu} --epochs {eps} {ad_flag} --lang {LANG} --output {OUTPUT_DIR}/quick_test")
            pause()
        elif c == "b":
            run("python -m fl_space.cli mount algo fedavg")
            run("python -m fl_space.cli run train")
            pause()
        elif c == "c":
            mu = ask("mu value", "0.01")
            run("python -m fl_space.cli mount algo fedprox")
            run(f"python -m fl_space.cli tune mu {mu}")
            run("python -m fl_space.cli run train")
            pause()
        elif c == "d":
            buf = ask("Buffer size", "5")
            run("python -m fl_space.cli mount algo fedbuff")
            run(f"python -m fl_space.cli tune buffer-size {buf}")
            run("python -m fl_space.cli run train")
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
            run("python -m fl_space.cli run train")
            pause()
        elif c == "x" or c == "":
            return

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
    while True:
        c = menu("Tune Params / 调参面板", [
            ("a", "Learning Rate", "学习率"),
            ("b", "Training Rounds", "训练轮次"),
            ("c", "Local Epochs", "本地Epoch"),
            ("d", "Batch Size", "Batch大小"),
            ("e", "mu (FedProx)", "mu参数"),
            ("f", "Random Seed", "随机种子"),
            ("g", "Dataset", "数据集"),
            ("h", "Experiment Scale", "实验规模"),
            ("i", "Early Stop Threshold", "早停阈值"),
            ("j", "Worker Threads", "训练线程"),
            ("k", "Non-IID (on/off)", "Non-IID开关"),
            ("l", "Dirichlet Alpha", "Dirichlet-Alpha"),
            ("m", "Device (cpu/cuda)", "设备选择"),
            ("n", "Buffer Size (FedBuff)", "缓冲大小"),
            ("s", "Show Current Values", "查看当前值"),
            ("r", "Reset to Defaults", "恢复默认"),
        ])
        params = {"a":"lr","b":"rounds","c":"epochs","d":"batch","e":"mu","f":"seed",
                  "g":"dataset","h":"scale","i":"early-stop","j":"workers","k":"non-iid",
                  "l":"alpha","m":"device","n":"buffer-size"}
        if c in params:
            val = ask(f"Value for {params[c]}")
            run(f"python -m fl_space.cli tune {params[c]} {val}")
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
        params = {"a":"algo","b":"isl","c":"backend","d":"body","e":"sats","f":"stations",
                  "g":"altitude","h":"inclination","i":"sim-hours","j":"timeslot-min",
                  "k":"distribution","l":"time-model","m":"staleness","n":"isl-buffer"}
        if c in params:
            val = ask(f"Value for {params[c]}")
            run(f"python -m fl_space.cli mount {params[c]} {val}")
        elif c == "s":
            run("python -m fl_space.cli mount show")
            pause()
        elif c == "r":
            run("python -m fl_space.cli mount clear")
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
    {'7. 配置面板          8. 3D可视化服务器' if LANG == 'zh' else '7. Config Panel      8. 3D Web Server'}
    {'9. 系统信息         10. 切换语言' if LANG == 'zh' else '9. System Info      10. Switch Language'}
    {'0. 退出' if LANG == 'zh' else '0. Exit'}
  ============================================================
""")
        c = input(f"  {t('select')} [0-10]: ").strip()

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
        elif c == "8": menu_3d()
        elif c == "9": menu_info()
        elif c == "10":
            LANG = "zh" if LANG == "en" else "en"
        else:
            print(f"  {t('invalid')}")
            import time; time.sleep(0.5)

if __name__ == "__main__":
    main()
