#!/usr/bin/env python
"""
CLI参数全面测试脚本
测试所有 tune、mount、run 命令的参数可用性
"""
import json
import os
import subprocess
import sys
from typing import Tuple

# 测试结果统计
passed = 0
failed = 0
errors = []


def run_cli(*args) -> Tuple[int, str, str]:
    """运行CLI命令并返回结果"""
    cmd = ["python", "-m", "fl_space.cli"] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8"
    )
    return result.returncode, result.stdout, result.stderr


def test_command(name: str, *args, expect_success=True) -> bool:
    """测试单个命令"""
    global passed, failed

    returncode, stdout, stderr = run_cli(*args)
    success = (returncode == 0) if expect_success else (returncode != 0)

    if success:
        passed += 1
        print(f"  ✓ {name}")
        return True
    else:
        failed += 1
        print(f"  ✗ {name}")
        errors.append({
            "test": name,
            "args": args,
            "returncode": returncode,
            "stdout": stdout[:200],
            "stderr": stderr[:200]
        })
        return False


def main():
    print("=" * 70)
    print("SpaceFL CLI 参数全面测试")
    print("=" * 70)

    # 清理之前的session文件
    if os.path.exists(".fls_session.json"):
        os.remove(".fls_session.json")

    # ========== 1. 基础命令测试 ==========
    print("\n【1. 基础命令测试】")
    test_command("help", "help")
    test_command("info", "info")

    # ========== 2. tune 参数测试 ==========
    print("\n【2. tune 参数测试】")
    test_command("tune lr", "tune", "lr", "0.001")
    test_command("tune rounds", "tune", "rounds", "500")
    test_command("tune epochs", "tune", "epochs", "10")
    test_command("tune batch", "tune", "batch", "64")
    test_command("tune mu", "tune", "mu", "0.1")
    test_command("tune buffer-size", "tune", "buffer-size", "10")
    test_command("tune seed", "tune", "seed", "2024")
    test_command("tune dataset mnist", "tune", "dataset", "mnist")
    test_command("tune dataset cifar10", "tune", "dataset", "cifar10")
    test_command("tune dataset femnist", "tune", "dataset", "femnist")
    test_command("tune scale small", "tune", "scale", "small")
    test_command("tune scale medium", "tune", "scale", "medium")
    test_command("tune scale large", "tune", "scale", "large")
    test_command("tune early-stop", "tune", "early-stop", "0.95")
    test_command("tune workers", "tune", "workers", "4")
    test_command("tune data-workers", "tune", "data-workers", "2")
    test_command("tune non-iid on", "tune", "non-iid", "on")
    test_command("tune non-iid off", "tune", "non-iid", "off")
    test_command("tune alpha", "tune", "alpha", "0.3")
    test_command("tune device cpu", "tune", "device", "cpu")
    test_command("tune device cuda", "tune", "device", "cuda")
    test_command("tune classes-per-client", "tune", "classes-per-client", "3")
    test_command("tune max-samples", "tune", "max-samples", "500")
    test_command("tune partition-strategy iid", "tune", "partition-strategy", "iid")
    test_command("tune partition-strategy dirichlet", "tune", "partition-strategy", "dirichlet")
    test_command("tune partition-strategy shard", "tune", "partition-strategy", "shard")
    test_command("tune partition-strategy probability", "tune", "partition-strategy", "probability")
    test_command("tune class-probability", "tune", "class-probability", "0.7")
    test_command("tune data-dir", "tune", "data-dir", "./mydata")
    test_command("tune preference-mode client_window", "tune", "preference-mode", "client_window")
    test_command("tune preference-mode class_balanced", "tune", "preference-mode", "class_balanced")
    test_command("tune preferred-clients-per-class", "tune", "preferred-clients-per-class", "2")
    test_command("tune sample-cap-strategy preserve", "tune", "sample-cap-strategy", "preserve")
    test_command("tune sample-cap-strategy balanced", "tune", "sample-cap-strategy", "balanced")
    test_command("tune show", "tune", "show")

    # ========== 3. mount 参数测试 ==========
    print("\n【3. mount 参数测试】")
    test_command("mount algo fedavg", "mount", "algo", "fedavg")
    test_command("mount algo fedprox", "mount", "algo", "fedprox")
    test_command("mount algo fedbuff", "mount", "algo", "fedbuff")
    test_command("mount isl disabled", "mount", "isl", "disabled")
    test_command("mount isl wgs84", "mount", "isl", "wgs84")
    test_command("mount isl-buffer", "mount", "isl-buffer", "80")
    test_command("mount isl-step", "mount", "isl-step", "30")
    test_command("mount time-model slot", "mount", "time-model", "slot")
    test_command("mount time-model physics", "mount", "time-model", "physics")
    test_command("mount time-model-args", "mount", "time-model-args", '{"key":"value"}')
    test_command("mount backend kepler", "mount", "backend", "kepler")
    test_command("mount backend skyfield", "mount", "backend", "skyfield")
    test_command("mount body earth", "mount", "body", "earth")
    test_command("mount body mars", "mount", "body", "mars")
    test_command("mount body moon", "mount", "body", "moon")
    test_command("mount body jupiter", "mount", "body", "jupiter")
    test_command("mount body saturn", "mount", "body", "saturn")
    test_command("mount body venus", "mount", "body", "venus")
    test_command("mount distribution uniform", "mount", "distribution", "uniform")
    test_command("mount distribution walker", "mount", "distribution", "walker")
    test_command("mount distribution cluster", "mount", "distribution", "cluster")
    test_command("mount staleness on", "mount", "staleness", "on")
    test_command("mount staleness off", "mount", "staleness", "off")
    test_command("mount sats", "mount", "sats", "10")
    test_command("mount stations", "mount", "stations", "5")
    test_command("mount sim-hours", "mount", "sim-hours", "48")
    test_command("mount timeslot-min", "mount", "timeslot-min", "2.0")
    test_command("mount altitude", "mount", "altitude", "550")
    test_command("mount inclination", "mount", "inclination", "45")
    test_command("mount show", "mount", "show")

    # ========== 4. run 命令测试 ==========
    print("\n【4. run 命令测试】")
    test_command("run show", "run", "show")
    test_command("run list presets", "run", "list", "presets")
    test_command("run list models", "run", "list", "models")
    test_command("run list satellites", "run", "list", "satellites")
    test_command("run list experiments", "run", "list", "experiments")

    # ========== 5. 参数边界测试 ==========
    print("\n【5. 参数边界测试】")
    test_command("tune lr 负数 (应失败)", "tune", "lr", "-0.01", expect_success=False)
    test_command("tune rounds 0 (应失败)", "tune", "rounds", "0", expect_success=False)
    test_command("tune alpha 负数 (应失败)", "tune", "alpha", "-0.5", expect_success=False)
    test_command("tune early-stop >1 (应失败)", "tune", "early-stop", "1.5", expect_success=False)
    test_command("mount sats 0 (应失败)", "mount", "sats", "0", expect_success=False)
    test_command("mount inclination >180 (应失败)", "mount", "inclination", "200", expect_success=False)
    test_command("mount algo 无效值 (应失败)", "mount", "algo", "invalid", expect_success=False)
    test_command("tune dataset 无效值 (应失败)", "tune", "dataset", "invalid", expect_success=False)

    # ========== 6. Session持久化测试 ==========
    print("\n【6. Session持久化测试】")
    global passed, failed, errors

    test_command("设置多个参数", "tune", "lr", "0.005")
    test_command("设置多个参数", "tune", "rounds", "100")
    test_command("设置多个参数", "mount", "algo", "fedprox")
    test_command("设置多个参数", "mount", "sats", "20")

    # 验证session文件存在
    if os.path.exists(".fls_session.json"):
        with open(".fls_session.json", encoding="utf-8") as f:
            session = json.load(f)
            if session["tune"]["lr"] == 0.005 and session["tune"]["rounds"] == 100:
                passed += 1
                print("  ✓ Session持久化验证")
            else:
                failed += 1
                print("  ✗ Session持久化验证")
                errors.append({"test": "Session持久化", "msg": "参数未正确保存"})
    else:
        failed += 1
        print("  ✗ Session文件不存在")
        errors.append({"test": "Session持久化", "msg": "session文件未创建"})

    # ========== 7. reset 和 clear 测试 ==========
    print("\n【7. 重置功能测试】")
    test_command("tune reset", "tune", "reset")
    test_command("mount clear", "mount", "clear")

    # 验证重置后的值
    with open(".fls_session.json", encoding="utf-8") as f:
        session = json.load(f)
        if session["tune"]["lr"] == 0.01 and session["mount"]["algo"] == "fedavg":
            passed += 1
            print("  ✓ 重置功能验证")
        else:
            failed += 1
            print("  ✗ 重置功能验证")
            errors.append({"test": "重置功能", "msg": "重置后参数不正确"})

    # ========== 8. JSON配置加载测试 ==========
    print("\n【8. JSON配置加载测试】")
    # 创建测试配置文件
    test_config = {
        "tune": {
            "lr": 0.002,
            "rounds": 200,
            "dataset": "cifar10"
        },
        "mount": {
            "algo": "fedprox",
            "sats": 15,
            "stations": 7
        }
    }
    with open("test_config.json", "w", encoding="utf-8") as f:
        json.dump(test_config, f, indent=2)

    test_command("mount config", "mount", "config", "test_config.json")

    # 验证加载
    with open(".fls_session.json", encoding="utf-8") as f:
        session = json.load(f)
        if (session["tune"]["lr"] == 0.002 and
            session["tune"]["rounds"] == 200 and
            session["mount"]["sats"] == 15):
            passed += 1
            print("  ✓ JSON配置加载验证")
        else:
            failed += 1
            print("  ✗ JSON配置加载验证")
            errors.append({"test": "JSON配置加载", "msg": "配置未正确加载"})

    # 清理测试文件
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")

    # ========== 测试结果汇总 ==========
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"总测试数: {passed + failed}")
    print(f"通过: {passed} ✓")
    print(f"失败: {failed} ✗")
    print(f"通过率: {passed / (passed + failed) * 100:.1f}%")

    if errors:
        print("\n失败的测试详情:")
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err['test']}")
            if 'args' in err:
                print(f"   命令: python -m fl_space.cli {' '.join(err['args'])}")
                print(f"   返回码: {err['returncode']}")
                if err['stderr']:
                    print(f"   错误: {err['stderr']}")

    print("\n" + "=" * 70)

    # 清理session文件
    if os.path.exists(".fls_session.json"):
        os.remove(".fls_session.json")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
