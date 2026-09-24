#!/usr/bin/env python
"""
CLI参数全面测试脚本
测试所有 tune、mount、run 命令的参数可用性
"""

import json
import os
import subprocess
import sys

# 测试结果统计
passed = 0
failed = 0
errors = []


def run_cli(*args) -> tuple[int, str, str]:
    """运行CLI命令并返回结果"""
    cmd = ["python", "-m", "fl_space.cli", *list(args)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result.returncode, result.stdout, result.stderr


def run_command_case(name: str, *args, expect_success=True) -> bool:
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
        errors.append(
            {
                "test": name,
                "args": args,
                "returncode": returncode,
                "stdout": stdout[:200],
                "stderr": stderr[:200],
            }
        )
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
    run_command_case("help", "help")
    run_command_case("info", "info")

    # ========== 2. tune 参数测试 ==========
    print("\n【2. tune 参数测试】")
    run_command_case("tune lr", "tune", "lr", "0.001")
    run_command_case("tune rounds", "tune", "rounds", "500")
    run_command_case("tune epochs", "tune", "epochs", "10")
    run_command_case("tune batch", "tune", "batch", "64")
    run_command_case("tune mu", "tune", "mu", "0.1")
    run_command_case("tune buffer-size", "tune", "buffer-size", "10")
    run_command_case("tune seed", "tune", "seed", "2024")
    run_command_case("tune dataset mnist", "tune", "dataset", "mnist")
    run_command_case("tune dataset cifar10", "tune", "dataset", "cifar10")
    run_command_case("tune dataset femnist", "tune", "dataset", "femnist")
    run_command_case("tune scale small", "tune", "scale", "small")
    run_command_case("tune scale medium", "tune", "scale", "medium")
    run_command_case("tune scale large", "tune", "scale", "large")
    run_command_case("tune early-stop", "tune", "early-stop", "0.95")
    run_command_case("tune workers", "tune", "workers", "4")
    run_command_case("tune data-workers", "tune", "data-workers", "2")
    run_command_case("tune non-iid on", "tune", "non-iid", "on")
    run_command_case("tune non-iid off", "tune", "non-iid", "off")
    run_command_case("tune alpha", "tune", "alpha", "0.3")
    run_command_case("tune device cpu", "tune", "device", "cpu")
    run_command_case("tune device cuda", "tune", "device", "cuda")
    run_command_case("tune classes-per-client", "tune", "classes-per-client", "3")
    run_command_case("tune max-samples", "tune", "max-samples", "500")
    run_command_case("tune partition-strategy iid", "tune", "partition-strategy", "iid")
    run_command_case("tune partition-strategy dirichlet", "tune", "partition-strategy", "dirichlet")
    run_command_case("tune partition-strategy shard", "tune", "partition-strategy", "shard")
    run_command_case(
        "tune partition-strategy probability", "tune", "partition-strategy", "probability"
    )
    run_command_case("tune class-probability", "tune", "class-probability", "0.7")
    run_command_case("tune data-dir", "tune", "data-dir", "./mydata")
    run_command_case(
        "tune preference-mode client_window", "tune", "preference-mode", "client_window"
    )
    run_command_case(
        "tune preference-mode class_balanced", "tune", "preference-mode", "class_balanced"
    )
    run_command_case("tune preferred-clients-per-class", "tune", "preferred-clients-per-class", "2")
    run_command_case("tune sample-cap-strategy preserve", "tune", "sample-cap-strategy", "preserve")
    run_command_case("tune sample-cap-strategy balanced", "tune", "sample-cap-strategy", "balanced")
    run_command_case("tune show", "tune", "show")

    # ========== 3. mount 参数测试 ==========
    print("\n【3. mount 参数测试】")
    run_command_case("mount algo fedavg", "mount", "algo", "fedavg")
    run_command_case("mount algo fedprox", "mount", "algo", "fedprox")
    run_command_case("mount algo fedbuff", "mount", "algo", "fedbuff")
    run_command_case("mount isl disabled", "mount", "isl", "disabled")
    run_command_case("mount isl wgs84", "mount", "isl", "wgs84")
    run_command_case("mount isl-buffer", "mount", "isl-buffer", "80")
    run_command_case("mount isl-step", "mount", "isl-step", "30")
    run_command_case("mount time-model slot", "mount", "time-model", "slot")
    run_command_case("mount time-model physics", "mount", "time-model", "physics")
    run_command_case("mount time-model-args", "mount", "time-model-args", '{"key":"value"}')
    run_command_case("mount backend kepler", "mount", "backend", "kepler")
    run_command_case("mount backend skyfield", "mount", "backend", "skyfield")
    run_command_case("mount body earth", "mount", "body", "earth")
    run_command_case("mount body mars", "mount", "body", "mars")
    run_command_case("mount body moon", "mount", "body", "moon")
    run_command_case("mount body jupiter", "mount", "body", "jupiter")
    run_command_case("mount body saturn", "mount", "body", "saturn")
    run_command_case("mount body venus", "mount", "body", "venus")
    run_command_case("mount distribution uniform", "mount", "distribution", "uniform")
    run_command_case("mount distribution walker", "mount", "distribution", "walker")
    run_command_case("mount distribution cluster", "mount", "distribution", "cluster")
    run_command_case("mount staleness on", "mount", "staleness", "on")
    run_command_case("mount staleness off", "mount", "staleness", "off")
    run_command_case("mount sats", "mount", "sats", "10")
    run_command_case("mount stations", "mount", "stations", "5")
    run_command_case("mount sim-hours", "mount", "sim-hours", "48")
    run_command_case("mount timeslot-min", "mount", "timeslot-min", "2.0")
    run_command_case("mount altitude", "mount", "altitude", "550")
    run_command_case("mount inclination", "mount", "inclination", "45")
    run_command_case("mount show", "mount", "show")

    # ========== 4. run 命令测试 ==========
    print("\n【4. run 命令测试】")
    run_command_case("run show", "run", "show")
    run_command_case("run list presets", "run", "list", "presets")
    run_command_case("run list models", "run", "list", "models")
    run_command_case("run list satellites", "run", "list", "satellites")
    run_command_case("run list experiments", "run", "list", "experiments")

    # ========== 5. 参数边界测试 ==========
    print("\n【5. 参数边界测试】")
    run_command_case("tune lr 负数 (应失败)", "tune", "lr", "-0.01", expect_success=False)
    run_command_case("tune rounds 0 (应失败)", "tune", "rounds", "0", expect_success=False)
    run_command_case("tune alpha 负数 (应失败)", "tune", "alpha", "-0.5", expect_success=False)
    run_command_case(
        "tune early-stop >1 (应失败)", "tune", "early-stop", "1.5", expect_success=False
    )
    run_command_case("mount sats 0 (应失败)", "mount", "sats", "0", expect_success=False)
    run_command_case(
        "mount inclination >180 (应失败)", "mount", "inclination", "200", expect_success=False
    )
    run_command_case("mount algo 无效值 (应失败)", "mount", "algo", "invalid", expect_success=False)
    run_command_case(
        "tune dataset 无效值 (应失败)", "tune", "dataset", "invalid", expect_success=False
    )

    # ========== 6. Session持久化测试 ==========
    print("\n【6. Session持久化测试】")
    global passed, failed, errors

    run_command_case("设置多个参数", "tune", "lr", "0.005")
    run_command_case("设置多个参数", "tune", "rounds", "100")
    run_command_case("设置多个参数", "mount", "algo", "fedprox")
    run_command_case("设置多个参数", "mount", "sats", "20")

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
    run_command_case("tune reset", "tune", "reset")
    run_command_case("mount clear", "mount", "clear")

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
        "tune": {"lr": 0.002, "rounds": 200, "dataset": "cifar10"},
        "mount": {"algo": "fedprox", "sats": 15, "stations": 7},
    }
    with open("test_config.json", "w", encoding="utf-8") as f:
        json.dump(test_config, f, indent=2)

    run_command_case("mount config", "mount", "config", "test_config.json")

    # 验证加载
    with open(".fls_session.json", encoding="utf-8") as f:
        session = json.load(f)
        if (
            session["tune"]["lr"] == 0.002
            and session["tune"]["rounds"] == 200
            and session["mount"]["sats"] == 15
        ):
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
            if "args" in err:
                print(f"   命令: python -m fl_space.cli {' '.join(err['args'])}")
                print(f"   返回码: {err['returncode']}")
                if err["stderr"]:
                    print(f"   错误: {err['stderr']}")

    print("\n" + "=" * 70)

    # 清理session文件
    if os.path.exists(".fls_session.json"):
        os.remove(".fls_session.json")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
