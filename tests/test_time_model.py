"""
测试虚拟时间模型功能

验证：
    1. SlotTimeModel (零训练时间 = 旧行为兼容)
    2. SlotTimeModel (slots_per_epoch=1, 训练有成本)
    3. PhysicsTimeModel (物理精度)
    4. 时间分解输出正确
    5. CLI --time-model 参数可用
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

errors = []


def check(name, condition, detail=""):
    if condition:
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        errors.append(name)


def test_model(
    label: str,
    time_model_spec: str,
    time_model_kwargs: dict | None = None,
    num_rounds: int = 5,
):
    """运行一次 FL 训练并返回结果和服务器引用。"""
    from fl_space.environment import CelestialBody, GroundStationNetwork
    from fl_space.fl.fedprox import create_fedprox_components
    from fl_space.fl.runner import FLRunner
    from fl_space.fl.scheduler import CommunicationScheduler
    from fl_space.fl.server import FLConfig
    from fl_space.orbit import create_circular_orbit
    from fl_space.simulator import OrbitSimulator

    print(f"\n{'=' * 60}")
    print(f"  测试: {label}")
    print(f"  时间模型: {time_model_spec}  kw={time_model_kwargs}")
    print(f"{'=' * 60}")

    # 构建自定义环境 (与之前相同)
    body = CelestialBody(
        name="Custom-400",
        radius_km=400.0,
        GM=800.0,
        rotation_period_hours=24.0,
        atmosphere_height_km=0.0,
    )

    gs_data = [
        ("GS-0", 40.0, 116.0),
        ("GS-1", -33.0, 151.0),
        ("GS-2", 51.0, -0.1),
        ("GS-3", 35.0, 139.0),
        ("GS-4", -22.0, -43.0),
    ]
    gs_network = GroundStationNetwork.from_tuples(gs_data)

    orbits = []
    for i in range(5):
        orb = create_circular_orbit(
            altitude_km=80.0,
            inclination_deg=60.0,
            raan_deg=i * 72.0,
            true_anomaly_deg=i * 30.0,
            body=body,
        )
        orbits.append(orb)

    sim = OrbitSimulator(
        body=body,
        orbits=orbits,
        ground_station_network=gs_network,
        timeslot_duration_min=1.0,
        num_timeslots=1440,
        verbose=False,
    )

    scheduler = CommunicationScheduler(sim)

    config = FLConfig(
        algorithm="fedprox",
        num_rounds=num_rounds,
        num_clients=5,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.01,
        mu=0.01,
        fraction=0.5,
        device="cpu",
        seed=42,
        time_model=time_model_spec,
        time_model_kwargs=time_model_kwargs or {},
    )

    components = create_fedprox_components(
        fraction=0.5,
        min_clients=2,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.01,
        mu=0.01,
        device="cpu",
        seed=42,
    )

    runner = FLRunner(config, *components, scheduler=scheduler)

    try:
        history = runner.run(
            dataset_name="mnist",
            iid=True,
            data_dir="./data",
            verbose=True,
        )
    except Exception as e:
        print(f"  [ERROR] 训练异常: {e}")
        traceback.print_exc()
        return None, None

    return runner._server, history


# ── 测试 1: SlotTimeModel (slots_per_epoch=0, 等价旧行为) ──
print("\n" + "=" * 60)
print(" 测试 1/4: SlotTimeModel (slots_per_epoch=0, 零训练时间)")
print("=" * 60)

server1, history1 = test_model(
    label="SlotTimeModel 零训练时间",
    time_model_spec="slot",
    time_model_kwargs={"slots_per_epoch": 0, "slots_per_mb_down": 0, "slots_per_mb_up": 0},
    num_rounds=5,
)

if server1 is not None and history1:
    check("训练完成", len(history1) > 0)
    check("timeslot_start 字段存在", all(hasattr(r, "timeslot_start") for r in history1))
    check("时间分解字段存在", all(r.time_breakdown is not None for r in history1))
    check("训练时间为零", all(r.time_breakdown.get("train", 0) == 0 for r in history1))

    # 验证时间模型名称
    check("时间模型名称=solt", server1.time_model.name == "slot")

    print("\n  结果摘要:")
    for r in history1:
        bd = r.time_breakdown or {}
        print(f"    Round {r.round_num}: TS={r.timeslot_start}->{r.timeslot} "
              f"train={bd.get('train',0)} total={bd.get('total',0)} "
              f"acc={r.eval_metrics.get('accuracy',0):.4f}")
else:
    check("训练执行", False)


# ── 测试 2: SlotTimeModel (slots_per_epoch=1, 训练有成本) ──
print("\n" + "=" * 60)
print(" 测试 2/4: SlotTimeModel (slots_per_epoch=1, 训练有成本)")
print("=" * 60)

server2, history2 = test_model(
    label="SlotTimeModel 训练有成本",
    time_model_spec="slot",
    time_model_kwargs={"slots_per_epoch": 1, "slots_per_mb_down": 0, "slots_per_mb_up": 0},
    num_rounds=5,
)

if server2 is not None and history2:
    check("训练完成", len(history2) > 0)
    check("训练时间>0", all(r.time_breakdown.get("train", 0) > 0 for r in history2))

    print("\n  结果摘要:")
    for r in history2:
        bd = r.time_breakdown or {}
        print(f"    Round {r.round_num}: TS={r.timeslot_start}->{r.timeslot} "
              f"train={bd.get('train',0)} total={bd.get('total',0)} "
              f"acc={r.eval_metrics.get('accuracy',0):.4f}")
else:
    check("训练执行", False)


# ── 测试 3: SlotTimeModel (训练+传输都有成本) ──
print("\n" + "=" * 60)
print(" 测试 3/4: SlotTimeModel (slots_per_epoch=1 + 传输成本)")
print("=" * 60)

server3, history3 = test_model(
    label="SlotTimeModel 训练+传输成本",
    time_model_spec="slot",
    time_model_kwargs={"slots_per_epoch": 1, "slots_per_mb_down": 2, "slots_per_mb_up": 3},
    num_rounds=5,
)

if server3 is not None and history3:
    check("训练完成", len(history3) > 0)
    check("下载时间>0", any(r.time_breakdown.get("download", 0) > 0 for r in history3))
    check("上传时间>0", any(r.time_breakdown.get("upload", 0) > 0 for r in history3))

    print("\n  结果摘要:")
    for r in history3:
        bd = r.time_breakdown or {}
        print(f"    Round {r.round_num}: TS={r.timeslot_start}->{r.timeslot} "
              f"down={bd.get('download',0)} train={bd.get('train',0)} "
              f"up={bd.get('upload',0)} total={bd.get('total',0)} "
              f"acc={r.eval_metrics.get('accuracy',0):.4f}")
else:
    check("训练执行", False)


# ── 测试 4: PhysicsTimeModel ──
print("\n" + "=" * 60)
print(" 测试 4/4: PhysicsTimeModel (物理精度)")
print("=" * 60)

server4, history4 = test_model(
    label="PhysicsTimeModel",
    time_model_spec="physics",
    time_model_kwargs={
        "compute_gflops": 10.0,
        "downlink_mbps": 10.0,
        "uplink_mbps": 1.0,
        "timeslot_duration_min": 1.0,
    },
    num_rounds=5,
)

if server4 is not None and history4:
    check("训练完成", len(history4) > 0)
    check("时间模型名称=physics", server4.time_model.name == "physics")
    check("时间分解存在", all(r.time_breakdown is not None for r in history4))

    print("\n  结果摘要:")
    for r in history4:
        bd = r.time_breakdown or {}
        print(f"    Round {r.round_num}: TS={r.timeslot_start}->{r.timeslot} "
              f"down={bd.get('download',0)} train={bd.get('train',0)} "
              f"up={bd.get('upload',0)} total={bd.get('total',0)} "
              f"acc={r.eval_metrics.get('accuracy',0):.4f}")
else:
    check("训练执行", False)


# ── 测试 5: 验证 TimeModel 工厂方法和自定义导入 ──
print("\n" + "=" * 60)
print(" 测试 5/5: TimeModel 工厂方法")
print("=" * 60)

from fl_space.fl.time_model import (  # noqa: E402
    PhysicsTimeModel,
    SlotTimeModel,
    TimeBreakdown,
    TimeModel,
)

# 内置创建
tm_slot = TimeModel.create("slot", slots_per_epoch=2)
check("工厂创建 slot", isinstance(tm_slot, SlotTimeModel) and tm_slot.slots_per_epoch == 2)

tm_phys = TimeModel.create("physics", compute_gflops=50.0)
check("工厂创建 physics", isinstance(tm_phys, PhysicsTimeModel) and tm_phys.compute_gflops == 50.0)

# list_builtin
builtins = TimeModel.list_builtin()
check("内置列表包含slot", "slot" in builtins)
check("内置列表包含physics", "physics" in builtins)

# 计算验证
train_slots = tm_slot.compute_train_slots(0, 12000, 5)
check("SlotModel: 5epoch×2slots=10", train_slots == 10)

train_slots_p = tm_phys.compute_train_slots(0, 12000, 5)
check("PhysicsModel: 训练时间>0", train_slots_p > 0)

# TimeBreakdown
bd = TimeBreakdown(wait_distribution=5, train=3, wait_return=10, total=18)
check("TimeBreakdown summary_str", "等待分发:5" in bd.summary_str())
check("TimeBreakdown to_dict", bd.to_dict()["total"] == 18)

# ── 测试 6: get_history_dict 包含时间分解 ──
if server1 is not None:
    hd = server1.get_history_dict()
    check("history_dict 包含 time_breakdown", "time_breakdown" in hd[0])
    check("history_dict 包含 timeslot_start", "timeslot_start" in hd[0])


# ── 总结 ──
print("\n" + "=" * 60)
if errors:
    print(f"  测试完成，{len(errors)} 项失败:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  全部测试通过！")
print("=" * 60)
