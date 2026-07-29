"""50轮 FedAvg + 10卫星 + 8地面站 快速测试"""
from fl_space.cli import main

main(["tune", "reset"])
main(["mount", "clear"])
main(["tune", "rounds", "50"])
main(["mount", "sats", "10"])
main(["mount", "stations", "8"])
main(["mount", "algo", "fedavg"])
print("\n=== 开始训练 ===")
main(["run", "train"])
