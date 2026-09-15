"""
fl_space — 太空联邦学习 (Space Federated Learning) 研究框架

模块化、可扩展的开源框架，用于研究简化太空联邦学习中的
轨道环境模拟、卫星通信、FL算法选择等关键问题。

分层架构:
    environment/  — 环境模拟层（天体、大气、地面站）
    orbit/        — 轨道力学层（轨道计算、可见性判断）
    simulator/    — 模拟器层（接触矩阵、主模拟器）
    config/       — 配置层（默认参数、配置加载）
    mission/      — 发射任务、卫星设计、部署与低阶航天物理
    integrations/ — 外部生态适配器（含 SpaceX API 只读适配）
    fl/           — 联邦学习层（可插拔 FL 算法框架）
    viz/          — 可视化层（matplotlib 地图/热力图）
"""

__version__ = "0.1.0"
__author__ = "FL Space Team"
