"""独立通信仿真层；不依赖 FL、训练框架或网络服务。"""

from .engine import DynamicCommunication
from .geometry import line_of_sight, orbit_position_provider, sample_geometry
from .interference import (
    AntiJammingConfig,
    InterferenceBurst,
    InterferenceEnvironment,
    interference_from_link,
)
from .models import (
    CommunicationEvent,
    GeometrySample,
    LinkState,
    Message,
    RFConfig,
    TransferResult,
)
from .packet_network import FlowMessage, PacketNetwork, PacketNetworkConfig, PacketNetworkResult
from .rf_channel import evaluate_link
from .scheduling import PacketScheduler

__all__ = [
    "AntiJammingConfig",
    "CommunicationEvent",
    "DynamicCommunication",
    "FlowMessage",
    "GeometrySample",
    "InterferenceBurst",
    "InterferenceEnvironment",
    "LinkState",
    "Message",
    "PacketNetwork",
    "PacketNetworkConfig",
    "PacketNetworkResult",
    "PacketScheduler",
    "RFConfig",
    "TransferResult",
    "evaluate_link",
    "interference_from_link",
    "line_of_sight",
    "orbit_position_provider",
    "sample_geometry",
]
