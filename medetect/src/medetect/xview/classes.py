"""xView データセットのクラス定義。

xView の type_id (非連続整数) を YOLO 用の 0 始まり連番インデックスへマッピングする。
公式 60 クラスセットに準拠。
"""

# xView type_id -> クラス名
XVIEW_TYPE_ID_TO_NAME: dict[int, str] = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck w/Flatbed",
    26: "Truck w/Liquid",
    27: "Crane Truck",
    28: "Railway Vehicle",
    29: "Passenger Car",
    32: "Cargo Car",
    33: "Flat Car",
    34: "Tank car",
    35: "Locomotive",
    36: "Maritime Vessel",
    37: "Motorboat",
    38: "Sailboat",
    40: "Tugboat",
    41: "Barge",
    42: "Fishing Vessel",
    44: "Ferry",
    45: "Yacht",
    47: "Container Ship",
    49: "Oil Tanker",
    50: "Engineering Vehicle",
    51: "Tower Crane",
    52: "Container Crane",
    53: "Reach Stacker",
    54: "Straddle Carrier",
    55: "Mobile Crane",
    56: "Dump Truck",
    57: "Haul Truck",
    59: "Scraper/Tractor",
    60: "Front loader/Bulldozer",
    61: "Excavator",
    62: "Cement Mixer",
    63: "Ground Grader",
    64: "Hut/Tent",
    65: "Shed",
    66: "Building",
    67: "Aircraft Hangar",
    68: "Damaged Building",
    71: "Facility",
    72: "Construction Site",
    73: "Vehicle Lot",
    74: "Helipad",
    76: "Storage Tank",
    77: "Shipping Container Lot",
    79: "Shipping Container",
    83: "Pylon",
    84: "Tower",
    86: "Airplane",
    89: "Motorboat",
    91: "Truck",
    93: "Building",
    94: "Tank",
}

# ソート済み type_id のリスト（YOLO インデックス順）
XVIEW_TYPE_IDS: list[int] = sorted(XVIEW_TYPE_ID_TO_NAME.keys())

# type_id -> YOLO クラスインデックス (0 始まり)
XVIEW_TYPE_ID_TO_INDEX: dict[int, int] = {
    type_id: idx for idx, type_id in enumerate(XVIEW_TYPE_IDS)
}

# YOLO クラスインデックス -> クラス名
XVIEW_CLASS_NAMES: list[str] = [
    XVIEW_TYPE_ID_TO_NAME[type_id] for type_id in XVIEW_TYPE_IDS
]

NUM_CLASSES: int = len(XVIEW_TYPE_IDS)
