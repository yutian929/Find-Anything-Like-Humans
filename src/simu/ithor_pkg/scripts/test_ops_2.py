from ai2thor.controller import Controller

# ====== 全局：记住上次选中的索引 ======
_last_selected_index = None

# ------- 工具函数 -------
def get_moveable_objects(controller):
    """
    返回（moveable_objects列表, all_objects字典）
    moveable_objects: ["ObjectName1", "ObjectName2", ...]  # 仅 moveable
    all_objects: { name: {"position": {...}, "rotation": {...}, "objectId": str} }  # 所有 objects
    """
    event = controller.step(action="Pass")
    moveable_objects = []
    all_objects = {}

    for obj in event.metadata["objects"]:
        all_objects[obj["name"]] = {
            "position": obj["position"],
            "rotation": obj["rotation"],
            "objectId": obj.get("objectId"),
        }
        if obj.get("moveable", False):  # 只把 moveable 加入列表
            moveable_objects.append(obj["name"])
    return moveable_objects, all_objects


def print_moveables_with_pose(
    moveable_objects, all_objects, title="可移动物体列表（索引 : 名称 | 位置(x,y,z) | 旋转(x,y,z)）:"
):
    print(title)
    if not moveable_objects:
        print("  <空>")
        return
    for i, name in enumerate(moveable_objects):
        p = all_objects[name]["position"]
        r = all_objects[name]["rotation"]
        print(
            f"{i} : {name} | pos=({p['x']:.3f},{p['y']:.3f},{p['z']:.3f}) | rot=({r['x']:.1f},{r['y']:.1f},{r['z']:.1f})"
        )


def input_with_default(prompt, default_str):
    """带默认值的输入：回车即取默认字符串。"""
    s = input(f"{prompt} [默认: {default_str}]：").strip()
    return default_str if s == "" else s


def parse_vec3_input(prompt, default_vec):
    """
    读取形如 x,y,z 的输入；空输入则返回默认；若解析失败则报错并返回 None
    default_vec: dict {"x":..,"y":..,"z":..}
    """
    default_str = f"{default_vec['x']},{default_vec['y']},{default_vec['z']}"
    raw = input_with_default(prompt, default_str)
    try:
        x, y, z = [float(v) for v in raw.split(",")]
        return {"x": x, "y": y, "z": z}
    except Exception:
        print("输入格式错误，应为形如：0.1, 1.0, -0.2")
        return None


def input_index_with_default(prompt, default_index, max_len):
    """
    读取索引；空输入取默认；检查范围。
    """
    raw = input_with_default(prompt, str(default_index))
    if not raw.isdigit():
        print("索引必须是数字。")
        return None
    idx = int(raw)
    if idx < 0 or idx >= max_len:
        print(f"索引越界，应在 [0, {max_len-1}] 内。")
        return None
    return idx


def list_moveables(controller):
    """仅列出所有可移动物体及其当前位置与旋转。"""
    moveable_objects, all_objects = get_moveable_objects(controller)
    if not moveable_objects:
        print("当前场景没有可移动物体。")
    else:
        print_moveables_with_pose(moveable_objects, all_objects)
    return controller.step(action="Done")


# ------- 原功能：随机传送 -------
def test_teleport(controller):
    event = controller.step(action="Pass")
    print("Initial agent metadata:")
    print(event.metadata["agent"])

    positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    import random

    position = random.choice(positions)
    print("Teleporting to:", position)

    event = controller.step(
        action="Teleport", position=position, rotation={"x": 0, "y": 0, "z": 0}
    )
    print("After teleport:")
    print(event.metadata["agent"])
    return controller.step(action="Done")


# ------- 新版设定物体位置（支持默认回车逻辑） -------
# 在文件顶部放一个全局变量（如果你之前已经有就不用重复定义）
try:
    _last_selected_index
except NameError:
    _last_selected_index = None


def test_set_objects(controller, index=None, position=None, rotation=None):
    """
    保持原代码结构，但修复：
    - SetObjectPoses 仅传 pickupable/moveable 的对象
    - 对所有可动对象全量回写，只有选中的被改动，其余照抄当前 pose
    - 索引/位置/旋转：回车=默认（索引默认沿用上一次；位置/旋转默认用当前值）
    """
    global _last_selected_index

    # 1) 拉一次全量对象
    event = controller.step(action="Pass")

    moveable_objects = []  # 仅 moveable 的名字列表（用于选择）
    # 记录所有对象的当前状态 + 标志位
    all_objects = {}  # name -> dict(...)
    eligible_names = []  # pickupable 或 moveable 的名字，真正会传给 SetObjectPoses

    for obj in event.metadata["objects"]:
        name = obj["name"]
        all_objects[name] = {
            "position": obj["position"],
            "rotation": obj["rotation"],
            "moveable": bool(obj.get("moveable", False)),
            "pickupable": bool(obj.get("pickupable", False)),
            "objectId": obj.get("objectId"),
        }
        if obj.get("moveable", False):
            moveable_objects.append(name)
        # 只有 pickupable/moveable 的才允许被 SetObjectPoses 接受
        if obj.get("moveable", False) or obj.get("pickupable", False):
            eligible_names.append(name)

    if not moveable_objects:
        print("当前场景没有可移动物体。")
        return controller.step(action="Done")

    # 2) 打印可移动物体清单（带当前位置/旋转）
    print("可移动物体列表（索引 : 名称 | 位置(x,y,z) | 旋转(x,y,z)）:")
    for i, name in enumerate(moveable_objects):
        p = all_objects[name]["position"]
        r = all_objects[name]["rotation"]
        print(
            f"{i} : {name} | pos=({p['x']:.3f},{p['y']:.3f},{p['z']:.3f}) | rot=({r['x']:.1f},{r['y']:.1f},{r['z']:.1f})"
        )

    # 3) 选择索引（回车默认：第一次 0，之后沿用上一次）
    if index is None:
        default_index = 0 if _last_selected_index is None else _last_selected_index
        raw = input(f"请输入要移动物体的索引（单个数字） [默认: {default_index}]：").strip()
        idx = default_index if raw == "" else (int(raw) if raw.isdigit() else None)
        if idx is None or idx < 0 or idx >= len(moveable_objects):
            print(f"索引无效，应在 [0, {len(moveable_objects)-1}] 内。")
            return controller.step(action="Done")
        index = idx
    _last_selected_index = index

    target_name = moveable_objects[index]
    cur_pos = all_objects[target_name]["position"]
    cur_rot = all_objects[target_name]["rotation"]
    print(f"已选择：{index} : {target_name}")
    print(
        f"当前位姿 → pos=({cur_pos['x']:.3f},{cur_pos['y']:.3f},{cur_pos['z']:.3f}), "
        f"rot=({cur_rot['x']:.1f},{cur_rot['y']:.1f},{cur_rot['z']:.1f})"
    )

    # 4) 位置/旋转输入（回车默认用当前）
    def _parse_vec3(prompt, default_vec):
        default_str = f"{default_vec['x']},{default_vec['y']},{default_vec['z']}"
        raw = input(f"{prompt} [默认: {default_str}]：").strip()
        if raw == "":
            return default_vec
        raw = raw.replace("，", ",")
        try:
            xs = [float(v.strip()) for v in raw.split(",")]
            if len(xs) != 3:
                raise ValueError
            return {"x": xs[0], "y": xs[1], "z": xs[2]}
        except Exception:
            print("输入格式错误，应为形如：0.1, 1.0, -0.2")
            return None

    if position is None:
        position = _parse_vec3(f"请输入 {target_name} 的目标位置 (x, y, z)", cur_pos)
        if position is None:
            return controller.step(action="Done")
    if rotation is None:
        rotation = _parse_vec3(f"请输入 {target_name} 的目标旋转 (x, y, z，角度)", cur_rot)
        if rotation is None:
            return controller.step(action="Done")

    # 5) 构造 SetObjectPoses：只对 eligible（pickupable/moveable）全量回写
    set_objects_poses = []
    for name in eligible_names:
        pos = all_objects[name]["position"]
        rot = all_objects[name]["rotation"]
        if name == target_name:
            pos = position
            rot = rotation

        item = {
            "objectName": name,  # 一定写 objectName
            "position": {
                "x": float(pos["x"]),
                "y": float(pos["y"]),
                "z": float(pos["z"]),
            },
            "rotation": {
                "x": float(rot["x"]),
                "y": float(rot["y"]),
                "z": float(rot["z"]),
            },
        }
        # 可选：补 objectId（有则更稳）
        oid = all_objects[name].get("objectId")
        if oid:
            item["objectId"] = oid

        set_objects_poses.append(item)

    event = controller.step(action="SetObjectPoses", objectPoses=set_objects_poses)
    if not event.metadata.get("lastActionSuccess", True):
        print("SetObjectPoses 失败：", event.metadata.get("errorMessage", "未知错误"))
    else:
        print(f"已移动 {target_name} 到位置 {position}，旋转 {rotation}。")

    return controller.step(action="Done")


# ------- 键盘控制 -------
def keyboard_ctl(controller):
    key = input("W/A/S/D: ")

    if key == "W":
        event = controller.step(action="MoveAhead")
    elif key == "A":
        event = controller.step(action="RotateLeft", degrees=30)
    elif key == "S":
        event = controller.step(action="MoveBack")
    elif key == "D":
        event = controller.step(action="RotateRight", degrees=30)
    else:  # special keys
        if key == "T":
            event = test_teleport(controller)
        elif key == "L":
            event = list_moveables(controller)  # 列出 + 当前位置/旋转
        elif key == "C":
            event = test_set_objects(controller)  # 索引/位置/旋转支持回车默认
        else:
            print("Invalid key")
            return None
    event = controller.step(action="Done")
    return event


# ------- 主程序 -------
if __name__ == "__main__":
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan1",
        gridSize=0.25,
        snapToGrid=False,
        rotateStepDegrees=90,
        renderDepthImage=True,
        renderInstanceSegmentation=False,
        width=640,
        height=480,
        fieldOfView=60.0,
    )

    while True:
        event = keyboard_ctl(controller)
        if event:
            try:
                print("当前智能体位置：", event.metadata["agent"]["position"])
            except Exception:
                pass
