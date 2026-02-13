from ai2thor.controller import Controller


def test_teleport(Controller):
    # Initial step
    event = controller.step(action="Pass")
    print("Initial agent metadata:")
    print(event.metadata["agent"])

    # Teleport to a new position and rotation
    positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    import random

    position = random.choice(positions)
    print("Teleporting to:", position)
    event = controller.step(
        action="Teleport", position=position, rotation={"x": 0, "y": 0, "z": 0}
    )

    print("After teleport:")
    print(event.metadata["agent"])

    controller.stop()


def test_set_objects(controller, position=None, rotation=None):
    # Initial step
    event = controller.step(action="Pass")
    # print("Initial event.metadata['objects']:")
    # print(event.metadata["objects"])
    print("Moveable objects:")
    moveable_objects = []
    all_objects = {}
    for obj in event.metadata["objects"]:
        all_objects[obj["name"]] = {
            "position": obj["position"],
            "rotation": obj["rotation"],
        }
        if obj["moveable"]:
            moveable_objects.append(obj["name"])
    print(moveable_objects)
    # {'name': 'Box_38ff2bc7', 'position': {'x': -0.47411420941352844, 'y': 1.0353775024414062, 'z': -0.7133176922798157},
    # 'rotation': {'x': 359.9937438964844, 'y': 270.0023498535156, 'z': 359.991455078125}, 'visible': True, 'isInteractable': True,
    # 'receptacle': True, 'toggleable': False, 'isToggled': False, 'breakable': False, 'isBroken': False, 'canFillWithLiquid': False,
    # 'isFilledWithLiquid': False, 'fillLiquid': None, 'dirtyable': False, 'isDirty': False, 'canBeUsedUp': False, 'isUsedUp': False,
    # 'cookable': False, 'isCooked': False, 'temperature': 'RoomTemp', 'isHeatSource': False, 'isColdSource': False, 'sliceable': False,
    # 'isSliced': False, 'openable': True, 'isOpen': True, 'openness': 1.0, 'pickupable': True, 'isPickedUp': False, 'moveable': False,
    # 'mass': 0.30000001192092896, 'salientMaterials': ['Paper'], 'receptacleObjectIds': [], 'distance': 1.423142671585083, 'objectType': 'Box',
    # 'objectId': 'Box|-00.47|+01.04|-00.71', 'assetId': 'Box_12', 'parentReceptacles': ['TVStand|-00.29|00.00|-00.77'], 'controlledObjects': None,
    # 'isMoving': False, 'axisAlignedBoundingBox': {'cornerPoints': [[...]...], 'center': {'x': -0.35312336683273315, 'y': 0.9564105272293091, 'z': -0.708641529083252},
    # 'size': {'x': 0.45968449115753174, 'y': 0.32325267791748047, 'z': 0.43987250328063965}}, 'objectOrientedBoundingBox': {'cornerPoints': [[...]...]}}
    if not position:
        position = input(f"Enter position for {moveable_objects[0]} (x, y, z): ")
        position = [float(coord) for coord in position.split(",")]
        position = {"x": position[0], "y": position[1], "z": position[2]}
    if not rotation:
        rotation = input(f"Enter rotation for {moveable_objects[0]} (x, y, z): ")
        rotation = [float(coord) for coord in rotation.split(",")]
        rotation = {"x": rotation[0], "y": rotation[1], "z": rotation[2]}
    all_objects[moveable_objects[0]]["position"] = position
    all_objects[moveable_objects[0]]["rotation"] = rotation
    set_objects_poses = []
    for name, p_r in all_objects.items():
        set_objects_poses.append(
            {
                "objectName": name,
                "rotation": {
                    "y": p_r["rotation"]["y"],
                    "x": p_r["rotation"]["x"],
                    "z": p_r["rotation"]["z"],
                },
                "position": {
                    "y": p_r["position"]["y"],
                    "x": p_r["position"]["x"],
                    "z": p_r["position"]["z"],
                },
            }
        )

    event = controller.step(action="SetObjectPoses", objectPoses=set_objects_poses)
    event = controller.step(action="Done")


def keyboard_ctl(controller):
    key = input("W/A/S/D: ")
    if key == "W":
        controller.step(action="MoveAhead")
    elif key == "A":
        controller.step(action="RotateLeft", degrees=30)
    elif key == "S":
        controller.step(action="MoveBack")
    elif key == "D":
        controller.step(action="RotateRight", degrees=30)
    else:  # special keys
        if key == "T":
            test_teleport(controller)
        elif key == "C":
            test_set_objects(controller)
        else:
            print("Invalid key")
            return None
    event = controller.step(action="Done")
    return event


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
    # test_teleport(controller)
    # test_set_objects(controller)
    while True:
        event = keyboard_ctl(controller)
        if event:
            print("Current agent position:", event.metadata["agent"]["position"])
