import rospy
from std_msgs.msg import String
import sys
import termios
import tty
import select


def getKey():
    """获取按键输入"""
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ""
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def keyboard_controller():
    """键盘控制节点主函数"""
    rospy.init_node("robothor_keyboard_controller", anonymous=True)
    key_pub = rospy.Publisher("/ithor/keyboard_cmd", String, queue_size=10)

    rospy.loginfo("键盘控制节点已启动")
    rospy.loginfo("按键控制:")
    rospy.loginfo("W: 前进")
    rospy.loginfo("A: 左转")
    rospy.loginfo("S: 后退")
    rospy.loginfo("D: 右转")

    try:
        while not rospy.is_shutdown():
            key = getKey()
            if key:
                # 如果按了Ctrl+C，结束程序
                if ord(key) == 3:
                    rospy.loginfo("正在退出")
                    break

                key_msg = String()
                key_msg.data = key
                key_pub.publish(key_msg)
                # rospy.loginfo(f"发送按键: {key}")

                # 添加短暂延迟，避免命令过快发送
                rospy.sleep(0.2)
    except Exception as e:
        rospy.logerr(f"键盘控制错误: {e}")
    finally:
        # 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))


if __name__ == "__main__":
    keyboard_controller()
