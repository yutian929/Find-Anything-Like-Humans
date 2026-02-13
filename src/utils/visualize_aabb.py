import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random


class AABBVisualizer:
    def __init__(self):
        """初始化AABB可视化器"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.boxes = []  # 存储所有框的信息 [(min_point, max_point, color), ...]
        self.setup_plot()

    def setup_plot(self):
        """设置绘图参数"""
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Vis AABB")
        # 设置初始视角
        self.ax.view_init(elev=30, azim=45)

    def draw_box(self, min_point, max_point, color=None):
        """绘制单个AABB框"""
        if color is None:
            # 随机生成一个颜色
            color = (random.random(), random.random(), random.random(), 0.5)

        # 提取坐标
        x_min, y_min, z_min = min_point
        x_max, y_max, z_max = max_point

        # 定义8个顶点
        vertices = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]

        # 定义6个面，每个面由4个顶点索引组成
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        ]

        # 创建3D多边形集合
        poly = Poly3DCollection(faces, alpha=0.5)
        poly.set_facecolor(color)
        poly.set_edgecolor("black")

        # 添加到当前图形
        self.ax.add_collection3d(poly)

        # 存储框信息
        self.boxes.append((min_point, max_point, color))

        # 更新坐标轴范围
        self.update_plot_limits()

    def update_plot_limits(self):
        """更新坐标轴的范围"""
        all_points = []
        for min_p, max_p, _ in self.boxes:
            all_points.extend([min_p, max_p])

        if all_points:
            # 转换为numpy数组，便于计算
            points_array = np.array(all_points)

            # 找到所有点的最小和最大值，增加一点缓冲
            min_vals = np.min(points_array, axis=0) - 0.1
            max_vals = np.max(points_array, axis=0) + 0.1

            # 设置轴的范围
            self.ax.set_xlim([min_vals[0], max_vals[0]])
            self.ax.set_ylim([min_vals[1], max_vals[1]])
            self.ax.set_zlim([min_vals[2], max_vals[2]])

    def redraw(self):
        """重新绘制所有框"""
        self.ax.clear()
        self.setup_plot()

        for min_point, max_point, color in self.boxes:
            self.draw_box(min_point, max_point, color)

        plt.draw()

    def run(self):
        """运行可视化循环"""
        plt.ion()  # 打开交互模式
        plt.show()

        print("Vis AABB")
        print("输入格式: x_min y_min z_min x_max y_max z_max")
        print("输入'q'退出，输入'c'清除所有框")

        while True:
            user_input = input("输入AABB坐标 >> ")

            if user_input.lower() == "q":
                break

            if user_input.lower() == "c":
                self.boxes = []
                self.redraw()
                continue

            try:
                # 解析输入的坐标
                coords = [float(x) for x in user_input.split()]
                if len(coords) != 6:
                    print("错误: 需要输入6个数值 (x_min y_min z_min x_max y_max z_max)")
                    continue

                min_point = coords[:3]
                max_point = coords[3:]

                # 验证min是否小于max
                if not all(
                    min_val <= max_val for min_val, max_val in zip(min_point, max_point)
                ):
                    print("警告: 最小值应小于对应的最大值")

                # 添加并绘制新框
                self.draw_box(min_point, max_point)
                plt.pause(0.1)  # 更新显示

            except ValueError:
                print("错误: 输入必须是数字")
            except Exception as e:
                print(f"错误: {e}")

        plt.ioff()
        plt.close()


if __name__ == "__main__":
    visualizer = AABBVisualizer()
    visualizer.run()
