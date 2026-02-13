import os
import sqlite3
import threading
import json
import time
from typing import Optional, Dict, List, Set, Tuple, Union
import numpy as np

import tf

TAU = 2 * np.pi  # 2π


class MemoryDB:
    """语义网格数据库，直接存储每个网格及方向的语义特征"""

    def __init__(
        self,
        db_path: str = None,
        renew_db: bool = False,
        grid_size: float = 0.05,
        max_yaws_per_grid: int = 4,
    ):
        self.db_path = db_path if db_path else ":memory:"
        self._lock = threading.Lock()
        if renew_db and self.db_path != ":memory:" and os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._init_database()
        self.grid_size = grid_size  # 网格大小，默认为0.05米
        self.max_yaws_per_grid = max_yaws_per_grid  # 网格方向数量，默认为4个方向

    def _init_database(self):
        """初始化数据库"""
        with self._lock:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,  -- 格式："i_j"
                    i INTEGER,
                    j INTEGER,
                    yaws TEXT,  -- JSON 格式的字典,键为bin编号，值为yaw角度
                    features TEXT  -- JSON 格式的字典,键为bin编号，值为特征列表
                )
            """
            )
            self.conn.commit()

    def _make_key(self, i: int, j: int) -> str:
        """生成网格的唯一键"""
        return f"{i}_{j}"

    def _update_grid(self, i: int, j: int, yaws: dict, features: dict):
        """
        更新网格数据。如果网格已存在，则覆盖其语义特征。
        """
        key = self._make_key(i, j)
        json_yaws = json.dumps(yaws)
        json_features = json.dumps(features)
        with self._lock:
            self.cursor.execute(
                """
                INSERT INTO memory (key, i, j, yaws, features)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET yaws=excluded.yaws, features=excluded.features
                """,
                (key, i, j, json_yaws, json_features),
            )
            self.conn.commit()

    def _get_grid_by_ij(
        self, i: int, j: int
    ) -> Optional[Dict[str, Union[int, dict, dict]]]:
        """
        根据网格ij坐标获取网格数据。
        返回包含i, j, yaws(dict)和features(dict)的字典，如果不存在则返回None。
        """
        key = self._make_key(i, j)
        with self._lock:
            self.cursor.execute(
                "SELECT i, j, yaws, features FROM memory WHERE key = ?", (key,)
            )
            row = self.cursor.fetchone()
            if not row:
                return None
            return {
                "i": row[0],
                "j": row[1],
                "yaws": json.loads(row[2]),
                "features": json.loads(row[3]),
            }

    def _get_all_ijyaw(self) -> List[Tuple[int, int, float]]:
        """
        获取所有网格的ij坐标和yaw。[(i1, j1, yaw1), (i1, j1, yaw2), ...]
        """
        res = []
        with self._lock:
            self.cursor.execute("SELECT i, j, yaws FROM memory")
            rows = self.cursor.fetchall()
            for row in rows:
                i, j, yaws_json = row
                yaws_dict = json.loads(yaws_json)
                for yaw in yaws_dict.values():
                    res.append((i, j, yaw))
        return res

    def _get_all_xyyaw(self) -> List[Tuple[float, float, float]]:
        """
        获取所有网格的xy坐标和yaw。[(x1, y1, yaw1), (x1, y1, yaw2), ...]
        """
        res = []
        with self._lock:
            self.cursor.execute("SELECT i, j, yaws FROM memory")
            rows = self.cursor.fetchall()
            for row in rows:
                i, j, yaws_json = row
                yaws_dict = json.loads(yaws_json)
                for yaw in yaws_dict.values():
                    res.append((i * self.grid_size, j * self.grid_size, yaw))
        return res

    def _pose_params_to_grid(
        self, x: float, y: float, z: float, yaw: float
    ) -> Tuple[int, int, float]:
        """
        将位姿参数转换为网格ij坐标。
        """
        i = int(x / self.grid_size)
        j = int(y / self.grid_size)
        yaw = yaw % TAU  # 确保yaw在0到2π之间
        return i, j, yaw

    def _grid_to_pose_params(self, i: int, j: int, yaw: float) -> List[float]:
        """
        将网格ij坐标转换为世界坐标和yaw。
        """
        return [i * self.grid_size, j * self.grid_size, yaw % TAU]

    def _yaw_to_bin(self, yaw: float) -> int:
        """将yaw角度映射到等分区间的bin编号"""
        bin_width = TAU / self.max_yaws_per_grid
        bin_idx = int((yaw % TAU) // bin_width)
        return bin_idx

    def _single_process(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        features: List[List[float]],
    ) -> Optional[str]:
        """
        单个网格处理函数。
        yaw: 当前方向
        features: 当前方向的所有特征（二维数组）
        """
        try:
            i, j, yaw = self._pose_params_to_grid(x, y, z, yaw)
            bin_idx = self._yaw_to_bin(yaw)
            existed_data = self._get_grid_by_ij(i, j)
            yaws = existed_data["yaws"] if existed_data else {}
            feats = existed_data["features"] if existed_data else {}
            yaws[bin_idx] = yaw
            feats[bin_idx] = features
            self._update_grid(i, j, yaws, feats)
        except Exception as e:
            print(f"Error processing _single_process: {e}")
            return None

    def query_by_ft(
        self, feature: List[float], num: int, mode: str = "xy"
    ) -> List[Dict[str, Union[int, int, float]]]:
        """
        查询时只对yaws/features字典中存在的bin做相似度计算。
        """
        res = []
        query_vector = np.array(feature)
        query_norm = np.linalg.norm(query_vector)

        with self._lock:
            self.cursor.execute("SELECT i, j, yaws, features FROM memory")
            while True:
                rows = self.cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    i, j, json_yaws, json_features = row
                    yaws_dict = json.loads(json_yaws)
                    features_dict = json.loads(json_features)
                    for bin_idx in yaws_dict.keys():
                        yaw = yaws_dict[bin_idx]
                        features_yaw = features_dict[bin_idx]
                        features_yaw = np.array(features_yaw)
                        db_norms = np.linalg.norm(features_yaw, axis=1)
                        similarities = np.dot(features_yaw, query_vector) / (
                            db_norms * query_norm
                        )
                        similarities = np.nan_to_num(similarities)
                        max_similarity = similarities.max()
                        if mode == "xy":
                            res.append(
                                {
                                    "x": i * self.grid_size,
                                    "y": j * self.grid_size,
                                    "yaw": yaw,
                                    "similarity": max_similarity,
                                }
                            )
                        elif mode == "ij":
                            res.append(
                                {
                                    "i": i,
                                    "j": j,
                                    "yaw": yaw,
                                    "similarity": max_similarity,
                                }
                            )
        return sorted(res, key=lambda x: x["similarity"], reverse=True)[:num]
