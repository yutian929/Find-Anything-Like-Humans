import os
import sqlite3
import threading
import json
import time
from typing import Optional, Dict, List, Set, Tuple, Union
import numpy as np


class PresetDB:
    """预设数据库，用于存储和管理预设信息"""

    def __init__(
        self,
        db_path: str = None,
        renew_db: bool = False,
    ):
        self.db_path = db_path if db_path else ":memory:"
        self._lock = threading.Lock()
        if renew_db and self.db_path != ":memory:" and os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        with self._lock:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS preset (
                    text TEXT PRIMARY KEY,
                    semantic_ft_text TEXT,  -- 文本的CLIP特征
                    semantic_ft_img TEXT  -- 图像的CLIP特征
                )
            """
            )
            self.conn.commit()

    def _update_entry(
        self, text: str, semantic_ft_text: List[float], semantic_ft_img: List[float]
    ):
        """
        更新预设数据。如果预设已存在，则覆盖其特征。
        """
        json_semantic_ft_text = json.dumps(semantic_ft_text)
        json_semantic_ft_img = json.dumps(semantic_ft_img)
        with self._lock:
            self.cursor.execute(
                """
                INSERT INTO preset (text, semantic_ft_text, semantic_ft_img)
                VALUES (?, ?, ?)
                ON CONFLICT(text) DO UPDATE SET semantic_ft_text=excluded.semantic_ft_text, semantic_ft_img=excluded.semantic_ft_img
                """,
                (text, json_semantic_ft_text, json_semantic_ft_img),
            )
            self.conn.commit()

    def query_by_semantic_text_ft(
        self, semantic_text_ft: List[float], num: int
    ) -> List[Dict]:
        """
        根据文本特征查询预设。
        """
        res = []
        semantic_text_ft = np.array(semantic_text_ft)
        semantic_text_ft_norm = np.linalg.norm(semantic_text_ft)
        # breakpoint()
        with self._lock:
            self.cursor.execute(
                "SELECT text, semantic_ft_text, semantic_ft_img FROM preset"
            )
            while True:
                rows = self.cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    text, json_semantic_ft_text, json_semantic_ft_img = row
                    semantic_ft_text = json.loads(json_semantic_ft_text)
                    semantic_ft_img = json.loads(json_semantic_ft_img)
                    similarity = np.dot(semantic_text_ft, semantic_ft_text) / (
                        semantic_text_ft_norm * np.linalg.norm(semantic_ft_text)
                    )

                    res.append(
                        {
                            "text": text,
                            "similarity": similarity,
                            "semantic_ft_text": semantic_ft_text,
                            "semantic_ft_img": semantic_ft_img,
                        }
                    )
        return sorted(res, key=lambda x: x["similarity"], reverse=True)[:num]
