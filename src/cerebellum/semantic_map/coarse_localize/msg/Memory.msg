# Memory.msg
Header header

# 位姿参数，包含x, y, z, qx, qy, qz, qw
float32[7] pose_params  # 扁平化存储

# 每个实例的clip特征，512维浮点数
float32[] semantic_fts  # 扁平化存储
uint16 semantic_ft_dim  # 每个实例特征的维度
