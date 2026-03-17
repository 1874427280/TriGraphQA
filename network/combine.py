import torch


def compute_monomer_aggregation(feat_monomer, edge_index, coords, seq_ids):
    """
    纯 GPU 张量运算，百倍提速：
    严格提取 dst 的特征，根据计算出的距离与序列差异权重，累加给 src 节点
    """
    src, dst = edge_index[0], edge_index[1]

    # 1. 计算序列差异权重
    seq_diff = torch.abs(seq_ids[src] - seq_ids[dst]).float()
    seq_diff[seq_diff == 0] = 1.0

    # 2. 计算空间距离权重
    dist = torch.norm(coords[src] - coords[dst], dim=1)
    weights = torch.log(seq_diff) / (dist + 1e-6)

    # 3. 消息生成与 Scatter 并行累加
    msg = weights.unsqueeze(1) * feat_monomer[dst]
    agg_feat = torch.zeros_like(feat_monomer)
    agg_feat.scatter_add_(0, src.unsqueeze(1).expand_as(msg), msg)

    return agg_feat


def combine_module_monomers(
        idg_string_pairs,
        final_idg_features,
        feat_A, edge_A, map_A_safe, coords_A, seqs_A,  # 直接接收单体图自带的坐标和序列张量
        feat_B, edge_B, map_B_safe, coords_B, seqs_B
):
    """
    整合模块主入口：光速合并交界面图和单体图特征，零 I/O 阻塞。
    """
    device = final_idg_features.device

    # 保证节点特征方向正确 [N, D]
    if feat_A.dim() == 2 and feat_A.size(0) != coords_A.size(0):
        feat_A = feat_A.T
    if feat_B.dim() == 2 and feat_B.size(0) != coords_B.size(0):
        feat_B = feat_B.T

    # 1. 对两条链分别进行内部全图特征聚合
    agg_A = compute_monomer_aggregation(feat_A, edge_A, coords_A.to(device), seqs_A.to(device))
    agg_B = compute_monomer_aggregation(feat_B, edge_B, coords_B.to(device), seqs_B.to(device))

    idg_indices_A, idg_indices_B = [], []

    # 2. 高速查字典，找出发生接触的残基在单体图特征矩阵中的索引
    for key_1, key_2 in idg_string_pairs:
        idx_a = map_A_safe.get(key_1, map_A_safe.get(key_2, -1))
        idx_b = map_B_safe.get(key_2, map_B_safe.get(key_1, -1))
        if idx_a != -1 and idx_b != -1:
            idg_indices_A.append(idx_a)
            idg_indices_B.append(idx_b)

    D_idg = final_idg_features.size(1)
    D_res = feat_A.size(1)

    # 兜底保护：如果字典里没查到任何接触对
    if len(idg_indices_A) == 0:
        return torch.zeros((1, D_idg + 2 * D_res), device=device)

    idx_A_tensor = torch.tensor(idg_indices_A, dtype=torch.long, device=device)
    idx_B_tensor = torch.tensor(idg_indices_B, dtype=torch.long, device=device)

    # 3. 切片提取对应位置的特征，并进行最终的通道维度拼接 (Concat)
    actual_len = min(final_idg_features.size(0), idx_A_tensor.size(0))
    feat_i = agg_A[idx_A_tensor[:actual_len]]
    feat_j = agg_B[idx_B_tensor[:actual_len]]

    return torch.cat([final_idg_features[:actual_len], feat_i, feat_j], dim=1)