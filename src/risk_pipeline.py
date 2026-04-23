"""
TEP 风险序列生成 Pipeline
从原始数据 → 一维风险序列 → 监督样本 (X, y)

对应论文: Real-time risk prediction of chemical processes based on 
         Attention-based Bi-LSTM (CJChE, 2024)

用法:
    1. 确保已安装依赖: pip install numpy pandas matplotlib scipy PyWavelets
    2. 运行: python src/risk_pipeline.py
    3. 结果保存在 results/ 和 figures/ 文件夹中

如果想用真实数据，修改下面的 DATA_SOURCE 变量即可。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pywt
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# 导入数据加载器
from data_loader import load_tep_data

# ============================
# 0. 全局参数配置（已复核，来自蓝本论文）
# ============================
CONFIG = {
    'variables': 22,           # XMEAS1 ~ XMEAS22
    'window_size': 150,        # 滑动窗口大小（论文确定值）
    'step_size': 1,            # 移动步长（论文确定值）
    'corr_threshold': 0.2,     # 相关分析阈值（论文确定值）
    'lookback': 12,            # 监督样本时间步（论文 Time stride=12）
    'wavelet': 'db4',          # 小波基函数（论文未给，工程默认值）
    'wavelet_level': 3,        # 分解层数（论文未给，工程默认值）
    'train_size': 6000,        # 训练样本数（论文确定值，Fault 11）
    'test_size': 3000,         # 测试样本数（论文确定值，Fault 11）
    'risk_levels': [           # 风险分级阈值（原文 Table 1）
        (0.0, 0.3, 'Low', '低风险'),
        (0.3, 0.6, 'Lower', '较低风险'),
        (0.6, 0.9, 'Higher', '较高风险'),
        (0.9, 1.0, 'High', '高风险'),
    ]
}

# ============================================================
# 【重要】数据来源设置
# ============================================================
# 选项1: 'simulated' —— 使用程序自动生成的模拟数据（不需要外部文件）
# 选项2: 'csv' —— 使用真实 CSV 数据（需要准备 normal.csv, fault4.csv, fault11.csv）
# ============================================================
DATA_SOURCE = 'simulated'  # ← 拿到真实数据后，改成 'csv'

# 如果用 CSV，修改下面的路径指向你的真实数据文件
CSV_PATHS = {
    'normal_path':  'data/normal.csv',
    'fault4_path':  'data/fault4.csv',
    'fault11_path': 'data/fault11.csv',
}

# 设置中文字体（解决图里中文显示方块的问题）
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont('src/SourceHanSansSC-Regular.otf')
    plt.rcParams['font.family'] = 'Source Han Sans SC'
except:
    # 如果加载失败，尝试系统默认中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================
# 1. 小波去噪
# ============================
def wavelet_denoise(data, wavelet='db4', level=3, threshold_mode='soft'):
    """
    对多变量时间序列做小波去噪
    
    大白话：像降噪耳机一样，去掉数据里的杂音
    
    参数:
        data: np.ndarray, shape=(n_samples, n_vars)
        wavelet: 小波基函数名称
        level: 分解层数
        threshold_mode: 'soft' 或 'hard'
    
    返回:
        去噪后的数据，形状与输入相同
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    denoised = np.zeros_like(data)
    
    for i in range(data.shape[1]):
        signal = np.array(data[:, i], copy=True)
        
        # 小波分解：把信号拆成不同频率的成分
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # 计算阈值（Universal Threshold）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # 对细节系数做阈值处理（把小的系数压到0，保留大的）
        new_coeffs = [coeffs[0]]  # 保留近似系数（低频，信号主体）
        for detail in coeffs[1:]:
            new_coeffs.append(pywt.threshold(detail, value=uthresh, mode=threshold_mode))
        
        # 重构：把处理后的成分拼回完整的信号
        denoised[:, i] = pywt.waverec(new_coeffs, wavelet)
    
    if squeeze:
        denoised = denoised.squeeze()
    
    return denoised


# ============================
# 2. 滑动窗口 + 相关分析 + 邻接矩阵
# ============================
def build_adjacency_matrix(window_data, method='spearman', threshold=0.2):
    """
    对单个窗口的数据计算相关矩阵并转为二值邻接矩阵
    
    大白话：看看这150个时间步里，哪两个传感器"步调一致"，
           如果一致程度超过0.2，就在它们之间画一条线
    
    参数:
        window_data: np.ndarray, shape=(window_size, n_vars)
        method: 'spearman' 或 'pearson'
        threshold: 相关系数阈值
    
    返回:
        adj_matrix: np.ndarray, shape=(n_vars, n_vars), 0/1 无向矩阵
    """
    n_vars = window_data.shape[1]
    
    if method == 'spearman':
        corr_matrix, _ = stats.spearmanr(window_data)
    else:
        corr_matrix = np.corrcoef(window_data.T)
    
    # 处理可能的 nan
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # 取绝对值，按阈值转二值矩阵
    adj_matrix = (np.abs(corr_matrix) >= threshold).astype(int)
    
    # 对角线置 0（节点不自连）
    np.fill_diagonal(adj_matrix, 0)
    
    # 确保无向（对称）
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    
    return adj_matrix


def sliding_window_networks(data, window_size=150, step_size=1, 
                            corr_threshold=0.2, method='spearman'):
    """
    对时间序列做滑动窗口，每个窗口生成一个邻接矩阵
    
    大白话：像看股票K线图，每150个点截一张图，
           每张图里看看传感器之间的关联关系
    
    参数:
        data: np.ndarray, shape=(n_samples, n_vars)
    
    返回:
        adj_matrices: list of np.ndarray
        window_indices: list of int
    """
    n_samples, n_vars = data.shape
    adj_matrices = []
    window_indices = []
    
    for start in tqdm(range(0, n_samples - window_size + 1, step_size), desc="构建网络"):
        end = start + window_size
        window_data = data[start:end, :]
        adj = build_adjacency_matrix(window_data, method=method, threshold=corr_threshold)
        adj_matrices.append(adj)
        window_indices.append(start)
    
    return adj_matrices, window_indices


# ============================
# 3. 结构熵计算（节点度 Shannon 熵）
# ============================
def structural_entropy(adj_matrix):
    """
    计算单个网络的结构熵（原文 Eq.1）
    
    大白话：看看这张"传感器关系网"有多"均匀"。
           如果每个传感器的连线数都差不多，就是"均匀"的，熵高；
           如果有的传感器连线特别多、有的特别少，就是"不均匀"的，熵低。
    
    公式: E = -Σ (k_i/Σk_i) · ln(k_i/Σk_i)
      k_i = 第i个节点的连线数（度）
    
    参数:
        adj_matrix: np.ndarray, shape=(n_vars, n_vars), 0/1 无向矩阵
    
    返回:
        float: 结构熵值
    """
    # 节点度：每个传感器连了多少条线
    degrees = adj_matrix.sum(axis=1)
    total_degree = degrees.sum()
    
    if total_degree == 0:
        return 0.0
    
    # 度分布概率：某个传感器的连线数占总连线数的比例
    p = degrees / total_degree
    
    # 避免 log(0)
    p = p[p > 0]
    
    # Shannon 熵：信息论里的"混乱程度"
    entropy = -np.sum(p * np.log(p))
    
    return entropy


def compute_entropy_sequence(adj_matrices):
    """对一系列邻接矩阵计算结构熵序列"""
    return np.array([structural_entropy(adj) for adj in adj_matrices])


# ============================
# 4. 跨工况统一 min-max 归一化
# ============================
def normalize_risk_sequence(entropy_dict, mode='global'):
    """
    对多工况的结构熵序列做归一化
    
    大白话：把不同工况算出来的"混乱程度"缩放到同一个尺度（0~1），
           这样才能比较"谁的风险更高"
    
    关键：论文明确说是"跨工况统一归一化"——
          把 No Fault、Fault 4、Fault 11 的熵全部混在一起，
          找到全局的最大最小值，再统一缩放。
    
    参数:
        entropy_dict: dict, {工况名: 熵序列数组}
        mode: 'global' 跨工况统一（原文方式）, 'local' 各工况独立
    
    返回:
        risk_dict: dict, {工况名: 相对风险序列数组, 范围[0,1]}
    """
    if mode == 'global':
        # 跨工况统一：把所有工况的熵拼在一起求全局 min/max
        all_entropies = np.concatenate(list(entropy_dict.values()))
        e_min = all_entropies.min()
        e_max = all_entropies.max()
    else:
        e_min = None
        e_max = None
    
    risk_dict = {}
    for name, entropy in entropy_dict.items():
        if mode == 'local':
            e_min = entropy.min()
            e_max = entropy.max()
        
        if e_max - e_min < 1e-10:
            risk = np.zeros_like(entropy)
        else:
            risk = (entropy - e_min) / (e_max - e_min)
        
        risk_dict[name] = risk
    
    return risk_dict


# ============================
# 5. 风险分级
# ============================
def risk_level(risk_value, levels=CONFIG['risk_levels']):
    """将单个风险值映射为风险等级"""
    for low, high, en, cn in levels:
        if low < risk_value <= high or (low == 0.0 and risk_value == 0.0):
            return en, cn
    return 'High', '高风险'


def risk_distribution(risk_sequence, levels=CONFIG['risk_levels']):
    """统计风险序列在各等级的频数"""
    dist = {name: 0 for _, _, _, name in levels}
    for rv in risk_sequence:
        _, cn = risk_level(rv, levels)
        dist[cn] += 1
    return dist


# ============================
# 6. 监督样本构造
# ============================
def create_supervised_samples(risk_sequence, lookback=12, step=1):
    """
    从一维风险序列构造监督学习样本 (X, y)
    
    大白话：用过去12个时间点的风险值，预测下一个时间点的风险值
           就像老师给你12道例题，让你预测第13题的答案
    
    参数:
        risk_sequence: np.ndarray, shape=(n_samples,)
        lookback: 输入时间步（过去看几步）
        step: 采样步长
    
    返回:
        X: np.ndarray, shape=(n_samples, lookback)
        y: np.ndarray, shape=(n_samples, 1)
    """
    X, y = [], []
    for i in range(0, len(risk_sequence) - lookback, step):
        X.append(risk_sequence[i:i+lookback])
        y.append(risk_sequence[i+lookback])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return X, y


def split_train_test(X, y, train_size=6000, test_size=3000):
    """
    划分训练集和测试集（按顺序切分，时序数据不打乱）
    
    大白话：前6000条给AI学习，后3000条用来考试
    """
    total = train_size + test_size
    X = X[:total]
    y = y[:total]
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    return X_train, y_train, X_test, y_test


# ============================
# 7. 可视化
# ============================
def plot_risk_sequences(risk_dict, save_path='figures/risk_sequences.png'):
    """绘制多工况风险序列对比图"""
    plt.figure(figsize=(14, 5))
    colors = {'normal': '#2E86AB', 'fault4': '#A23B72', 'fault11': '#F18F01'}
    labels = {'normal': 'No Fault', 'fault4': 'Fault 4', 'fault11': 'Fault 11'}
    
    for name, risk in risk_dict.items():
        plt.plot(risk, label=labels.get(name, name), 
                color=colors.get(name, 'gray'), alpha=0.8, linewidth=0.8)
    
    # 画风险分级虚线
    for low, high, _, cn in CONFIG['risk_levels'][1:]:
        plt.axhline(y=low, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.xlabel('时间窗口索引', fontsize=12)
    plt.ylabel('相对风险值 $R_t$', fontsize=12)
    plt.title('不同工况下的风险序列对比', fontsize=14)
    plt.legend(fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'已保存: {save_path}')


def plot_risk_distribution(risk_dict, save_path='figures/risk_distribution.png'):
    """绘制风险分级分布对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors_bar = ['#2E86AB', '#A23B72', '#F18F01']
    labels = {'normal': 'No Fault', 'fault4': 'Fault 4', 'fault11': 'Fault 11'}
    level_names = ['低风险', '较低风险', '较高风险', '高风险']
    
    for idx, (name, risk) in enumerate(risk_dict.items()):
        dist = risk_distribution(risk)
        values = [dist.get(ln, 0) for ln in level_names]
        
        axes[idx].bar(level_names, values, color=colors_bar[idx], alpha=0.8)
        axes[idx].set_title(labels.get(name, name), fontsize=12)
        axes[idx].set_ylabel('频数', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=15)
    
    plt.suptitle('风险等级分布', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'已保存: {save_path}')


def plot_entropy_raw(entropy_dict, save_path='figures/entropy_raw.png'):
    """绘制原始结构熵序列对比（归一化前）"""
    plt.figure(figsize=(14, 5))
    colors = {'normal': '#2E86AB', 'fault4': '#A23B72', 'fault11': '#F18F01'}
    labels = {'normal': 'No Fault', 'fault4': 'Fault 4', 'fault11': 'Fault 11'}
    
    for name, entropy in entropy_dict.items():
        plt.plot(entropy, label=labels.get(name, name), 
                color=colors.get(name, 'gray'), alpha=0.8, linewidth=0.8)
    
    plt.xlabel('时间窗口索引', fontsize=12)
    plt.ylabel('结构熵 $E$', fontsize=12)
    plt.title('原始结构熵（归一化前）', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'已保存: {save_path}')


# ============================
# 8. 主流程：一键运行
# ============================
def main():
    """
    完整 Pipeline 主函数：
    原始数据 → 小波去噪 → 滑动窗口 → 相关网络 → 结构熵 → 归一化 → 监督样本
    """
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("=" * 60)
    print("TEP 风险序列生成 Pipeline")
    print("=" * 60)
    
    # -------- Step 0: 加载数据 --------
    print("\n[Step 0] 加载数据...")
    
    if DATA_SOURCE == 'simulated':
        print("  使用模拟数据（无需外部文件）")
        df_normal, df_fault4, df_fault11 = load_tep_data('simulated')
    elif DATA_SOURCE == 'csv':
        print("  使用 CSV 真实数据")
        df_normal, df_fault4, df_fault11 = load_tep_data('csv', **CSV_PATHS)
    else:
        raise ValueError(f"未知的 DATA_SOURCE: {DATA_SOURCE}")
    
    print(f"  No Fault: {df_normal.shape}")
    print(f"  Fault 4:  {df_fault4.shape}")
    print(f"  Fault 11: {df_fault11.shape}")
    
    # -------- Step 1: 小波去噪 --------
    print("\n[Step 1] 小波去噪 (db4, level=3)...")
    data_normal = wavelet_denoise(df_normal.values, 
                                   wavelet=CONFIG['wavelet'], 
                                   level=CONFIG['wavelet_level'])
    data_fault4 = wavelet_denoise(df_fault4.values, 
                                   wavelet=CONFIG['wavelet'], 
                                   level=CONFIG['wavelet_level'])
    data_fault11 = wavelet_denoise(df_fault11.values, 
                                    wavelet=CONFIG['wavelet'], 
                                    level=CONFIG['wavelet_level'])
    
    # 对齐长度（小波重构可能多/少几个点）
    min_len = min(data_normal.shape[0], data_fault4.shape[0], data_fault11.shape[0])
    data_normal = data_normal[:min_len, :]
    data_fault4 = data_fault4[:min_len, :]
    data_fault11 = data_fault11[:min_len, :]
    print(f"  去噪后长度: {min_len}")
    
    # -------- Step 2: 滑动窗口 + 邻接矩阵 --------
    print("\n[Step 2] 滑动窗口构建复杂网络...")
    print(f"  window={CONFIG['window_size']}, step={CONFIG['step_size']}, threshold={CONFIG['corr_threshold']}")
    
    adj_normal, idx_normal = sliding_window_networks(
        data_normal, CONFIG['window_size'], CONFIG['step_size'], CONFIG['corr_threshold'])
    adj_fault4, idx_fault4 = sliding_window_networks(
        data_fault4, CONFIG['window_size'], CONFIG['step_size'], CONFIG['corr_threshold'])
    adj_fault11, idx_fault11 = sliding_window_networks(
        data_fault11, CONFIG['window_size'], CONFIG['step_size'], CONFIG['corr_threshold'])
    
    print(f"  No Fault:  {len(adj_normal)} 个窗口")
    print(f"  Fault 4:   {len(adj_fault4)} 个窗口")
    print(f"  Fault 11:  {len(adj_fault11)} 个窗口")
    
    # -------- Step 3: 结构熵 --------
    print("\n[Step 3] 计算结构熵序列...")
    entropy_normal = compute_entropy_sequence(adj_normal)
    entropy_fault4 = compute_entropy_sequence(adj_fault4)
    entropy_fault11 = compute_entropy_sequence(adj_fault11)
    
    entropy_dict = {
        'normal': entropy_normal,
        'fault4': entropy_fault4,
        'fault11': entropy_fault11,
    }
    print(f"  No Fault:  E ∈ [{entropy_normal.min():.3f}, {entropy_normal.max():.3f}]")
    print(f"  Fault 4:   E ∈ [{entropy_fault4.min():.3f}, {entropy_fault4.max():.3f}]")
    print(f"  Fault 11:  E ∈ [{entropy_fault11.min():.3f}, {entropy_fault11.max():.3f}]")
    
    # -------- Step 4: 跨工况统一归一化 --------
    print("\n[Step 4] 跨工况统一 min-max 归一化...")
    risk_dict = normalize_risk_sequence(entropy_dict, mode='global')
    
    print(f"  No Fault:  R ∈ [{risk_dict['normal'].min():.3f}, {risk_dict['normal'].max():.3f}]")
    print(f"  Fault 4:   R ∈ [{risk_dict['fault4'].min():.3f}, {risk_dict['fault4'].max():.3f}]")
    print(f"  Fault 11:  R ∈ [{risk_dict['fault11'].min():.3f}, {risk_dict['fault11'].max():.3f}]")
    
    # -------- Step 5: 风险分级统计 --------
    print("\n[Step 5] 风险分级分布统计...")
    for name in ['normal', 'fault4', 'fault11']:
        dist = risk_distribution(risk_dict[name])
        label = {'normal': 'No Fault', 'fault4': 'Fault 4', 'fault11': 'Fault 11'}[name]
        print(f"  {label}: {dist}")
    
    # -------- Step 6: 可视化 --------
    print("\n[Step 6] 生成可视化图表...")
    plot_entropy_raw(entropy_dict, 'figures/01_entropy_raw.png')
    plot_risk_sequences(risk_dict, 'figures/02_risk_sequences.png')
    plot_risk_distribution(risk_dict, 'figures/03_risk_distribution.png')
    
    # -------- Step 7: 监督样本构造（Fault 11） --------
    print("\n[Step 7] 构造监督学习样本 (Fault 11)...")
    risk_fault11 = risk_dict['fault11']
    X, y = create_supervised_samples(risk_fault11, lookback=CONFIG['lookback'])
    X_train, y_train, X_test, y_test = split_train_test(
        X, y, CONFIG['train_size'], CONFIG['test_size'])
    
    print(f"  总样本: X={X.shape}, y={y.shape}")
    print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
    
    # -------- 保存结果 --------
    print("\n[Step 8] 保存结果...")
    
    # 保存风险序列
    risk_df = pd.DataFrame({
        'No_Fault': risk_dict['normal'],
        'Fault_4': risk_dict['fault4'],
        'Fault_11': risk_dict['fault11'],
    })
    risk_df.to_csv('results/risk_sequences.csv', index=False)
    print("  已保存: results/risk_sequences.csv")
    
    # 保存监督样本
    np.save('results/X_train.npy', X_train)
    np.save('results/y_train.npy', y_train)
    np.save('results/X_test.npy', X_test)
    np.save('results/y_test.npy', y_test)
    print("  已保存: results/X_train.npy, y_train.npy, X_test.npy, y_test.npy")
    
    # 保存配置
    with open('results/config.json', 'w') as f:
        json.dump({k: v for k, v in CONFIG.items() if k != 'risk_levels'}, f, indent=2)
    print("  已保存: results/config.json")
    
    print("\n" + "=" * 60)
    print("[OK] Pipeline 完成！")
    print("  图表在 figures/ 文件夹里")
    print("  数据在 results/ 文件夹里")
    print("  下一步：训练 LSTM / Bi-LSTM / Attention-Bi-LSTM")
    print("=" * 60)
    
    return risk_dict, X_train, y_train, X_test, y_test


if __name__ == '__main__':
    risk_dict, X_train, y_train, X_test, y_test = main()
