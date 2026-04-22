"""
数据加载适配器
支持多种格式的 TEP 数据输入：
  - 模拟数据（内置生成器，无需外部文件）
  - .csv 文件（最推荐，最容易处理）
  - .mat 文件（MATLAB 格式，需要 scipy）
  - .dat 文件（文本格式，常见于华盛顿大学原始数据）

用法（最简单的方式）：
    from data_loader import load_tep_data
    df_normal, df_fault4, df_fault11 = load_tep_data(source='simulated')

如果拿到了真实数据：
    df_normal, df_fault4, df_fault11 = load_tep_data(
        source='csv',
        normal_path='data/normal.csv',
        fault4_path='data/fault4.csv',
        fault11_path='data/fault11.csv'
    )
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# TEP 标准变量名（论文只用到前22个）
XMEAS_COLS = [f'XMEAS{i}' for i in range(1, 42)]  # XMEAS1 ~ XMEAS41
XMV_COLS = [f'XMV{i}' for i in range(1, 12)]     # XMV1 ~ XMV11
ALL_COLS = XMEAS_COLS + XMV_COLS  # 共52个变量

# 论文使用的22个变量
PAPER_COLS = [f'XMEAS{i}' for i in range(1, 23)]


def generate_simulated_data(n_samples=12000, seed_base=42):
    """
    生成模拟 TEP 数据（当没有真实数据时使用）
    
    模拟数据的设计思路：
    - 正常工况：变量间有稳定的模块化相关结构
    - Fault 4：阶跃变化破坏部分相关结构
    - Fault 11：随机扰动让相关结构混乱
    
    参数:
        n_samples: 每个工况生成多少个点
        seed_base: 随机种子基础值
    
    返回:
        (df_normal, df_fault4, df_fault11): 三个 DataFrame
    """
    np.random.seed(seed_base)
    n_vars = 22
    
    def make_correlated_block(n, block_size, strength):
        """生成一个相关变量块"""
        cov = np.eye(block_size)
        for i in range(block_size):
            for j in range(i+1, block_size):
                cov[i, j] = cov[j, i] = strength * (0.5 + 0.5*np.random.rand())
        return np.random.multivariate_normal(np.zeros(block_size), cov, size=n)
    
    def make_mode(mode_name):
        data = np.zeros((n_samples, n_vars))
        
        # 正常：4个模块，各自内部相关
        data[:, 0:6] = make_correlated_block(n_samples, 6, 0.7)
        data[:, 6:12] = make_correlated_block(n_samples, 6, 0.5)
        data[:, 12:18] = make_correlated_block(n_samples, 6, 0.3)
        data[:, 18:22] = make_correlated_block(n_samples, 4, 0.8)
        
        # 全局弱耦合
        data += np.random.normal(0, 0.1, (n_samples, n_vars))
        
        # 基础值
        base = np.array([0.5, 3664, 4509, 9.2, 26.9, 42.3,
                         2.5, 0.5, 0.5, 0.5, 47.5, 47.5,
                         41.0, 18.0, 50.5, 2300, 0.1, 0.1,
                         0.1, 0.1, 94.0, 77.0])
        data += base
        
        fault_start = int(n_samples * 0.1)
        
        if mode_name == 'fault4':
            # 破坏 Group A 内部相关 + Group A/D 产生新耦合
            data[fault_start:, 0:6] += np.random.normal(0, 2.0, (n_samples-fault_start, 6))
            shared = np.random.normal(0, 1.5, n_samples-fault_start)
            for i in range(6):
                data[fault_start:, i] += shared * 0.5
            for i in range(18, 22):
                data[fault_start:, i] += shared * 0.8 + np.random.normal(5, 1, n_samples-fault_start)
        
        elif mode_name == 'fault11':
            # 全局时变扰动，打乱所有相关结构
            for t in range(fault_start, n_samples):
                phase = np.sin(0.01 * (t - fault_start))
                data[t, :] += np.random.normal(phase * 3.0, 1.5, n_vars)
            # 随机引入负相关
            for i, j in [(0, 15), (3, 9), (7, 18), (11, 20)]:
                shared = np.random.normal(0, 2.0, n_samples-fault_start)
                data[fault_start:, i] += shared
                data[fault_start:, j] -= shared * 0.5
        
        return pd.DataFrame(data, columns=PAPER_COLS)
    
    df_normal = make_mode('normal')
    df_fault4 = make_mode('fault4')
    df_fault11 = make_mode('fault11')
    
    return df_normal, df_fault4, df_fault11


def load_csv_data(normal_path, fault4_path, fault11_path):
    """
    从 CSV 文件加载 TEP 数据
    
    要求：
      - CSV 文件包含至少 XMEAS1 ~ XMEAS22 这22列
      - 列名可以是 XMEAS1, XMEAS2, ... 或 xmeas_1, xmeas_2, ...
    
    参数:
        normal_path: 正常工况 CSV 文件路径
        fault4_path: Fault 4 CSV 文件路径
        fault11_path: Fault 11 CSV 文件路径
    
    返回:
        (df_normal, df_fault4, df_fault11)
    """
    print(f"[DataLoader] 从 CSV 加载数据...")
    
    df_normal = pd.read_csv(normal_path)
    df_fault4 = pd.read_csv(fault4_path)
    df_fault11 = pd.read_csv(fault11_path)
    
    # 标准化列名：把 xmeas_1 变成 XMEAS1
    for df in [df_normal, df_fault4, df_fault11]:
        df.columns = [c.strip().replace('xmeas_', 'XMEAS').replace('XMEAS_', 'XMEAS').upper() for c in df.columns]
    
    # 检查是否有论文需要的22个变量
    available = set(df_normal.columns)
    needed = set(PAPER_COLS)
    missing = needed - available
    
    if missing:
        print(f"  ⚠️ 警告：缺少以下变量: {sorted(missing)}")
        print(f"  可用变量: {sorted(available)[:10]}...")
        raise ValueError(f"CSV 文件缺少必要的变量: {missing}")
    
    # 只保留论文需要的22个变量
    df_normal = df_normal[PAPER_COLS]
    df_fault4 = df_fault4[PAPER_COLS]
    df_fault11 = df_fault11[PAPER_COLS]
    
    print(f"  No Fault: {df_normal.shape}")
    print(f"  Fault 4:  {df_fault4.shape}")
    print(f"  Fault 11: {df_fault11.shape}")
    
    return df_normal, df_fault4, df_fault11


def load_mat_data(filepath, fault_number=0):
    """
    尝试从 MathWorks 格式的 .mat 文件加载 TEP 数据
    
    注意：MATLAB table 格式较复杂，此函数尽力解析，
          如果失败，建议先用 MATLAB 导出为 CSV。
    
    参数:
        filepath: .mat 文件路径
        fault_number: 要提取的故障编号（0=正常）
    
    返回:
        DataFrame，只包含 XMEAS1~XMEAS22
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("请先安装 scipy: pip install scipy")
    
    print(f"[DataLoader] 尝试读取 MAT 文件: {filepath}")
    m = loadmat(filepath)
    
    # MathWorks TEP 数据的常见结构
    # 有时数据直接是数组，有时是 table
    
    # 尝试直接找数值数组
    for key in m.keys():
        if key.startswith('__'):
            continue
        val = m[key]
        if isinstance(val, np.ndarray) and val.dtype in [np.float64, np.float32, np.int32]:
            if len(val.shape) == 2 and val.shape[1] >= 22:
                print(f"  找到数组 '{key}': shape={val.shape}")
                df = pd.DataFrame(val[:, :22], columns=PAPER_COLS)
                return df
    
    # 如果是 table 结构，尝试解析
    # （这部分可能因 MATLAB 版本不同而变化）
    print("  ⚠️ 标准数组未找到，尝试解析 MATLAB table 结构...")
    print("  如果解析失败，建议用 MATLAB 将数据导出为 CSV 格式。")
    
    raise NotImplementedError(
        "MAT 文件格式较复杂，自动解析失败。\n"
        "建议方法1：用 MATLAB 打开文件，执行 writetable(your_table, 'data.csv') 导出为 CSV\n"
        "建议方法2：找老师要 CSV 格式的数据"
    )


def load_tep_data(source='simulated', **kwargs):
    """
    统一的 TEP 数据加载入口
    
    参数:
        source: 'simulated' | 'csv' | 'mat'
        **kwargs: 根据 source 不同传入不同参数
    
    返回:
        (df_normal, df_fault4, df_fault11): 三个 DataFrame，shape=(n_samples, 22)
    
    示例:
        # 使用模拟数据（不需要任何外部文件）
        df_normal, df_fault4, df_fault11 = load_tep_data('simulated')
        
        # 从 CSV 加载（最推荐）
        df_normal, df_fault4, df_fault11 = load_tep_data(
            'csv',
            normal_path='data/normal.csv',
            fault4_path='data/fault4.csv',
            fault11_path='data/fault11.csv'
        )
    """
    if source == 'simulated':
        return generate_simulated_data(**kwargs)
    
    elif source == 'csv':
        return load_csv_data(
            kwargs['normal_path'],
            kwargs['fault4_path'],
            kwargs['fault11_path']
        )
    
    elif source == 'mat':
        # 通常 .mat 文件包含多个工况，需要分别提取
        raise NotImplementedError(
            "MAT 文件加载需要知道具体文件结构。\n"
            "请先运行 'python src/inspect_mat.py data/yourfile.mat' 查看文件内容，\n"
            "然后联系会编程的同学或老师帮忙提取。"
        )
    
    else:
        raise ValueError(f"未知的 source: {source}，可选: 'simulated', 'csv', 'mat'")


if __name__ == '__main__':
    # 简单测试
    print("=" * 50)
    print("数据加载器测试")
    print("=" * 50)
    
    print("\n--- 测试模拟数据 ---")
    df_n, df_f4, df_f11 = load_tep_data('simulated')
    print(f"正常工况前5行:\n{df_n.head()}")
    print(f"\nFault 4 前5行:\n{df_f4.head()}")
