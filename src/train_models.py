"""
神经网络训练脚本：LSTM / Bi-LSTM / Attention-Bi-LSTM

用法:
    1. 确保已运行过 risk_pipeline.py 生成训练数据
    2. 运行: python src/train_models.py
       或者: python src/train_models.py --config config/default.json
    3. 结果保存在 results/ 和 figures/ 文件夹中

依赖:
    pip install torch numpy pandas matplotlib
"""

import argparse
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from config_loader import PROJECT_ROOT, load_project_config, resolve_project_path

# 尝试加载中文字体
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(PROJECT_ROOT / 'src' / 'SourceHanSansSC-Regular.otf'))
    plt.rcParams['font.family'] = 'Source Han Sans SC'
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 0. 检查 PyTorch 是否可用
# ============================================================
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    print(f"PyTorch 版本: {torch.__version__}")
except ImportError:
    print("=" * 60)
    print("[ERROR] 错误：没有安装 PyTorch")
    print("=" * 60)
    print("请运行以下命令安装:")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("")
    print("说明:")
    print("  - 如果电脑没有 NVIDIA 显卡，用上面的命令装 CPU 版")
    print("  - 如果有显卡且装了 CUDA，可以装 GPU 版（训练更快）:")
    print("    pip install torch")
    exit(1)


# ============================================================
# 1. 加载训练数据
# ============================================================
def load_data(results_dir='results', val_ratio=0.1, batch_size=32):
    """从 risk_pipeline.py 生成的 .npy 文件加载数据"""
    print("\n[Step 0] 加载训练数据...")
    
    try:
        X_train = np.load(os.path.join(results_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(results_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(results_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(results_dir, 'y_test.npy'))
    except FileNotFoundError:
        print("[ERROR] 找不到训练数据文件！")
        print("   请先运行: python src/risk_pipeline.py")
        exit(1)
    
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio 必须在 0 和 1 之间，当前值为: {val_ratio}")

    val_size = max(1, int(len(X_train) * val_ratio))
    if val_size >= len(X_train):
        val_size = len(X_train) - 1

    if val_size <= 0:
        raise ValueError("训练集样本数量不足，无法切分验证集。")

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证集: X={X_val.shape}, y={y_val.shape}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
    
    # 转换为 PyTorch 张量
    # 输入形状: (batch, seq_len, features)
    # 我们每条样本是12个时间步的1维序列，所以 features=1
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (6000, 12, 1)
    y_train_t = torch.FloatTensor(y_train)                # (6000, 1)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)    # (3000, 12, 1)
    y_test_t = torch.FloatTensor(y_test)                  # (3000, 1)
    
    # 构建 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_test_t, y_test_t


# ============================================================
# 2. 定义三种模型
# ============================================================

class LSTMModel(nn.Module):
    """
    基础 LSTM 模型
    
    大白话：LSTM 就像一个有"记忆力"的神经网络，
           它能记住之前看到的数据，用来预测下一步。
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x 形状: (batch, seq_len, input_size)
        # lstm_out 形状: (batch, seq_len, hidden_size)
        # hidden 形状: (num_layers, batch, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 映射到输出
        output = self.fc(last_hidden)     # (batch, output_size)
        return output


class BiLSTMModel(nn.Module):
    """
    双向 LSTM（Bi-LSTM）
    
    大白话：两个 LSTM 背靠背，一个从左往右看数据，一个从右往左看数据。
           这样能同时利用"过去"和"未来"的信息。
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向 LSTM（bidirectional=True）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 双向的输出是 2*hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        output = self.fc(last_hidden)
        return output


class AttentionBiLSTMModel(nn.Module):
    """
    加性注意力 + 双向 LSTM（Attention-Bi-LSTM）
    
    大白话：Bi-LSTM 生成了每个时间步的"理解"，
           Attention 就像一个"聚光灯"，自动找到对当前预测最重要的那个时间步。
    
    注意力机制（加性 Attention，原文 Eq.4）：
        score = v^T · tanh(W_h · h + W_s · s)
        其中 h 是所有时间步的隐藏状态，s 是前一步的输出
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, attention_dim=64, output_size=1):
        super(AttentionBiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 加性 Attention 参数
        self.W_h = nn.Linear(hidden_size * 2, attention_dim)  # 把隐藏状态映射到 attention_dim
        self.W_s = nn.Linear(hidden_size * 2, attention_dim)  # 把"前一步输出"映射到 attention_dim
        self.v = nn.Linear(attention_dim, 1, bias=False)       # 计算注意力得分
        
        # 输出层
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def attention(self, lstm_output, last_hidden):
        """
        加性注意力计算
        
        参数:
            lstm_output: (batch, seq_len, hidden*2)  所有时间步的隐藏状态
            last_hidden: (batch, hidden*2)            最后一个时间步的隐藏状态
        
        返回:
            context: (batch, hidden*2)  加权后的上下文向量
            weights: (batch, seq_len)   注意力权重（可用于可视化）
        """
        # 把 last_hidden 扩展到和 lstm_output 一样的 seq_len
        # last_hidden: (batch, 1, hidden*2)
        last_hidden_expanded = last_hidden.unsqueeze(1).expand(-1, lstm_output.size(1), -1)
        
        # 计算能量得分: tanh(W_h·h + W_s·s)
        energy = torch.tanh(self.W_h(lstm_output) + self.W_s(last_hidden_expanded))
        
        # 计算注意力权重: softmax(v^T · energy)
        attention_weights = torch.softmax(self.v(energy).squeeze(-1), dim=1)
        # attention_weights: (batch, seq_len)
        
        # 加权求和: context = Σ(weight_i * h_i)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context: (batch, hidden*2)
        
        return context, attention_weights
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        last_hidden = lstm_out[:, -1, :]      # (batch, hidden*2)
        
        # 注意力加权
        context, attention_weights = self.attention(lstm_out, last_hidden)
        
        # 输出预测值
        output = self.fc(context)             # (batch, 1)
        return output, attention_weights


# ============================================================
# 3. 训练函数
# ============================================================
def train_model(model, train_loader, val_loader, model_name, epochs=50, lr=1e-3, device='cpu', patience=10):
    """
    训练一个模型
    
    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model_name: 模型名称（用于打印）
        epochs: 训练轮数（论文用500，这里默认50先跑通）
        lr: 学习率
        device: 'cpu' 或 'cuda'
    
    返回:
        训练好的模型
    """
    batch_size = train_loader.batch_size or 32
    print(f"\n[训练] {model_name}")
    print(f"  参数: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    model = model.to(device)
    criterion = nn.MSELoss()           # 损失函数：均方误差（论文用的 MSE）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器：Adam
    
    loss_history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            if model_name == 'Attention-Bi-LSTM':
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history['train'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                if model_name == 'Attention-Bi-LSTM':
                    val_outputs, _ = model(val_x)
                else:
                    val_outputs = model(val_x)

                val_loss += criterion(val_outputs, val_y).item()

        avg_val_loss = val_loss / len(val_loader)
        loss_history['val'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        if epochs_without_improvement >= patience:
            print(f"  [INFO] 验证集损失连续 {patience} 轮未改善，提前停止训练。")
            break
    
    model.load_state_dict(best_state_dict)
    print(f"  [OK] {model_name} 训练完成")
    return model, loss_history


# ============================================================
# 4. 评估函数
# ============================================================
def evaluate_model(model, X_test, y_test, model_name, device='cpu'):
    """
    在测试集上评估模型，计算 RMSE、ARGE、R²
    
    参数:
        model: 训练好的模型
        X_test: 测试输入
        y_test: 测试真实值
        model_name: 模型名称
    
    返回:
        dict: {'RMSE': ..., 'ARGE': ..., 'R2': ..., 'y_pred': ...}
    """
    model.eval()
    X_test = X_test.to(device)
    y_test_np = y_test.numpy().flatten()
    
    with torch.no_grad():
        if model_name == 'Attention-Bi-LSTM':
            y_pred, _ = model(X_test)
        else:
            y_pred = model(X_test)
    
    y_pred_np = y_pred.cpu().numpy().flatten()
    
    # RMSE: 均方根误差（越小越好）
    rmse = np.sqrt(np.mean((y_test_np - y_pred_np) ** 2))
    
    # ARGE: 平均相对误差（越小越好）
    # 注意避免除以0
    arge = np.mean(np.abs((y_test_np - y_pred_np) / (y_test_np + 1e-8)))
    
    # R²: 决定系数（越接近1越好）
    ss_res = np.sum((y_test_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    print(f"\n  [{model_name}] 测试结果:")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    ARGE: {arge:.6f}")
    print(f"    R2:   {r2:.6f}")
    
    return {
        'RMSE': rmse,
        'ARGE': arge,
        'R2': r2,
        'y_pred': y_pred_np
    }


# ============================================================
# 5. 可视化
# ============================================================
def plot_predictions(y_test, results_dict, save_path='figures/04_prediction_comparison.png'):
    """
    绘制三种模型的预测对比图
    
    参数:
        y_test: 真实值
        results_dict: {'LSTM': {'y_pred': ...}, 'Bi-LSTM': ..., 'Attention-Bi-LSTM': ...}
    """
    y_true = y_test.numpy().flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 只画前500个点，太多看不清
    n_show = min(500, len(y_true), *(len(result['y_pred']) for result in results_dict.values()))
    t = np.arange(n_show)
    
    # 图1: LSTM
    axes[0, 0].plot(t, y_true[:n_show], label='真实值', color='#2E86AB', linewidth=1)
    axes[0, 0].plot(t, results_dict['LSTM']['y_pred'][:n_show], label='预测值', color='#E85D04', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('LSTM', fontsize=12)
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('风险值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2: Bi-LSTM
    axes[0, 1].plot(t, y_true[:n_show], label='真实值', color='#2E86AB', linewidth=1)
    axes[0, 1].plot(t, results_dict['Bi-LSTM']['y_pred'][:n_show], label='预测值', color='#A23B72', linewidth=1, alpha=0.8)
    axes[0, 1].set_title('Bi-LSTM', fontsize=12)
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('风险值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3: Attention-Bi-LSTM
    axes[1, 0].plot(t, y_true[:n_show], label='真实值', color='#2E86AB', linewidth=1)
    axes[1, 0].plot(t, results_dict['Attention-Bi-LSTM']['y_pred'][:n_show], label='预测值', color='#F18F01', linewidth=1, alpha=0.8)
    axes[1, 0].set_title('Attention-Bi-LSTM', fontsize=12)
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('风险值')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 图4: 三模型叠加对比
    axes[1, 1].plot(t, y_true[:n_show], label='真实值', color='black', linewidth=1.5)
    axes[1, 1].plot(t, results_dict['LSTM']['y_pred'][:n_show], label='LSTM', color='#E85D04', linewidth=0.8, alpha=0.7)
    axes[1, 1].plot(t, results_dict['Bi-LSTM']['y_pred'][:n_show], label='Bi-LSTM', color='#A23B72', linewidth=0.8, alpha=0.7)
    axes[1, 1].plot(t, results_dict['Attention-Bi-LSTM']['y_pred'][:n_show], label='Attention-Bi-LSTM', color='#F18F01', linewidth=0.8, alpha=0.7)
    axes[1, 1].set_title('三模型叠加对比', fontsize=12)
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('风险值')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('模型预测结果对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n已保存预测对比图: {save_path}")


def plot_metrics_bar(results_dict, save_path='figures/05_metrics_comparison.png'):
    """绘制三种模型的指标对比柱状图"""
    models = list(results_dict.keys())
    rmse_vals = [results_dict[m]['RMSE'] for m in models]
    arge_vals = [results_dict[m]['ARGE'] for m in models]
    r2_vals = [results_dict[m]['R2'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#E85D04', '#A23B72', '#F18F01']
    
    # RMSE（越小越好）
    axes[0].bar(models, rmse_vals, color=colors, alpha=0.8)
    axes[0].set_title('RMSE (越低越好)', fontsize=12)
    axes[0].set_ylabel('RMSE')
    for i, v in enumerate(rmse_vals):
        axes[0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # ARGE（越小越好）
    axes[1].bar(models, arge_vals, color=colors, alpha=0.8)
    axes[1].set_title('ARGE (越低越好)', fontsize=12)
    axes[1].set_ylabel('ARGE')
    for i, v in enumerate(arge_vals):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # R²（越接近1越好）
    axes[2].bar(models, r2_vals, color=colors, alpha=0.8)
    axes[2].set_title(r'$R^2$ (越接近1越好)', fontsize=12)
    axes[2].set_ylabel(r'$R^2$')
    for i, v in enumerate(r2_vals):
        axes[2].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('三模型评价指标对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存指标对比图: {save_path}")


# ============================================================
# 6. 主函数
# ============================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='神经网络训练脚本')
    parser.add_argument(
        '--config',
        default=None,
        help='配置文件路径，默认使用 config/default.json'
    )
    return parser.parse_args()


def main(config_path=None):
    config, resolved_config_path = load_project_config(config_path)
    training_config = config['training']
    output_config = config['output']

    results_dir = str(resolve_project_path(output_config['results_dir']))
    figures_dir = str(resolve_project_path(output_config['figures_dir']))

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config['seed'])

    print("=" * 60)
    print("神经网络训练脚本")
    print("模型: LSTM / Bi-LSTM / Attention-Bi-LSTM")
    print("=" * 60)
    print(f"[INFO] 使用配置文件: {resolved_config_path}")
    print(f"\n[WARN] 提示：当前 epochs={training_config['epochs']}。")
    print("    论文用的 epochs=500，若算力充足可在配置文件里修改。")
    print("    CPU 上跑 50 epoch 大约需要 2~5 分钟。")
    
    # 设置设备
    requested_device = training_config.get('device', 'auto')
    if requested_device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif requested_device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] 配置要求使用 CUDA，但当前不可用，自动回退到 CPU。")
        device = 'cpu'
    else:
        device = requested_device

    print(f"\n使用设备: {device}")
    if device == 'cpu':
        print("（提示：如果有 NVIDIA 显卡且装了 CUDA，训练会快很多）")
    
    # 加载数据
    train_loader, val_loader, X_test, y_test = load_data(
        results_dir=results_dir,
        val_ratio=training_config['val_ratio'],
        batch_size=training_config['batch_size']
    )
    
    # 获取参数
    input_size = X_test.shape[2]   # 1
    
    epochs = training_config['epochs']
    hidden_size = training_config['hidden_size']
    num_layers = training_config['num_layers']
    attention_dim = training_config['attention_dim']
    learning_rate = training_config['learning_rate']
    patience = training_config['patience']
    
    # -------- 训练 LSTM --------
    model_lstm = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model_lstm, loss_lstm = train_model(
        model_lstm,
        train_loader,
        val_loader,
        'LSTM',
        epochs=epochs,
        lr=learning_rate,
        device=device,
        patience=patience
    )
    result_lstm = evaluate_model(model_lstm, X_test, y_test, 'LSTM', device=device)
    
    # -------- 训练 Bi-LSTM --------
    model_bilstm = BiLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model_bilstm, loss_bilstm = train_model(
        model_bilstm,
        train_loader,
        val_loader,
        'Bi-LSTM',
        epochs=epochs,
        lr=learning_rate,
        device=device,
        patience=patience
    )
    result_bilstm = evaluate_model(model_bilstm, X_test, y_test, 'Bi-LSTM', device=device)
    
    # -------- 训练 Attention-Bi-LSTM --------
    model_att = AttentionBiLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attention_dim=attention_dim
    )
    model_att, loss_att = train_model(
        model_att,
        train_loader,
        val_loader,
        'Attention-Bi-LSTM',
        epochs=epochs,
        lr=learning_rate,
        device=device,
        patience=patience
    )
    result_att = evaluate_model(model_att, X_test, y_test, 'Attention-Bi-LSTM', device=device)
    
    # -------- 汇总结果 --------
    results_dict = {
        'LSTM': result_lstm,
        'Bi-LSTM': result_bilstm,
        'Attention-Bi-LSTM': result_att,
    }
    
    print("\n" + "=" * 60)
    print("[INFO] 三模型指标汇总")
    print("=" * 60)
    summary_df = pd.DataFrame({
        'LSTM': [result_lstm['RMSE'], result_lstm['ARGE'], result_lstm['R2']],
        'Bi-LSTM': [result_bilstm['RMSE'], result_bilstm['ARGE'], result_bilstm['R2']],
        'Attention-Bi-LSTM': [result_att['RMSE'], result_att['ARGE'], result_att['R2']],
    }, index=['RMSE', 'ARGE', 'R2'])
    print(summary_df.to_string())
    
    # -------- 保存结果 --------
    summary_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'))
    print(f"\n已保存: {os.path.join(output_config['results_dir'], 'model_comparison.csv')}")
    
    # 保存每个模型的预测值
    np.save(os.path.join(results_dir, 'y_pred_lstm.npy'), result_lstm['y_pred'])
    np.save(os.path.join(results_dir, 'y_pred_bilstm.npy'), result_bilstm['y_pred'])
    np.save(os.path.join(results_dir, 'y_pred_attention.npy'), result_att['y_pred'])
    print(f"已保存预测结果: {os.path.join(output_config['results_dir'], 'y_pred_*.npy')}")

    # 保存模型权重
    if output_config.get('save_weights', True):
        torch.save(model_lstm.state_dict(), os.path.join(results_dir, 'model_lstm.pt'))
        torch.save(model_bilstm.state_dict(), os.path.join(results_dir, 'model_bilstm.pt'))
        torch.save(model_att.state_dict(), os.path.join(results_dir, 'model_attention.pt'))
        print(f"已保存模型权重: {os.path.join(output_config['results_dir'], 'model_*.pt')}")

    with open(os.path.join(results_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"已保存训练配置: {os.path.join(output_config['results_dir'], 'train_config.json')}")
    
    # -------- 可视化 --------
    print("\n[可视化] 生成图表...")
    plot_predictions(y_test, results_dict, save_path=os.path.join(figures_dir, '04_prediction_comparison.png'))
    plot_metrics_bar(results_dict, save_path=os.path.join(figures_dir, '05_metrics_comparison.png'))
    
    print("\n" + "=" * 60)
    print("[OK] 全部完成！")
    print(f"  图表在 {output_config['figures_dir']}/ 文件夹里")
    print(f"  结果在 {output_config['results_dir']}/ 文件夹里")
    print("=" * 60)
    
    return results_dict


if __name__ == '__main__':
    args = parse_args()
    main(config_path=args.config)
