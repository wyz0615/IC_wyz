# %% SVM 分类分析 - 可选择是否区分兴奋性和抑制性RR神经元
"""
目标：
1. 可选择使用所有RR神经元或区分兴奋/抑制性RR神经元
2. 使用SVM进行二分类或四分类分析
3. 评估分类性能并比较不同策略的效果
"""

import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import json
from scipy import ndimage
import re
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')

# %% 开关配置 - 在这里设置分析模式
SEPARATE_EXC_INH = True  # True: 区分兴奋性和抑制性RR神经元; False: 不区分，使用所有RR神经元
USE_PCA = True  # 是否使用PCA降维
N_FRAMES = 2  # 选择连续的时间帧数量
PCA_COMPONENTS = 20  # PCA降维的维度，当USE_PCA=True时生效
BINARY_CLASSIFICATION = False  # True: 二分类(IC vs LC); False: 四分类(IC2, IC4, LC2, LC4)
RANDOM_STATE = 7  # 随机种子，确保结果可重复
SHUFFLE_DATA = True  # 是否在训练前打乱数据

# %% 配置
class ExpConfig:
    def __init__(self, file_path=None):
        if file_path is not None:
            try:
                self.load_config(file_path)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self.set_default_config()
        else:
            self.set_default_config()
        self.preprocess_cfg = {
            'preprocess': True,
            'win_size': 150
        }

    def load_config(self, file_path):
        if not file_path.endswith('.json'):
            raise NotImplementedError("目前仅支持JSON格式的配置文件")
        import json
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        required_keys = ['DATA_PATH']
        missing = [k for k in required_keys if k not in config_data]
        if missing:
            raise KeyError(f"配置文件缺少字段: {', '.join(missing)}")
        self.data_path = config_data.get("DATA_PATH")
        self.trial_info = config_data.get("TRIAL_INFO", {})
        self.exp_info = config_data.get("EXP_INFO")

    def set_default_config(self):
        self.data_path = "C:\\Users\\wangy\\Desktop\\IC\\m79"
        self.trial_info = {
            "TRIAL_START_SKIP": 0,
            "TOTAL_TRIALS": 180
        }
        self.exp_info = {
            "t_stimulus": 12,
            "l_stimulus": 8,
            "l_trials": 32,
            "IPD": 2.0,
            "ISI": 6.0
        }

cfg = ExpConfig(r"C:\Users\wangy\Desktop\IC\m79\m79.json") 


# %% 数据加载函数
def process_trigger(txt_file, IPD=cfg.exp_info["IPD"], ISI=cfg.exp_info["ISI"], fre=None, min_sti_gap=4.0):
    """处理触发文件（与RR筛选代码完全一致）"""
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    ch_str = parts[1]
                    abs_ts = float(parts[2]) if len(parts) >= 3 else None
                    data.append((time_val, ch_str, abs_ts))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError("未能从文件中读取到有效数据")
    
    times, channels, abs_timestamps = zip(*data)
    times = np.array(times)
    
    ch_numeric = []
    valid_indices = []
    for i, ch_str in enumerate(channels):
        try:
            ch_val = float(ch_str)
            ch_numeric.append(ch_val)
            valid_indices.append(i)
        except ValueError:
            continue
    
    if not valid_indices:
        raise ValueError("未找到有效的数值通道数据")
    
    t = times[valid_indices]
    ch = np.array(ch_numeric)
    
    cam_t_raw = t[ch == 1]
    sti_t_raw = t[ch == 2]
    
    if len(cam_t_raw) == 0:
        raise ValueError("未检测到相机触发(值=1)")
    if len(sti_t_raw) == 0:
        raise ValueError("未检测到刺激触发(值=2)")
    
    sti_t = np.sort(sti_t_raw)
    if len(sti_t) > 0:
        keep = np.ones(len(sti_t), dtype=bool)
        for i in range(1, len(sti_t)):
            if (sti_t[i] - sti_t[i-1]) < min_sti_gap:
                keep[i] = False
        sti_t = sti_t[keep]
    
    if fre is None:
        dt = np.diff(cam_t_raw)
        fre = 1 / np.median(dt)

    IPD_frames = max(1, round(IPD * fre))
    isi_frames = round((IPD + ISI) * fre)
    
    cam_t = cam_t_raw.copy()
    nFrames = len(cam_t)
    start_edge = np.zeros(len(sti_t), dtype=int)
    
    for k in range(len(sti_t)):
        idx = np.argmin(np.abs(cam_t - sti_t[k]))
        start_edge[k] = idx
    
    end_edge = start_edge + IPD_frames - 1
    
    valid = (start_edge >= 0) & (end_edge < nFrames) & (start_edge <= end_edge)
    start_edge = start_edge[valid]
    end_edge = end_edge[valid]
    
    if len(start_edge) >= 2:
        d = np.diff(start_edge)
        while len(d) > 0 and d[-1] not in [isi_frames-1, isi_frames, isi_frames+1, isi_frames+2]:
            start_edge = start_edge[:-1]
            end_edge = end_edge[:-1]
            if len(start_edge) >= 2:
                d = np.diff(start_edge)
            else:
                break
    
    stimuli_array = np.zeros(nFrames)
    for i in range(len(start_edge)):
        stimuli_array[start_edge[i]:end_edge[i]+1] = 1
    
    return {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array,
        'camera_frames': len(cam_t),
        'stimuli_count': len(start_edge)
    }


def segment_neuron_data(neuron_data, trigger_data, label, pre_frames=cfg.exp_info["t_stimulus"], post_frames=cfg.exp_info["l_trials"]-cfg.exp_info["t_stimulus"]):
    """数据分割（与RR筛选代码完全一致）"""
    total_frames = pre_frames + post_frames
    segments = np.zeros((len(trigger_data), neuron_data.shape[1], total_frames))
    labels = []

    for i in range(len(trigger_data)):
        start = trigger_data[i] - pre_frames
        end = trigger_data[i] + post_frames
        if start < 0 or end >= neuron_data.shape[0]:
            print(f"警告: 第{i}个刺激的时间窗口超出边界，跳过")
            continue
        segment = neuron_data[start:end, :]
        segments[i] = segment.T
        labels.append(label[i])
    labels = np.array(labels)
    return segments, labels


def reclassify(stimulus_data):
    """刺激重新分类函数"""
    if BINARY_CLASSIFICATION:
        # 二分类：IC2和IC4合并为IC类(1)，LC2和LC4合并为LC类(2)
        mapping = {
            'IC2': 1,  # IC类
            'IC4': 1,  # IC类
            'LC2': 2,  # LC类
            'LC4': 2,  # LC类
        }
    else:
        # 四分类：每个刺激类型单独一类
        mapping = {
            'IC2': 1,  # 类别 1
            'IC4': 2,  # 类别 2  
            'LC2': 3,  # 类别 3
            'LC4': 4,  # 类别 4
        }
    
    new_labels = []
    for label in stimulus_data:
        new_labels.append(mapping.get(label, 0))
    return np.array(new_labels)


def load_data(data_path=cfg.data_path, start_idx=cfg.trial_info["TRIAL_START_SKIP"], 
              end_idx=cfg.trial_info["TRIAL_START_SKIP"] + cfg.trial_info["TOTAL_TRIALS"]):
    """加载数据（与RR筛选代码完全一致）"""
    print("开始加载数据...")
    mat_file = os.path.join(data_path, 'wholebrain_output.mat')
    if not os.path.exists(mat_file):
        raise ValueError(f"未找到神经数据文件: {mat_file}")
    
    try:
        data = h5py.File(mat_file, 'r')
    except Exception as e:
        raise ValueError(f"无法读取mat文件: {mat_file}，错误信息: {e}")

    if 'whole_trace_ori' not in data or 'whole_center' not in data:
        raise ValueError("mat文件缺少必要的数据集")

    neuron_data = np.array(data['whole_trace_ori'])
    print(f"原始神经数据形状: {neuron_data.shape}")
    
    neuron_data = np.nan_to_num(neuron_data, nan=0.0, posinf=0.0, neginf=0.0)
    neuron_pos = np.array(data['whole_center'])
    
    if neuron_pos.shape[0] > 2:
        neuron_pos = neuron_pos[0:2, :]

    # 触发数据
    trigger_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')])
    trigger_txt_candidates = [f for f in trigger_files if 'trigger' in os.path.basename(f).lower()]
    if not trigger_txt_candidates:
        raise FileNotFoundError(f"在 {data_path} 中未找到包含 'trigger' 字样的触发txt文件。")
    trigger_data = process_trigger(trigger_txt_candidates[0])
    
    # 刺激数据 - 使用stim_type.csv
    stimulus_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv') and 'stim_type' in f]
    if not stimulus_files:
        # 尝试解析txt文件
        txt_stim_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt') and 'stimuli' in f]
        if not txt_stim_files:
            raise FileNotFoundError(f"在 {data_path} 中未找到刺激csv/txt文件。")
        try:
            with open(txt_stim_files[-1], 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(txt_stim_files[-1], 'r', encoding='utf-8', errors='ignore') as f:
                print("警告: UTF-8 解码失败，已忽略非法字节继续解析 stimuli txt。")
                content = f.read()
        match = re.search(r"实际呈现序列 \(已呈现\): \r?\n?(\[.*?\])", content, re.DOTALL)
        if match:
            stim_list_str = match.group(1).replace("'", "\"")
            stim_list = json.loads(stim_list_str)
            stimulus_data = np.array(stim_list)
        else:
            raise ValueError("未能从 stimuli_...txt 文件中解析出刺激序列。")
    else:
        # 使用CSV文件
        stimulus_df = pd.read_csv(stimulus_files[0], header=None)
        stimulus_data = stimulus_df.iloc[:, 0].values.astype(str)
    
    # 调试信息：检查刺激数据
    print(f"刺激数据长度: {len(stimulus_data)}")
    print(f"刺激数据唯一值: {np.unique(stimulus_data)}")
    
    start_edges = trigger_data['start_edge'][start_idx:end_idx]
    if len(stimulus_data) < (end_idx - start_idx):
        print(f"警告: 刺激数据 ({len(stimulus_data)}个) 少于期望的试次数量 ({end_idx - start_idx}个)。")
        num_trials = min(len(stimulus_data), len(start_edges))
        start_edges = start_edges[:num_trials]
        stimulus_data = stimulus_data[:num_trials]
    else:
        stimulus_data = stimulus_data[start_idx:end_idx]
    
    print(f"最终使用的刺激数据长度: {len(stimulus_data)}")
    print(f"触发数据长度: {len(start_edges)}")
    
    return neuron_data, neuron_pos, start_edges, stimulus_data


def preprocess_dff_consistent(neuron_data):
    """计算 dF/F（与RR筛选代码完全一致的预处理逻辑）"""
    print("计算 dF/F（与RR筛选一致的预处理）...")
    
    # 第一步：去除负值神经元（与RR筛选完全一致）
    mask = np.any(neuron_data <= 0, axis=0)
    keep_idx = np.where(~mask)[0]
    
    neuron_data_filtered = neuron_data[:, keep_idx]
    print(f"去除负值后保留的神经元数量: {len(keep_idx)} (原始: {neuron_data.shape[1]})")
    
    # 第二步：计算 dF/F（与RR筛选完全一致的参数）
    if cfg.preprocess_cfg["preprocess"]:
        win_size = cfg.preprocess_cfg["win_size"]
        if win_size % 2 == 0:
            win_size += 1
        
        T, N = neuron_data_filtered.shape
        F0_dynamic = np.zeros((T, N), dtype=float)
        for i in range(N):
            # 使用与RR筛选完全相同的参数：percentile=8, mode='reflect'
            F0_dynamic[:, i] = ndimage.percentile_filter(
                neuron_data_filtered[:, i], 
                percentile=8, 
                size=win_size, 
                mode='reflect'
            )
        
        dff = (neuron_data_filtered - F0_dynamic) / F0_dynamic
    else:
        dff = neuron_data_filtered
    
    print(f"dF/F计算完成，形状: {dff.shape}")
    
    return dff, keep_idx


# %% SVM 相关函数
def find_optimal_frames(segments_rr, labels, rr_original_indices=None, cfg=cfg, n_frames=2):
    """
    基于保存的兴奋/抑制性RR索引找到最优时间帧
    """
    n_trials, n_neurons, n_timepoints = segments_rr.shape
    
    print(f"\n基于保存的RR索引统计响应分布...")
    
    # 1. 加载保存的兴奋性和抑制性RR神经元索引
    data_path = cfg.data_path
    
    # 加载兴奋性RR神经元原始索引
    rr_exc_orig_path = os.path.join(data_path, "all_stimuli_rr_excitatory_original_indices.npy")
    if os.path.exists(rr_exc_orig_path):
        rr_exc_original = np.load(rr_exc_orig_path)
        print(f"  加载兴奋性RR神经元: {len(rr_exc_original)} 个")
    else:
        print(f"错误: 未找到兴奋性RR索引文件 ({rr_exc_orig_path})")
        return None, None
    
    # 加载抑制性RR神经元原始索引
    rr_inh_orig_path = os.path.join(data_path, "all_stimuli_rr_inhibitory_original_indices.npy")
    if os.path.exists(rr_inh_orig_path):
        rr_inh_original = np.load(rr_inh_orig_path)
        print(f"  加载抑制性RR神经元: {len(rr_inh_original)} 个")
    else:
        print(f"错误: 未找到抑制性RR索引文件 ({rr_inh_orig_path})")
        return None, None
    
    # 2. 将原始索引映射到当前segments_rr中的索引
    # 注意：segments_rr已经是从原始数据中提取的RR神经元子集
    # 我们需要找到这些RR神经元在原始索引中的位置
    
    # 如果提供了rr_original_indices，直接使用它
    if rr_original_indices is not None:
        rr_union_original = rr_original_indices
        print(f"  使用提供的RR神经元原始索引: {len(rr_union_original)} 个")
    else:
        # 否则加载RR神经元并集原始索引
        rr_union_orig_path = os.path.join(data_path, "all_stimuli_rr_union_original_indices.npy")
        if os.path.exists(rr_union_orig_path):
            rr_union_original = np.load(rr_union_orig_path)
            print(f"  RR神经元并集总数: {len(rr_union_original)} 个")
        else:
            print(f"错误: 未找到RR并集索引文件 ({rr_union_orig_path})")
            return None, None
    
    # 在并集中找到兴奋性神经元的索引位置
    excited_neurons = []
    for orig_idx in rr_exc_original:
        pos = np.where(rr_union_original == orig_idx)[0]
        if len(pos) > 0:
            excited_neurons.append(pos[0])
    
    # 在并集中找到抑制性神经元的索引位置
    inhibited_neurons = []
    for orig_idx in rr_inh_original:
        pos = np.where(rr_union_original == orig_idx)[0]
        if len(pos) > 0:
            inhibited_neurons.append(pos[0])
    
    excited_neurons = np.array(excited_neurons, dtype=int)
    inhibited_neurons = np.array(inhibited_neurons, dtype=int)
    
    print(f"  映射后兴奋性RR神经元: {len(excited_neurons)} 个")
    print(f"  映射后抑制性RR神经元: {len(inhibited_neurons)} 个")
    print(f"  总RR神经元: {n_neurons} 个")
    print(f"  寻找连续 {n_frames} 帧...")
    
    if len(excited_neurons) == 0 and len(inhibited_neurons) == 0:
        print("错误: 没有找到有效的兴奋性或抑制性RR神经元")
        return None, None
    
    # 3. 确保索引不越界
    # 过滤掉超出当前segments_rr范围的索引
    excited_neurons = excited_neurons[excited_neurons < n_neurons]
    inhibited_neurons = inhibited_neurons[inhibited_neurons < n_neurons]
    
    print(f"  有效兴奋性RR神经元: {len(excited_neurons)} 个")
    print(f"  有效抑制性RR神经元: {len(inhibited_neurons)} 个")
    
    # 4. 统计兴奋性神经元的峰值
    excitation_score = np.zeros(n_timepoints)
    if len(excited_neurons) > 0:
        for t in range(1, n_timepoints - 1):
            for neuron_idx in excited_neurons:
                for trial in range(n_trials):
                    current = segments_rr[trial, neuron_idx, t]
                    left = segments_rr[trial, neuron_idx, t-1]
                    right = segments_rr[trial, neuron_idx, t+1]
                    
                    if current > left and current > right and current > 0:
                        excitation_score[t] += 1
    
    # 5. 统计抑制性神经元的谷值
    inhibition_score = np.zeros(n_timepoints)
    if len(inhibited_neurons) > 0:
        for t in range(1, n_timepoints - 1):
            for neuron_idx in inhibited_neurons:
                for trial in range(n_trials):
                    current = segments_rr[trial, neuron_idx, t]
                    left = segments_rr[trial, neuron_idx, t-1]
                    right = segments_rr[trial, neuron_idx, t+1]
                    
                    if current < left and current < right and current < 0:
                        inhibition_score[t] += 1
    
    # 6. 寻找连续 n_frames 的最优窗口
    total_score = excitation_score + inhibition_score
    
    window_scores = []
    for start in range(n_timepoints - n_frames + 1):
        window_score = np.sum(total_score[start:start + n_frames])
        window_scores.append((start, window_score))
    
    window_scores.sort(key=lambda x: x[1], reverse=True)
    best_start = window_scores[0][0]
    optimal_frames = np.arange(best_start, best_start + n_frames)
    
    print(f"  选中的连续时间帧: {optimal_frames}")
    print(f"  各帧的响应统计:")
    for frame in optimal_frames:
        print(f"    Frame {frame}: 兴奋 {int(excitation_score[frame])}, "
              f"抑制 {int(inhibition_score[frame])}, 合计 {int(total_score[frame])}")
    
    statistics_dict = {
        'excitation_score': excitation_score,
        'inhibition_score': inhibition_score,
        'total_score': total_score,
        'optimal_frames': optimal_frames,
        'excited_neurons': excited_neurons,
        'inhibited_neurons': inhibited_neurons,
        'window_scores': window_scores,
        'rr_exc_original_count': len(rr_exc_original),
        'rr_inh_original_count': len(rr_inh_original),
        'rr_exc_mapped_count': len(excited_neurons),
        'rr_inh_mapped_count': len(inhibited_neurons)
    }
    
    return optimal_frames, statistics_dict

def extract_features(segments_rr, optimal_frames):
    """
    为每个 trial 提取特征向量。
    """
    n_trials, n_neurons, n_timepoints = segments_rr.shape
    
    features = np.zeros((n_trials, n_neurons))
    
    for trial in range(n_trials):
        for neuron in range(n_neurons):
            features[trial, neuron] = np.mean(segments_rr[trial, neuron, optimal_frames])
    
    print(f"\n特征提取完成:")
    print(f"  特征矩阵形状: {features.shape}")
    print(f"  每个 trial 特征维度: {n_neurons}")
    print(f"  特征计算: 对选定的 {len(optimal_frames)} 帧 ΔF/F 取平均值")
    
    return features


def visualize_statistics(statistics_dict, cfg=cfg, save_dir='svm_results', mode='all_rr'):
    """可视化统计结果"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    excitation_score = statistics_dict['excitation_score']
    inhibition_score = statistics_dict['inhibition_score']
    total_score = statistics_dict['total_score']
    optimal_frames = statistics_dict['optimal_frames']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 所有时间点的兴奋和抑制响应
    time_axis = np.arange(len(total_score))
    axes[0].bar(time_axis, excitation_score, alpha=0.7, color='red', label='Excitation (peaks)')
    axes[0].bar(time_axis, inhibition_score, alpha=0.7, color='blue', label='Inhibition (valleys)', 
                bottom=excitation_score)
    
    for frame in optimal_frames:
        axes[0].axvline(x=frame, color='green', linestyle='--', linewidth=2, 
                       label=f'Selected frame {frame}' if frame == optimal_frames[0] else '')
    
    axes[0].axvline(x=cfg.exp_info['t_stimulus'], color='orange', linestyle='--', linewidth=1.5, 
                   label='Stimulus onset')
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Count of responses')
    axes[0].set_title(f'Excitation and Inhibition Distribution Across Time ({mode})')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 右图: 总响应分布和连续窗口得分
    axes[1].bar(time_axis, total_score, alpha=0.7, color='steelblue')
    
    for frame in optimal_frames:
        axes[1].axvline(x=frame, color='green', linestyle='--', linewidth=2)
    
    axes[1].set_xlabel('Time (frames)')
    axes[1].set_ylabel('Total response count')
    axes[1].set_title(f'Total Response Score (Excitation + Inhibition) ({mode})')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, f'response_statistics_{mode}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"统计图已保存: {out_path}")


def train_svm(features, labels, save_dir='svm_results', n_components=PCA_COMPONENTS, use_pca=USE_PCA, mode='all_rr'):
    """
    训练 SVM 分类器，使用 8:2 训练/测试集划分。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始 SVM 训练（8:2 训练/测试集划分）...")
    print(f"  模式: {mode}")
    print(f"  原始特征矩阵形状: {features.shape}")
    
    # 检查标签分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"  标签类型及数量: {dict(zip(unique_labels, label_counts))}")
    
    # 如果所有标签都是0，无法训练SVM
    if len(unique_labels) <= 1:
        print("错误: 需要至少2个类别的标签才能进行SVM分类！")
        print("当前所有标签都是同一个类别。")
        return None, None, None, None
    
    # 第一步：数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 第二步：PCA 降维（可选）
    if use_pca:
        max_comp = min(features_scaled.shape[0], features_scaled.shape[1])
        n_components_eff = max(1, min(n_components, max_comp))
        if n_components_eff != n_components:
            print(f"\n请求的 n_components={n_components} 超过样本/特征限制，自动调整为 {n_components_eff}")
        else:
            print(f"\n执行 PCA 降维到 {n_components_eff} 维...")
        pca = PCA(n_components=n_components_eff)
        features_pca = pca.fit_transform(features_scaled)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"  PCA 降维完成")
        print(f"  降维后特征矩阵形状: {features_pca.shape}")
        print(f"  保留方差比例: {explained_variance*100:.2f}%")
    else:
        pca = None
        features_pca = features_scaled
        print(f"\n已配置为不使用 PCA，使用标准化后的原始特征，形状: {features_pca.shape}")
    
    # 第三步：数据shuffle（新增步骤）
    if SHUFFLE_DATA:
        print(f"\n执行数据shuffle...")
        print(f"  打乱前前5个标签: {labels[:5]}")
        
        # 使用sklearn的shuffle函数打乱特征和标签
        features_shuffled, labels_shuffled = shuffle(features_pca, labels, random_state=RANDOM_STATE)
        print(f"  打乱后前5个标签: {labels_shuffled[:5]}")
        
        # 更新变量
        features_pca = features_shuffled
        labels = labels_shuffled
    else:
        print(f"\n跳过数据shuffle（SHUFFLE_DATA=False）")
    
    # 第四步：8:2 训练/测试集划分（分层以保证类平衡）
    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, labels, 
        test_size=0.2, 
        stratify=labels,
        random_state=RANDOM_STATE
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 个样本（{X_train.shape[0]/len(labels)*100:.1f}%）")
    print(f"  测试集: {X_test.shape[0]} 个样本（{X_test.shape[0]/len(labels)*100:.1f}%）")
    print(f"  训练集标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  测试集标签分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # 第五步：训练 SVM（线性核）
    svm_model = SVC(kernel='linear', random_state=RANDOM_STATE)
    svm_model.fit(X_train, y_train)
    
    print(f"\n模型训练完成:")
    print(f"  支持向量数: {len(svm_model.support_vectors_)}")
    
    # 第六步：在训练集上评估
    y_train_pred = svm_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # 第七步：在测试集上评估
    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n性能指标:")
    print(f"  训练集准确率: {train_accuracy:.4f}")
    print(f"  测试集准确率: {test_accuracy:.4f}")
    print(f"  过拟合差异: {train_accuracy - test_accuracy:.4f}")
    
    # 测试集详细分类报告
    unique_test_labels = np.unique(y_test)
    target_names = []
    for lbl in unique_test_labels:
        if BINARY_CLASSIFICATION:
            # 二分类标签名称
            if lbl == 1:
                target_names.append('IC (1)')
            elif lbl == 2:
                target_names.append('LC (2)')
            else:
                target_names.append(f'Class {int(lbl)}')
        else:
            # 四分类标签名称
            if lbl == 1:
                target_names.append('IC2 (1)')
            elif lbl == 2:
                target_names.append('IC4 (2)')
            elif lbl == 3:
                target_names.append('LC2 (3)')
            elif lbl == 4:
                target_names.append('LC4 (4)')
            else:
                target_names.append(f'Class {int(lbl)}')
    
    class_report = classification_report(y_test, y_test_pred, target_names=target_names)
    print(f"\n测试集分类报告:\n{class_report}")
    
    # 混淆矩阵
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    
    # 打印混淆矩阵
    print(f"\n训练集混淆矩阵:")
    print_confusion_matrix(conf_matrix_train, target_names)
    
    print(f"\n测试集混淆矩阵:")
    print_confusion_matrix(conf_matrix_test, target_names)
    
    results_dict = {
        'scaler': scaler,
        'pca': pca,
        'svm_model': svm_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_confusion_matrix': conf_matrix_train,
        'test_confusion_matrix': conf_matrix_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'class_report': class_report,
        'mode': mode,
        'shuffled': SHUFFLE_DATA,
        'random_state': RANDOM_STATE
    }
    results_dict['class_names'] = target_names
    
    return svm_model, scaler, pca, results_dict


def print_confusion_matrix(conf_matrix, class_names):
    """打印格式化的混淆矩阵"""
    n_classes = len(class_names)
    
    # 打印表头
    header = "True \\ Pred" + "".join([f"{name:>12}" for name in class_names])
    print(header)
    print("-" * len(header))
    
    # 打印每一行
    for i in range(n_classes):
        row = f"{class_names[i]:>12}"
        for j in range(n_classes):
            row += f"{conf_matrix[i, j]:>12}"
        print(row)
    
    # 打印统计信息
    print("\n混淆矩阵统计:")
    for i in range(n_classes):
        true_positives = conf_matrix[i, i]
        false_positives = sum(conf_matrix[j, i] for j in range(n_classes) if j != i)
        false_negatives = sum(conf_matrix[i, j] for j in range(n_classes) if j != i)
        true_negatives = np.sum(conf_matrix) - true_positives - false_positives - false_negatives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {class_names[i]}:")
        print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")


def visualize_svm_results(results_dict, save_dir='svm_results', pca=None):
    """
    可视化 SVM 结果，包括混淆矩阵和 PCA 方差解释
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    conf_matrix_train = results_dict['train_confusion_matrix']
    conf_matrix_test = results_dict['test_confusion_matrix']
    train_accuracy = results_dict['train_accuracy']
    test_accuracy = results_dict['test_accuracy']
    class_names = results_dict.get('class_names', None)
    mode = results_dict.get('mode', 'all_rr')
    
    # 如果有 PCA，显示 3 列：训练混淆矩阵 | 测试混淆矩阵 | PCA 方差
    if pca is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        if class_names is None:
            class_names = ['Class %d' % i for i in range(conf_matrix_train.shape[0])]
        sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        axes[0].set_title(f'Training Set Confusion Matrix\n(Accuracy: {train_accuracy:.4f}, {mode})')
        
        sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        axes[1].set_title(f'Test Set Confusion Matrix\n(Accuracy: {test_accuracy:.4f}, {mode})')

        # PCA 解释方差
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[2].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, alpha=0.6, label='Individual')
        axes[2].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'r-o', label='Cumulative')
        axes[2].set_xlabel('PCA Component')
        axes[2].set_ylabel('Explained Variance Ratio')
        axes[2].set_title(f'PCA Explained Variance\n(Total: {cumsum_var[-1]*100:.2f}%, {mode})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if class_names is None:
            class_names = ['Class %d' % i for i in range(conf_matrix_train.shape[0])]
        sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        axes[0].set_title(f'Training Set Confusion Matrix\n(Accuracy: {train_accuracy:.4f}, {mode})')
        
        sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        axes[1].set_title(f'Test Set Confusion Matrix\n(Accuracy: {test_accuracy:.4f}, {mode})')
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, f'svm_results_{mode}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"SVM 结果图已保存: {out_path}")


def save_svm_results(results_dict, optimal_frames, neurons_count, save_dir, mode, separate_exc_inh=False, 
                    exc_neurons_count=0, inh_neurons_count=0):
    """保存SVM结果到JSON文件"""
    # 计算混淆矩阵的详细统计信息
    conf_matrix_test = results_dict['test_confusion_matrix']
    class_names = results_dict.get('class_names', [])
    
    confusion_matrix_stats = {}
    for i, class_name in enumerate(class_names):
        true_positives = conf_matrix_test[i, i]
        false_positives = sum(conf_matrix_test[j, i] for j in range(len(class_names)) if j != i)
        false_negatives = sum(conf_matrix_test[i, j] for j in range(len(class_names)) if j != i)
        true_negatives = np.sum(conf_matrix_test) - true_positives - false_positives - false_negatives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        confusion_matrix_stats[class_name] = {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    results_dict_save = {
        'optimal_frames': optimal_frames.tolist(),
        'rr_neurons_count': neurons_count,
        'train_size': len(results_dict['y_train']),
        'test_size': len(results_dict['y_test']),
        'train_accuracy': float(results_dict['train_accuracy']),
        'test_accuracy': float(results_dict['test_accuracy']),
        'overfitting_gap': float(results_dict['train_accuracy'] - results_dict['test_accuracy']),
        'train_confusion_matrix': results_dict['train_confusion_matrix'].tolist(),
        'test_confusion_matrix': results_dict['test_confusion_matrix'].tolist(),
        'confusion_matrix_stats': confusion_matrix_stats,  # 新增：混淆矩阵详细统计
        'class_report': results_dict['class_report'],
        'svm_kernel': 'linear',
        'pca_used': USE_PCA,
        'pca_components': PCA_COMPONENTS if USE_PCA else None,
        'neuron_selection': mode,
        'separate_exc_inh': separate_exc_inh,
        'binary_classification': BINARY_CLASSIFICATION,
        'use_pca': USE_PCA,
        'n_frames': N_FRAMES,
        'shuffled': SHUFFLE_DATA,
        'random_state': RANDOM_STATE
    }
    
    # 如果区分模式，添加详细的神经元数量信息
    if separate_exc_inh:
        results_dict_save['exc_neurons_count'] = exc_neurons_count
        results_dict_save['inh_neurons_count'] = inh_neurons_count
    
    result_json_path = os.path.join(save_dir, f'svm_classification_results_{mode}.json')
    with open(result_json_path, 'w') as f:
        json.dump(results_dict_save, f, indent=2)
    print(f"\n结果已保存: {result_json_path}")
    
    return result_json_path


# %% 主程序
if __name__ == "__main__":
    print("=" * 80)
    print("SVM 分类分析 - 可选择是否区分兴奋性和抑制性RR神经元")
    print("=" * 80)
    
    # 显示当前配置
    print(f"\n当前配置:")
    print(f"  区分兴奋/抑制性: {SEPARATE_EXC_INH}")
    print(f"  使用PCA: {USE_PCA}")
    if USE_PCA:
        print(f"  PCA维度: {PCA_COMPONENTS}")
    print(f"  连续时间帧数: {N_FRAMES}")
    print(f"  分类类型: {'二分类 (IC vs LC)' if BINARY_CLASSIFICATION else '四分类 (IC2, IC4, LC2, LC4)'}")
    print(f"  数据shuffle: {SHUFFLE_DATA}")
    print(f"  随机种子: {RANDOM_STATE}")
    
    # 0. 确定实验体目录
    data_path = cfg.data_path
    print(f"\n使用数据路径: {data_path}")
    
    # 确定保存目录（在实验体目录下创建 svm_results）
    save_dir = os.path.join(data_path, 'svm_results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    # 1. 加载数据（使用与RR筛选完全相同的函数）
    neuron_data, neuron_pos, start_edges, stimulus_data = load_data()
    
    # 检查刺激数据
    print(f"\n刺激数据检查:")
    print(f"  刺激数据长度: {len(stimulus_data)}")
    print(f"  刺激数据唯一值: {np.unique(stimulus_data)}")
    
    labels = reclassify(stimulus_data)
    
    # 检查标签分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"\n最终标签分布:")
    label_info = dict(zip(unique_labels, label_counts))
    for label, count in label_info.items():
        if label == 0:
            print(f"  未识别标签: {count} 个")
        else:
            if BINARY_CLASSIFICATION:
                if label == 1:
                    print(f"  IC类: {count} 个")
                elif label == 2:
                    print(f"  LC类: {count} 个")
            else:
                if label == 1:
                    print(f"  IC2: {count} 个")
                elif label == 2:
                    print(f"  IC4: {count} 个")
                elif label == 3:
                    print(f"  LC2: {count} 个")
                elif label == 4:
                    print(f"  LC4: {count} 个")
    
    # 如果所有标签都是0，停止执行
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        print("\n错误: 所有标签都是0，无法进行分类分析！")
        exit(1)
    
    # 2. 预处理 - 计算 dF/F（使用与RR筛选完全相同的预处理逻辑）
    dff, keep_idx = preprocess_dff_consistent(neuron_data)
    
    # 3. 分割数据（使用与RR筛选完全相同的分割函数和参数）
    segments_all, labels = segment_neuron_data(dff, start_edges, labels)
    print(f"数据分割完成，segments_all 形状: {segments_all.shape}, labels 形状: {labels.shape}")
    
    # 检查分割后的标签
    unique_labels_after, counts_after = np.unique(labels, return_counts=True)
    print(f"分割后标签分布: {dict(zip(unique_labels_after, counts_after))}")
    
    # 4. 根据开关选择神经元模式
    if SEPARATE_EXC_INH:
        print(f"\n模式: 区分兴奋性和抑制性RR神经元")
        
        # 分别加载兴奋性和抑制性RR神经元原始索引
        rr_exc_path = os.path.join(data_path, "all_stimuli_rr_excitatory_original_indices.npy")
        rr_inh_path = os.path.join(data_path, "all_stimuli_rr_inhibitory_original_indices.npy")
        
        if not os.path.exists(rr_exc_path) or not os.path.exists(rr_inh_path):
            print(f"错误: 未找到兴奋性或抑制性RR索引文件")
            print(f"  兴奋性文件: {rr_exc_path}")
            print(f"  抑制性文件: {rr_inh_path}")
            exit(1)
        
        rr_exc_original = np.load(rr_exc_path)
        rr_inh_original = np.load(rr_inh_path)
        
        print(f"  兴奋性RR神经元: {len(rr_exc_original)} 个")
        print(f"  抑制性RR神经元: {len(rr_inh_original)} 个")
        
        # 将原始索引映射到过滤后的索引
        rr_exc_filtered = []
        for orig_idx in rr_exc_original:
            pos = np.where(keep_idx == orig_idx)[0]
            if len(pos) > 0:
                rr_exc_filtered.append(pos[0])
        
        rr_inh_filtered = []
        for orig_idx in rr_inh_original:
            pos = np.where(keep_idx == orig_idx)[0]
            if len(pos) > 0:
                rr_inh_filtered.append(pos[0])
        
        rr_exc_filtered = np.array(rr_exc_filtered, dtype=int)
        rr_inh_filtered = np.array(rr_inh_filtered, dtype=int)
        
        print(f"  映射后兴奋性RR神经元: {len(rr_exc_filtered)} 个")
        print(f"  映射后抑制性RR神经元: {len(rr_inh_filtered)} 个")
        
        if len(rr_exc_filtered) == 0 and len(rr_inh_filtered) == 0:
            print("错误: 没有有效的兴奋性或抑制性RR神经元映射")
            exit(1)
        
        # 提取兴奋性和抑制性神经元的 dF/F
        segments_exc = segments_all[:, rr_exc_filtered, :] if len(rr_exc_filtered) > 0 else None
        segments_inh = segments_all[:, rr_inh_filtered, :] if len(rr_inh_filtered) > 0 else None
        
        print(f"  兴奋性RR神经元分割数据形状: {segments_exc.shape if segments_exc is not None else 'None'}")
        print(f"  抑制性RR神经元分割数据形状: {segments_inh.shape if segments_inh is not None else 'None'}")
        
        # 分别进行三个SVM分析：兴奋性、抑制性、合并
        
        # 4.1 兴奋性RR神经元分析
        if segments_exc is not None and len(rr_exc_filtered) > 0:
            print(f"\n{'='*50}")
            print("进行兴奋性RR神经元SVM分析")
            print(f"{'='*50}")
            
            # 找到最优时间帧（使用兴奋性神经元）
            optimal_frames_exc, statistics_dict_exc = find_optimal_frames(
                segments_exc, labels, rr_original_indices=rr_exc_original, cfg=cfg, n_frames=N_FRAMES
            )
            
            if optimal_frames_exc is not None:
                # 可视化统计结果
                visualize_statistics(statistics_dict_exc, cfg=cfg, save_dir=save_dir, mode='excitatory_only')
                
                # 提取特征
                features_exc = extract_features(segments_exc, optimal_frames_exc)
                
                # 训练SVM
                svm_model_exc, scaler_exc, pca_exc, results_dict_exc = train_svm(
                    features_exc, labels, 
                    save_dir=save_dir, 
                    n_components=PCA_COMPONENTS,
                    use_pca=USE_PCA,
                    mode='excitatory_only'
                )
                
                if svm_model_exc is not None:
                    # 可视化SVM结果
                    visualize_svm_results(results_dict_exc, save_dir=save_dir, pca=pca_exc)
                    
                    # 保存结果
                    save_svm_results(
                        results_dict_exc, optimal_frames_exc, len(rr_exc_filtered), 
                        save_dir, 'excitatory_only', separate_exc_inh=True,
                        exc_neurons_count=len(rr_exc_filtered), inh_neurons_count=0
                    )
        
        # 4.2 抑制性RR神经元分析
        if segments_inh is not None and len(rr_inh_filtered) > 0:
            print(f"\n{'='*50}")
            print("进行抑制性RR神经元SVM分析")
            print(f"{'='*50}")
            
            # 找到最优时间帧（使用抑制性神经元）
            optimal_frames_inh, statistics_dict_inh = find_optimal_frames(
                segments_inh, labels, rr_original_indices=rr_inh_original, cfg=cfg, n_frames=N_FRAMES
            )
            
            if optimal_frames_inh is not None:
                # 可视化统计结果
                visualize_statistics(statistics_dict_inh, cfg=cfg, save_dir=save_dir, mode='inhibitory_only')
                
                # 提取特征
                features_inh = extract_features(segments_inh, optimal_frames_inh)
                
                # 训练SVM
                svm_model_inh, scaler_inh, pca_inh, results_dict_inh = train_svm(
                    features_inh, labels, 
                    save_dir=save_dir, 
                    n_components=PCA_COMPONENTS,
                    use_pca=USE_PCA,
                    mode='inhibitory_only'
                )
                
                if svm_model_inh is not None:
                    # 可视化SVM结果
                    visualize_svm_results(results_dict_inh, save_dir=save_dir, pca=pca_inh)
                    
                    # 保存结果
                    save_svm_results(
                        results_dict_inh, optimal_frames_inh, len(rr_inh_filtered), 
                        save_dir, 'inhibitory_only', separate_exc_inh=True,
                        exc_neurons_count=0, inh_neurons_count=len(rr_inh_filtered)
                    )
        
        # 4.3 合并兴奋性和抑制性RR神经元分析
        print(f"\n{'='*50}")
        print("进行合并兴奋性和抑制性RR神经元SVM分析")
        print(f"{'='*50}")
        
        # 合并兴奋性和抑制性神经元数据
        if segments_exc is not None and segments_inh is not None:
            segments_combined = np.concatenate([segments_exc, segments_inh], axis=1)
        elif segments_exc is not None:
            segments_combined = segments_exc
        else:
            segments_combined = segments_inh
            
        print(f"  合并后RR神经元分割数据形状: {segments_combined.shape}")
        
        # 找到最优时间帧（使用合并的神经元）
        combined_original_indices = np.concatenate([rr_exc_original, rr_inh_original])
        optimal_frames_combined, statistics_dict_combined = find_optimal_frames(
            segments_combined, labels, rr_original_indices=combined_original_indices, cfg=cfg, n_frames=N_FRAMES
        )
        
        if optimal_frames_combined is not None:
            # 可视化统计结果
            visualize_statistics(statistics_dict_combined, cfg=cfg, save_dir=save_dir, mode='combined_exc_inh')
            
            # 提取特征
            features_combined = extract_features(segments_combined, optimal_frames_combined)
            
            # 训练SVM
            svm_model_combined, scaler_combined, pca_combined, results_dict_combined = train_svm(
                features_combined, labels, 
                save_dir=save_dir, 
                n_components=PCA_COMPONENTS,
                use_pca=USE_PCA,
                mode='combined_exc_inh'
            )
            
            if svm_model_combined is not None:
                # 可视化SVM结果
                visualize_svm_results(results_dict_combined, save_dir=save_dir, pca=pca_combined)
                
                # 保存结果
                save_svm_results(
                    results_dict_combined, optimal_frames_combined, segments_combined.shape[1], 
                    save_dir, 'combined_exc_inh', separate_exc_inh=True,
                    exc_neurons_count=len(rr_exc_filtered), inh_neurons_count=len(rr_inh_filtered)
                )
        
    else:
        print(f"\n模式: 不区分兴奋性和抑制性，使用所有RR神经元")
        
        # 加载所有RR神经元原始索引（兴奋性和抑制性并集）
        rr_union_path = os.path.join(data_path, "all_stimuli_rr_union_original_indices.npy")
        
        if os.path.exists(rr_union_path):
            rr_neurons_original = np.load(rr_union_path)
            print(f"  加载 RR 神经元并集原始索引: {len(rr_neurons_original)} 个")
            print(f"  从文件: {rr_union_path}")
        else:
            print(f"错误: 未找到 RR 并集索引文件 ({rr_union_path})")
            print("请先运行 loaddata_final.py 生成 RR 神经元索引")
            exit(1)
        
        # 将原始索引映射到过滤后的索引
        rr_idx_filtered = []
        for orig_idx in rr_neurons_original:
            pos = np.where(keep_idx == orig_idx)[0]
            if len(pos) > 0:
                rr_idx_filtered.append(pos[0])
        
        rr_idx_filtered = np.array(rr_idx_filtered, dtype=int)
        
        print(f"  映射后 RR 神经元数: {len(rr_idx_filtered)}")
        if len(rr_idx_filtered) == 0:
            print("错误: 没有有效的 RR 神经元映射，请检查索引文件")
            exit(1)
        
        # 提取所有RR神经元的 dF/F
        segments_rr = segments_all[:, rr_idx_filtered, :]
        print(f"  所有RR神经元分割数据形状: {segments_rr.shape}")
        
        # 5. 找到最优的连续时间帧
        optimal_frames, statistics_dict = find_optimal_frames(
            segments_rr, labels, rr_original_indices=rr_neurons_original, cfg=cfg, n_frames=N_FRAMES
        )
        
        if optimal_frames is None:
            print("错误: 无法找到最优时间帧")
            exit(1)
        
        # 6. 可视化统计结果
        visualize_statistics(statistics_dict, cfg=cfg, save_dir=save_dir, mode='all_rr')
        
        # 7. 提取特征
        features = extract_features(segments_rr, optimal_frames)

        # 8. 训练 SVM
        print(f"\n使用所有RR神经元进行SVM分析...")
        svm_model, scaler, pca, results_dict = train_svm(
            features, labels, 
            save_dir=save_dir, 
            n_components=PCA_COMPONENTS,
            use_pca=USE_PCA,
            mode='all_rr'
        )
        
        if svm_model is None:
            print("SVM 训练失败，无法继续执行。")
            exit(1)
        
        # 9. 可视化 SVM 结果
        visualize_svm_results(results_dict, save_dir=save_dir, pca=pca)
        
        # 10. 保存模型和结果
        save_svm_results(
            results_dict, optimal_frames, len(rr_idx_filtered), 
            save_dir, 'all_rr', separate_exc_inh=False
        )

    print("\n" + "=" * 80)
    print("SVM 分类分析完成！")
    if SEPARATE_EXC_INH:
        print("模式: 区分兴奋/抑制性RR神经元")
        print("已分别保存兴奋性、抑制性和合并的SVM结果")
    else:
        print("模式: 使用所有RR神经元（不区分）")
    print(f"分类类型: {'二分类 (IC vs LC)' if BINARY_CLASSIFICATION else '四分类 (IC2, IC4, LC2, LC4)'}")
    print(f"数据Shuffle: {SHUFFLE_DATA}")
    print(f"随机种子: {RANDOM_STATE}")
    print("=" * 80)
    print(f"所有结果已保存到: {save_dir}")