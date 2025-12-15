'''
对四个刺激类别（IC2、IC4、LC2、LC4）分别筛选RR神经元，并保存各时间段信息


'''
import h5py
import os
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage # 预处理需要用到
import time # 导入 time 模块用于计时

# %% 定义配置

class ExpConfig:
    def __init__(self, file_path = None):
        # 加载配置文件
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
            'win_size' : 150
        }

    def load_config(self, file_path):
        # 从文件加载配置
        if not file_path.endswith('.json'):
            raise NotImplementedError("目前仅支持JSON格式的配置文件")
        # 解析配置数据
        import json
        with open(file_path, 'r') as f:
            config_data = json.load(f)  

        # 检查必要字段
        required_keys = ['DATA_PATH']
        missing = [k for k in required_keys if k not in config_data]
        if missing:
            raise KeyError(f"配置文件缺少字段: {', '.join(missing)}")
        
        # 赋值配置
        self.data_path = config_data.get("DATA_PATH")
        self.trial_info = config_data.get("TRIAL_INFO", {})
        self.exp_info = config_data.get("EXP_INFO")


    def set_default_config(self):
        # 设置默认配置
        # 数据路径
        self.data_path = r'C:\Users\wangy\Desktop\IC\m79'
        # 试次信息
        self.trial_info = {
            "TRIAL_START_SKIP": 0,
            "TOTAL_TRIALS": 180
        }
        # 刺激参数
        self.exp_info = {
            "t_stimulus": 12,  #刺激前帧数
            "l_stimulus": 8,   #刺激持续帧数
            "l_trials": 32,    #单试次总帧数
            "IPD":2.0,
            "ISI":6.0
        }


cfg = ExpConfig(r'C:\Users\wangy\Desktop\IC\m79\m79.json')

# %% 预处理相关函数定义(通用)
# 从matlab改过来的，经过检查应该无误
def process_trigger(txt_file, IPD=cfg.exp_info["IPD"], ISI=cfg.exp_info["ISI"], fre=None, min_sti_gap=4.0):
    """
    处理触发文件，修改自step1x_trigger_725right.m
    """
    
    # 读入文件
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
    
    # 解析数据
    times, channels, abs_timestamps = zip(*data)
    times = np.array(times)
    
    # 转换通道为数值，非数值的设为NaN
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
    
    # 只保留有效数据
    t = times[valid_indices]
    ch = np.array(ch_numeric)
    
    # 相机帧与刺激起始时间
    cam_t_raw = t[ch == 1]
    sti_t_raw = t[ch == 2]
    
    if len(cam_t_raw) == 0:
        raise ValueError("未检测到相机触发(值=1)")
    if len(sti_t_raw) == 0:
        raise ValueError("未检测到刺激触发(值=2)")
    
    # 去重/合并：将时间靠得很近的"2"视作同一次刺激
    sti_t = np.sort(sti_t_raw)
    if len(sti_t) > 0:
        keep = np.ones(len(sti_t), dtype=bool)
        for i in range(1, len(sti_t)):
            if (sti_t[i] - sti_t[i-1]) < min_sti_gap:
                keep[i] = False  # 合并到前一个
        sti_t = sti_t[keep]
    
    # 帧率估计或使用给定值
    if fre is None:
        dt = np.diff(cam_t_raw)
        fre = 1 / np.median(dt)  # 用相机帧时间戳的中位间隔

    IPD_frames = max(1, round(IPD * fre))
    isi_frames = round((IPD + ISI) * fre)
    
    # 把每个刺激时间映射到最近的相机帧索引
    cam_t = cam_t_raw.copy()
    nFrames = len(cam_t)
    start_edge = np.zeros(len(sti_t), dtype=int)        #所有刺激起始帧
    
    for k in range(len(sti_t)):
        idx = np.argmin(np.abs(cam_t - sti_t[k]))
        start_edge[k] = idx
    
    end_edge = start_edge + IPD_frames - 1
    
    # 边界裁剪，避免越界
    valid = (start_edge >= 0) & (end_edge < nFrames) & (start_edge <= end_edge)
    start_edge = start_edge[valid]
    end_edge = end_edge[valid]
    
    # 尾段完整性检查（与旧逻辑一致）
    if len(start_edge) >= 2:
        d = np.diff(start_edge)
        while len(d) > 0 and d[-1] not in [isi_frames-1, isi_frames, isi_frames+1, isi_frames+2]:
            # 丢掉最后一个可疑的刺激段
            start_edge = start_edge[:-1]
            end_edge = end_edge[:-1]
            if len(start_edge) >= 2:
                d = np.diff(start_edge)
            else:
                break
    
    # 生成0/1刺激数组（可视化/保存用）
    stimuli_array = np.zeros(nFrames)
    for i in range(len(start_edge)):
        stimuli_array[start_edge[i]:end_edge[i]+1] = 1
    
    # 保存结果到mat文件
    save_path = os.path.join(os.path.dirname(txt_file), 'visual_stimuli_with_label.mat')
    scipy.io.savemat(save_path, {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array
    })
    
    return {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array,
        'camera_frames': len(cam_t),
        'stimuli_count': len(start_edge)
    }

# ========== 核心修改: 单类别RR神经元筛选函数 (原 rr_selection) ========== 
def _rr_selection_single(trials, t_stimulus=cfg.exp_info["t_stimulus"], l=cfg.exp_info["l_stimulus"], reliability_threshold=0.65, snr_threshold=0.8, effect_size_threshold=0.5, response_ratio_threshold=0.6, class_label="All"):
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    对一组试次（来自一个刺激类别）进行快速RR神经元筛选
    """
    n_trials, n_neurons, n_timepoints = trials.shape
    
    print(f"正在对类别 {class_label} 进行筛选, 试次数量: {n_trials}, 神经元数量: {n_neurons}")
    
    # 定义时间窗口
    baseline_pre = np.arange(0, t_stimulus)
    baseline_post = np.arange(t_stimulus + l, n_timepoints)
    stimulus_window = np.arange(t_stimulus, t_stimulus + l)
    
    # 1. 响应性检测 - 向量化计算
    # 计算基线和刺激期的平均值
    baseline_pre_mean = np.mean(trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_mean = np.mean(trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的平均
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2
    
    stimulus_mean = np.mean(trials[:, :, stimulus_window], axis=2)  # (trials, neurons)
    
    # 简化的响应性检测：基于效应大小和标准误差
    baseline_pre_std = np.std(trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_std = np.std(trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的标准差
    baseline_std = (baseline_pre_std + baseline_post_std) / 2
    
    stimulus_std = np.std(trials[:, :, stimulus_window], axis=2)
    
    # Cohen's d效应大小
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)
    
    # 响应性标准：平均效应大小 > 阈值 且 至少指定比例试次有响应
    response_ratio = np.mean(effect_size > effect_size_threshold, axis=0)
    
    # 兴奋性响应 (Excitatory): 响应比例 > 阈值 且 平均响应 > 平均基线比例 > 阈值
    enhanced_neurons = np.where((response_ratio > response_ratio_threshold) & 
                                (np.mean(stimulus_mean > baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()
    # 抑制性响应 (Inhibitory): 响应比例 > 阈值 且 平均响应 < 平均基线比例 > 阈值
    inhibitory_neurons = np.where((response_ratio > response_ratio_threshold) &
                                  (np.mean(stimulus_mean < baseline_mean, axis=0) > response_ratio_threshold))[0].tolist()

    # 2. 可靠性检测 - 简化版本
    # 计算每个神经元在每个试次的信噪比
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    noise_level = baseline_std + 1e-8
    snr = signal_strength / noise_level
    
    # 可靠性：指定比例的试次信噪比 > 阈值
    reliability_ratio = np.mean(snr > snr_threshold, axis=0)
    reliable_neurons = np.where(reliability_ratio >= reliability_threshold)[0].tolist()
    
    # 3. 最终RR神经元
    rr_enhanced_neurons = list(set(enhanced_neurons) & set(reliable_neurons))
    rr_inhibitory_neurons = list(set(inhibitory_neurons) & set(reliable_neurons))
    
    print(f"类别 {class_label} 筛选结果: 兴奋性RR: {len(rr_enhanced_neurons)}, 抑制性RR: {len(rr_inhibitory_neurons)}")

    # 返回神经元在输入 trials 中的**索引**
    return set(rr_enhanced_neurons), set(rr_inhibitory_neurons)

# ========== 新增: 分类别RR神经元筛选函数 (满足用户需求) ========== 
def rr_selection_by_class(segments, labels, **kwargs):
    """
    分刺激类型筛选 RR 神经元，然后取并集。
    
    参数:
    segments: (n_trials, n_neurons, n_timepoints)
    labels: (n_trials,) 包含类别标签的 NumPy 数组
    **kwargs: 传递给 _rr_selection_single 的筛选参数
    
    返回:
    rr_enhanced_neurons: 对任一刺激类别有兴奋性 RR 的神经元全局索引 (列表)
    rr_inhibitory_neurons: 对任一刺激类别有抑制性 RR 的神经元全局索引 (列表)
    """
    start_time = time.time()
    print("\n开始分类别 RR 神经元筛选...")
    
    all_class_ids = sorted(np.unique(labels))
    # 类别 0 通常是无效/跳过的试次，跳过
    valid_class_ids = [cls for cls in all_class_ids if cls > 0]
    
    # 初始化全局 RR 神经元集合（存储神经元在 segments/labels 中的**列索引**）
    global_rr_enhanced_set = set()
    global_rr_inhibitory_set = set()
    
    # 将 segments 转换为 (n_trials, n_neurons, n_timepoints)
    n_neurons = segments.shape[1]
    
    for class_id in valid_class_ids:
        # 1. 筛选出当前类别的试次
        class_mask = (labels == class_id)
        class_segments = segments[class_mask, :, :]
        
        # 检查试次数量
        if class_segments.shape[0] < 2:
            print(f"警告: 类别 {class_id} 试次数量不足({class_segments.shape[0]})，跳过该类别筛选。")
            continue
            
        # 2. 对当前类别的试次进行 RR 筛选
        # _rr_selection_single 返回的是**当前 class_segments** 中的索引
        rr_exc_local_indices, rr_inh_local_indices = _rr_selection_single(
            class_segments, 
            class_label=str(int(class_id)), 
            **kwargs
        )
        
        # 3. 将结果（局部索引）合并到全局集合中
        # 注意：由于我们是对整个 segments 数组的子集进行操作，
        # _rr_selection_single 返回的索引是针对 segments 数组的**列索引** (即神经元索引)，
        # 因此可以直接合并，无需映射。
        global_rr_enhanced_set.update(rr_exc_local_indices)
        global_rr_inhibitory_set.update(rr_inh_local_indices)

    # 结果转为列表并排序
    rr_enhanced_neurons = sorted(list(global_rr_enhanced_set))
    rr_inhibitory_neurons = sorted(list(global_rr_inhibitory_set))
    
    elapsed_time = time.time() - start_time
    print(f"\n分类别 RR 筛选完成，总耗时: {elapsed_time:.2f}秒")
    print(f"最终筛选结果 (取并集): 兴奋性RR神经元总数: {len(rr_enhanced_neurons)}, 抑制性RR神经元总数: {len(rr_inhibitory_neurons)}")
    
    # 同时返回所有可靠神经元的集合，以备不时之需（但原逻辑中未使用）
    return rr_enhanced_neurons, rr_inhibitory_neurons


# ========== 数据分割函数 (保持不变) ========== 
def segment_neuron_data(neuron_data, trigger_data, label, pre_frames=cfg.exp_info["t_stimulus"], post_frames=cfg.exp_info["l_trials"]-cfg.exp_info["t_stimulus"]):
    """
    改进的数据分割函数
    """
    total_frames = pre_frames + post_frames
    # segment 形状: (n_triggers, n_neurons, n_timepoints)
    segments = np.zeros((len(trigger_data), neuron_data.shape[1], total_frames))
    labels = []

    for i in range(len(trigger_data)): # 遍历每个触发事件
        start = trigger_data[i] - pre_frames
        end = trigger_data[i] + post_frames
        # 边界检查
        if start < 0 or end >= neuron_data.shape[0]:
            print(f"警告: 第{i}个刺激的时间窗口超出边界，跳过")
            continue
        segment = neuron_data[start:end, :]
        segments[i] = segment.T
        labels.append(label[i])
    labels = np.array(labels)
    return segments, labels

# =================================================================
# %% 缓存函数 (修改版，保存keep_idx)
# =================================================================
def save_preprocessed_data_npz(segments, labels, neuron_pos_filtered, keep_idx, file_path):
    """保存预处理中间数据 (segments, labels, filtered_neuron_pos, keep_idx) 到 .npz 文件。"""
    try:
        np.savez_compressed(
            file_path, 
            segments=segments, 
            labels=labels, 
            neuron_pos_filtered=neuron_pos_filtered,
            keep_idx=keep_idx  # 新增
        )
        print(f"已将预处理中间数据保存到缓存文件: {file_path}")
    except Exception as e:
        print(f"保存预处理数据失败: {e}")

def load_preprocessed_data_npz(file_path):
    """从 .npz 文件加载预处理中间数据。"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"尝试从缓存文件加载预处理中间数据: {file_path}")
        return data['segments'], data['labels'], data['neuron_pos_filtered'], data['keep_idx']
    except Exception as e:
        print(f"加载预处理数据失败: {e}")
        return None, None, None, None

# ========== 修改: 保存时间段信息的函数 (支持IC2、IC4、LC2、LC4、基线、空白六个时间段) ==========
def save_stimulus_periods(ic2_time_indices, ic4_time_indices, lc2_time_indices, lc4_time_indices, 
                         baseline_time_indices, blank_screen_indices, file_path):
    """保存IC2、IC4、LC2、LC4、基线、空白屏幕时间段的时间点信息"""
    try:
        # 保存为npy文件，方便后续加载
        np.save(file_path.replace('.mat', '_ic2.npy'), ic2_time_indices)
        np.save(file_path.replace('.mat', '_ic4.npy'), ic4_time_indices)
        np.save(file_path.replace('.mat', '_lc2.npy'), lc2_time_indices)
        np.save(file_path.replace('.mat', '_lc4.npy'), lc4_time_indices)
        np.save(file_path.replace('.mat', '_baseline.npy'), baseline_time_indices)
        np.save(file_path.replace('.mat', '_blank_screen.npy'), blank_screen_indices)
        
        # 同时保存为mat文件，保持兼容性
        scipy.io.savemat(file_path, {
            'ic2_time_indices': ic2_time_indices,
            'ic4_time_indices': ic4_time_indices,
            'lc2_time_indices': lc2_time_indices,
            'lc4_time_indices': lc4_time_indices,
            'baseline_time_indices': baseline_time_indices,
            'blank_screen_indices': blank_screen_indices
        })
        print(f"✅ 时间段信息已保存到: {file_path}")
        print(f"   IC2时间段: {len(ic2_time_indices)}个时间点")
        print(f"   IC4时间段: {len(ic4_time_indices)}个时间点")
        print(f"   LC2时间段: {len(lc2_time_indices)}个时间点")
        print(f"   LC4时间段: {len(lc4_time_indices)}个时间点")
        print(f"   基线时间段: {len(baseline_time_indices)}个时间点")
        print(f"   空白屏幕时间段: {len(blank_screen_indices)}个时间点")
        
        # 打印保存的文件路径
        print(f"   保存的文件:")
        print(f"     - {file_path}")
        print(f"     - {file_path.replace('.mat', '_ic2.npy')}")
        print(f"     - {file_path.replace('.mat', '_ic4.npy')}")
        print(f"     - {file_path.replace('.mat', '_lc2.npy')}")
        print(f"     - {file_path.replace('.mat', '_lc4.npy')}")
        print(f"     - {file_path.replace('.mat', '_baseline.npy')}")
        print(f"     - {file_path.replace('.mat', '_blank_screen.npy')}")
        
    except Exception as e:
        print(f"❌ 保存时间段信息失败: {e}")
        import traceback
        traceback.print_exc()

def load_stimulus_periods(file_path):
    """加载时间段信息"""
    try:
        # 优先加载npy文件
        ic2_file = file_path.replace('.mat', '_ic2.npy')
        ic4_file = file_path.replace('.mat', '_ic4.npy')
        lc2_file = file_path.replace('.mat', '_lc2.npy')
        lc4_file = file_path.replace('.mat', '_lc4.npy')
        baseline_file = file_path.replace('.mat', '_baseline.npy')
        blank_screen_file = file_path.replace('.mat', '_blank_screen.npy')
        
        if os.path.exists(ic2_file):
            ic2_time_indices = np.load(ic2_file)
            ic4_time_indices = np.load(ic4_file)
            lc2_time_indices = np.load(lc2_file)
            lc4_time_indices = np.load(lc4_file)
            baseline_time_indices = np.load(baseline_file)
            blank_screen_indices = np.load(blank_screen_file)
            print(f"✅ 从npy文件加载时间段信息成功")
        else:
            # 回退到mat文件
            data = scipy.io.loadmat(file_path)
            ic2_time_indices = data['ic2_time_indices'].flatten()
            ic4_time_indices = data['ic4_time_indices'].flatten()
            lc2_time_indices = data['lc2_time_indices'].flatten()
            lc4_time_indices = data['lc4_time_indices'].flatten()
            baseline_time_indices = data['baseline_time_indices'].flatten()
            blank_screen_indices = data['blank_screen_indices'].flatten()
            print(f"✅ 从mat文件加载时间段信息成功")
        
        print(f"   IC2时间段: {len(ic2_time_indices)}个时间点")
        print(f"   IC4时间段: {len(ic4_time_indices)}个时间点")
        print(f"   LC2时间段: {len(lc2_time_indices)}个时间点")
        print(f"   LC4时间段: {len(lc4_time_indices)}个时间点")
        print(f"   基线时间段: {len(baseline_time_indices)}个时间点")
        print(f"   空白屏幕时间段: {len(blank_screen_indices)}个时间点")
        
        return ic2_time_indices, ic4_time_indices, lc2_time_indices, lc4_time_indices, baseline_time_indices, blank_screen_indices
        
    except Exception as e:
        print(f"❌ 加载时间段信息失败: {e}")
        return None, None, None, None, None, None

# ========== 修改: 计算空白屏幕时间段的函数 (只取刺激开始前的时间段) ==========
def calculate_blank_screen_periods(start_edges, total_frames, t_stimulus=cfg.exp_info["t_stimulus"], l_stimulus=cfg.exp_info["l_stimulus"]):
    """
    计算空白屏幕时间段（只取刺激开始前的时间段）
    
    参数:
    - start_edges: 刺激开始时间点
    - total_frames: 总帧数
    - t_stimulus: 刺激前帧数
    - l_stimulus: 刺激持续帧数
    
    返回:
    - blank_screen_indices: 空白屏幕时间点的数组
    """
    print("\n🖥️  计算空白屏幕时间段...")
    
    if len(start_edges) == 0:
        # 如果没有刺激，则整个记录都是空白屏幕
        blank_screen_indices = np.arange(total_frames)
        print(f"   没有刺激试次，整个记录都是空白屏幕: {len(blank_screen_indices)}个时间点")
        return blank_screen_indices
    
    # 第一个刺激开始前
    first_stimulus_start = start_edges[0] + t_stimulus
    
    print(f"   第一个刺激开始帧: {first_stimulus_start}")
    print(f"   总帧数: {total_frames}")
    
    # 只取记录开始到第一个刺激开始前的时间段
    blank_screen_indices = list(range(0, first_stimulus_start))
    
    print(f"   空白屏幕时间段: 前{len(blank_screen_indices)}帧 (仅刺激开始前)")
    
    return np.array(blank_screen_indices)

# %% 实际功能函数
# ========== 加载数据 (修改刺激数据加载部分) ==============================
def load_data(data_path = cfg.data_path, start_idx=cfg.trial_info["TRIAL_START_SKIP"], end_idx=cfg.trial_info["TRIAL_START_SKIP"] + cfg.trial_info["TOTAL_TRIALS"]):
    '''
    加载神经数据、位置数据、触发数据和刺激数据
    '''
    ######### 读取神经数据 #########
    print("开始处理数据...")
    mat_file = os.path.join(data_path, 'wholebrain_output.mat')
    if not os.path.exists(mat_file):
        raise ValueError(f"未找到神经数据文件: {mat_file}")
    try:
        data = h5py.File(mat_file, 'r')
    except Exception as e:
        raise ValueError(f"无法读取mat文件: {mat_file}，错误信息: {e}")

    # 检查关键数据集是否存在
    if 'whole_trace_ori' not in data or 'whole_center' not in data:
        raise ValueError("mat文件缺少必要的数据集（'whole_trace_ori' 或 'whole_center'）")

    # ==========神经数据================
    neuron_data = data['whole_trace_ori']
    # 转化成numpy数组
    neuron_data = np.array(neuron_data)
    print(f"原始神经数据形状: {neuron_data.shape}")
    
    # 只做基本的数据清理：移除NaN和Inf
    neuron_data = np.nan_to_num(neuron_data, nan=0.0, posinf=0.0, neginf=0.0)
    neuron_pos = data['whole_center']
    # 检查和处理neuron_pos维度
    if len(neuron_pos.shape) != 2:
        raise ValueError(f"neuron_pos 应为2D数组，实际为: {neuron_pos.shape}")
    
    # 灵活处理不同维度的neuron_pos
    if neuron_pos.shape[0] > 2:
        # 标准格式 (4, n)，提取前两维
        neuron_pos = neuron_pos[0:2, :]
    elif neuron_pos.shape[0] == 2:
        # 已经是2维，直接使用
        print(f"检测到2维neuron_pos格式: {neuron_pos.shape}")
    else:
        raise ValueError(f"不支持的neuron_pos维度: {neuron_pos.shape[0]}，期望为2、3或4维")

    # 触发数据
    trigger_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')])
    # 过滤出名称中包含 trigger 的 txt，确保我们读取正确的触发文件
    trigger_txt_candidates = [f for f in trigger_files if 'trigger' in os.path.basename(f).lower()]
    if not trigger_txt_candidates:
        raise FileNotFoundError(f"在 {data_path} 中未找到包含 'trigger' 字样的触发txt文件。")
    trigger_data = process_trigger(trigger_txt_candidates[0])
    
    # 刺激数据 - 修改部分：从 stim_type.csv 文件加载
    stim_type_file = os.path.join(data_path, "stim_type.csv")
    if os.path.exists(stim_type_file):
        print(f"✅ 从 stim_type.csv 文件加载刺激标签")
        stimulus_df = pd.read_csv(stim_type_file, header=None)
        stimulus_data = stimulus_df.iloc[:, 0].values.astype(str)
        print(f"加载了 {len(stimulus_data)} 个刺激标签")
        print(f"刺激标签示例: {stimulus_data[:10]}")  # 显示前10个标签
    else:
        raise FileNotFoundError(f"在 {data_path} 中未找到 stim_type.csv 文件")
    
    # 保持指定试验数，去掉首尾 - 对触发数据和刺激数据同时处理
    start_edges = trigger_data['start_edge'][start_idx:end_idx]
    # 确保 stimulus_data 和 start_edges 长度一致
    if len(stimulus_data) < (end_idx - start_idx):
        print(f"警告: 刺激数据 ({len(stimulus_data)}个) 少于期望的试次数量 ({end_idx - start_idx}个)。")
        num_trials = min(len(stimulus_data), len(start_edges))
        start_edges = start_edges[:num_trials]
        stimulus_data = stimulus_data[:num_trials]
    else:
        stimulus_data = stimulus_data[start_idx:end_idx] # 使用 start_idx:end_idx 
    
    # 返回原始数据，用于后续的昂贵预处理步骤
    return neuron_data, neuron_pos, start_edges, stimulus_data 


# ========== 预处理的耗时部分：去除负值神经元 + 矫正 + 分割trial (修改版) ==================
def filter_and_segment_data(neuron_data, neuron_pos, start_edge, stimulus_data, cfg=cfg):
    """执行耗时的神经元过滤、dF/F预处理和数据分割步骤。"""

    # =========== 第一步 提取仅有正值的神经元==================
    # 带负值的神经元索引
    mask = np.any(neuron_data <= 0, axis=0)  # 每列是否存在 <=0
    keep_idx = np.where(~mask)[0]

    # 如果 neuron_pos 与 neuron_data 的列对齐，则同步删除对应列
    if neuron_pos.shape[1] == neuron_data.shape[1]:
        neuron_data_filtered = neuron_data[:, keep_idx]
        neuron_pos_filtered = neuron_pos[:, keep_idx]
    else:
        # 如果长度不匹配，理论上应该在 load_data 阶段就报错，这里保留原始逻辑
        raise ValueError(f"警告: neuron_pos 列数({neuron_pos.shape[1]}) 与 neuron_data 列数({neuron_data.shape[1]}) 不匹配，未修改 neuron_pos")
    
    # =========== 第二步 预处理 (dF/F) ===========================
    if cfg.preprocess_cfg["preprocess"]:
        win_size = cfg.preprocess_cfg["win_size"]
        if win_size % 2 == 0:
            win_size += 1
        T, N = neuron_data_filtered.shape
        F0_dynamic = np.zeros((T, N), dtype=float)
        for i in range(N):
            # ndimage.percentile_filter 输出每帧的窗口百分位值
            F0_dynamic[:, i] = ndimage.percentile_filter(neuron_data_filtered[:, i], percentile=8, size=win_size, mode='reflect')
        # 计算 dF/F（逐帧）
        dff = (neuron_data_filtered - F0_dynamic) / F0_dynamic
    else:
        dff = neuron_data_filtered

    # =========== 第三步 分割神经数据 =====================================
    labels = reclassify(stimulus_data)
    segments, labels = segment_neuron_data(dff, start_edge, labels)

    return segments, labels, neuron_pos_filtered, keep_idx  # 返回 keep_idx

# ========== 修改: 分别计算IC2、IC4、LC2、LC4、基线时间段 ==========
def calculate_stimulus_periods(start_edges, stimulus_data, total_frames, cfg=cfg):
    """
    分别计算IC2、IC4、LC2、LC4、基线时间段的时间点集合
    
    参数:
    - start_edges: 刺激开始时间点
    - stimulus_data: 刺激标签数据
    - total_frames: 总帧数
    - cfg: 配置对象
    
    返回:
    - ic2_time_indices: IC2刺激期间的所有时间点
    - ic4_time_indices: IC4刺激期间的所有时间点
    - lc2_time_indices: LC2刺激期间的所有时间点
    - lc4_time_indices: LC4刺激期间的所有时间点
    - baseline_time_indices: 基线期间的所有时间点
    """
    print("\n🔍 开始详细调试时间段计算...")
    
    t_stimulus = cfg.exp_info["t_stimulus"]  # 刺激前帧数
    l_stimulus = cfg.exp_info["l_stimulus"]  # 刺激持续帧数
    
    print(f"配置参数: t_stimulus={t_stimulus}, l_stimulus={l_stimulus}")
    print(f"总帧数: {total_frames}")
    print(f"刺激开始时间点数量: {len(start_edges)}")
    print(f"刺激标签数量: {len(stimulus_data)}")
    
    # 使用重新分类后的标签
    labels = reclassify(stimulus_data)
    
    ic2_time_indices = []
    ic4_time_indices = []
    lc2_time_indices = []
    lc4_time_indices = []
    
    # 详细检查每个试次
    ic2_count = 0
    ic4_count = 0
    lc2_count = 0
    lc4_count = 0
    other_count = 0
    
    for i, (start_frame, label) in enumerate(zip(start_edges, labels)):
        stimulus_start = start_frame + t_stimulus
        stimulus_end = stimulus_start + l_stimulus
        
        # 检查是否超出范围
        if stimulus_end > total_frames:
            print(f"警告: 第{i}个试次的刺激期超出数据范围，跳过")
            continue
        
        stimulus_period = list(range(stimulus_start, stimulus_end))
        
        # 使用重新分类后的数字标签进行判断
        if label == 1:  # IC2 -> 1
            ic2_time_indices.extend(stimulus_period)
            ic2_count += 1
        elif label == 2:  # IC4 -> 2
            ic4_time_indices.extend(stimulus_period)
            ic4_count += 1
        elif label == 3:  # LC2 -> 3
            lc2_time_indices.extend(stimulus_period)
            lc2_count += 1
        elif label == 4:  # LC4 -> 4
            lc4_time_indices.extend(stimulus_period)
            lc4_count += 1
        else:
            other_count += 1
    
    print(f"\n📊 时间段统计:")
    print(f"  IC2试次: {ic2_count} 个")
    print(f"  IC4试次: {ic4_count} 个")
    print(f"  LC2试次: {lc2_count} 个")
    print(f"  LC4试次: {lc4_count} 个")
    print(f"  其他试次: {other_count} 个")
    print(f"  IC2时间段: {len(ic2_time_indices)}个时间点")
    print(f"  IC4时间段: {len(ic4_time_indices)}个时间点")
    print(f"  LC2时间段: {len(lc2_time_indices)}个时间点")
    print(f"  LC4时间段: {len(lc4_time_indices)}个时间点")
    
    # 计算基线时间段: 第一个刺激开始后、最后一个刺激结束前，但不属于任何刺激的时间点
    if len(start_edges) > 0:
        # 第一个刺激开始时间 (第一个试次的刺激开始)
        first_stimulus_start = min(start_edges) + t_stimulus
        
        # 最后一个刺激结束时间 (最后一个试次的刺激结束)
        last_stimulus_end = max(start_edges) + t_stimulus + l_stimulus
        
        print(f"\n基线计算:")
        print(f"  第一个刺激开始: {first_stimulus_start}")
        print(f"  最后一个刺激结束: {last_stimulus_end}")
        
        # 所有刺激时间点的并集
        all_stimulus_indices = set(ic2_time_indices) | set(ic4_time_indices) | set(lc2_time_indices) | set(lc4_time_indices)
        baseline_time_indices = []
        for frame in range(first_stimulus_start, last_stimulus_end):
            if frame not in all_stimulus_indices:
                baseline_time_indices.append(frame)
    else:
        print("警告: 没有刺激试次，无法计算基线时间段")
        baseline_time_indices = []
    
    baseline_time_indices = sorted(baseline_time_indices)
    
    print(f"  基线时间段: {len(baseline_time_indices)}个时间点")
    
    # 验证时间点没有重叠
    ic2_set = set(ic2_time_indices)
    ic4_set = set(ic4_time_indices)
    lc2_set = set(lc2_time_indices)
    lc4_set = set(lc4_time_indices)
    baseline_set = set(baseline_time_indices)
    
    # 检查所有可能的重叠
    if ic2_set & ic4_set:
        print(f"警告: IC2和IC4时间段有重叠: {len(ic2_set & ic4_set)}个时间点")
    if ic2_set & lc2_set:
        print(f"警告: IC2和LC2时间段有重叠: {len(ic2_set & lc2_set)}个时间点")
    if ic2_set & lc4_set:
        print(f"警告: IC2和LC4时间段有重叠: {len(ic2_set & lc4_set)}个时间点")
    if ic4_set & lc2_set:
        print(f"警告: IC4和LC2时间段有重叠: {len(ic4_set & lc2_set)}个时间点")
    if ic4_set & lc4_set:
        print(f"警告: IC4和LC4时间段有重叠: {len(ic4_set & lc4_set)}个时间点")
    if lc2_set & lc4_set:
        print(f"警告: LC2和LC4时间段有重叠: {len(lc2_set & lc4_set)}个时间点")
    
    if ic2_set & baseline_set:
        print(f"警告: IC2和基线时间段有重叠: {len(ic2_set & baseline_set)}个时间点")
    if ic4_set & baseline_set:
        print(f"警告: IC4和基线时间段有重叠: {len(ic4_set & baseline_set)}个时间点")
    if lc2_set & baseline_set:
        print(f"警告: LC2和基线时间段有重叠: {len(lc2_set & baseline_set)}个时间点")
    if lc4_set & baseline_set:
        print(f"警告: LC4和基线时间段有重叠: {len(lc4_set & baseline_set)}个时间点")
    
    return (np.array(ic2_time_indices), np.array(ic4_time_indices), 
            np.array(lc2_time_indices), np.array(lc4_time_indices), 
            np.array(baseline_time_indices))

# %% 特殊函数（和刺激类型等相关）
def reclassify(stimulus_data):
    '''
    刺激重新分类函数：将字符串标签转换为数值类别。
    IC2->1, IC4->2, LC2->3, LC4->4
    '''
    mapping = {
        'IC2': 1,  # 类别 1
        'IC4': 2,  # 类别 2
        'LC2': 3,  # 类别 3
        'LC4': 4,  # 类别 4
    }
    
    new_labels = []
    unknown_labels = set()
    
    for label in stimulus_data:
        clean_label = str(label).strip()  # 清理空格
        mapped_label = mapping.get(clean_label, 0)
        new_labels.append(mapped_label)
        
        if mapped_label == 0 and clean_label not in unknown_labels:
            unknown_labels.add(clean_label)
            print(f"警告: 未知刺激标签 '{clean_label}'，映射为类别 0")
    
    if unknown_labels:
        print(f"发现 {len(unknown_labels)} 个未知标签: {unknown_labels}")
    
    print(f"重新分类统计:")
    for key, value in mapping.items():
        count = sum(1 for label in new_labels if label == value)
        print(f"  {key} -> 类别 {value}: {count} 个试次")
    
    return np.array(new_labels)

# %% =============  主程序逻辑 (修改为调用 rr_selection_by_class) =============================
if __name__ == "__main__":
    print("开始运行主程序")

    # 定义缓存文件路径
    cache_file = os.path.join(cfg.data_path, "preprocessed_data_cache.npz") 
    print(f"预处理数据缓存文件路径: {cache_file}")

    # 定义时间段文件路径
    periods_file = os.path.join(cfg.data_path, "stimulus_periods.mat")
    print(f"时间段信息文件路径: {periods_file}")

    # 1. 尝试加载缓存数据
    segments, labels, neuron_pos_filtered, keep_idx = None, None, None, None
    load_from_cache_successful = False
    
    if os.path.exists(cache_file):
        segments_cached, labels_cached, neuron_pos_filtered_cached, keep_idx_cached = load_preprocessed_data_npz(cache_file)
        if segments_cached is not None:
              segments = segments_cached
              labels = labels_cached
              neuron_pos_filtered = neuron_pos_filtered_cached
              keep_idx = keep_idx_cached
              load_from_cache_successful = True
              print("缓存加载成功，跳过原始数据加载和预处理步骤。")
    else:
        print("未找到缓存文件，需要执行完整的加载和预处理流程...")

    # 2. 如果缓存加载失败，执行完整的加载和预处理流程
    if not load_from_cache_successful:
        print("执行完整的加载和预处理流程...")
        
        # 2a. 加载原始数据 (.mat, .txt, .csv)
        neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
        
        # 2b. 执行昂贵的预处理和分割步骤
        segments, labels, neuron_pos_filtered, keep_idx = filter_and_segment_data(
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data, cfg
        )
        
        # 2c. 保存缓存
        save_preprocessed_data_npz(segments, labels, neuron_pos_filtered, keep_idx, cache_file)
        
        # 2d. 计算并保存时间段信息
        print("\n计算IC2、IC4、LC2、LC4、基线、空白屏幕时间段...")
        total_frames = neuron_data_orig.shape[0]  # 连续记录的总帧数
        
        # 计算原有的五个时间段
        ic2_time_indices, ic4_time_indices, lc2_time_indices, lc4_time_indices, baseline_time_indices = calculate_stimulus_periods(
            start_edges, stimulus_data, total_frames, cfg
        )
        
        # 新增：计算空白屏幕时间段 (只取刺激开始前)
        blank_screen_indices = calculate_blank_screen_periods(
            start_edges, total_frames, cfg.exp_info["t_stimulus"], cfg.exp_info["l_stimulus"]
        )
        
        # 保存时间段信息（现在包含六个时间段）
        save_stimulus_periods(ic2_time_indices, ic4_time_indices, lc2_time_indices, lc4_time_indices, 
                             baseline_time_indices, blank_screen_indices, periods_file)

    # 3. 如果缓存加载成功，检查时间段信息文件是否存在且完整，如果不存在或不完整则重新计算
    else:
        print("缓存加载成功，检查时间段信息文件...")
        
        # 检查所有必需的时间段文件是否存在
        required_files = [
            "stimulus_periods.mat",
            "stimulus_periods_ic2.npy",
            "stimulus_periods_ic4.npy",
            "stimulus_periods_lc2.npy",
            "stimulus_periods_lc4.npy",
            "stimulus_periods_baseline.npy",
            "stimulus_periods_blank_screen.npy"
        ]
        
        all_files_exist = all(os.path.exists(os.path.join(cfg.data_path, f)) for f in required_files)
        
        if not all_files_exist:
            print("时间段信息文件不完整，需要重新计算...")
            
            # 重新加载原始数据来计算时间段
            neuron_data_orig, neuron_pos_orig, start_edges, stimulus_data = load_data()
            total_frames = neuron_data_orig.shape[0]
            
            # 计算原有的五个时间段
            ic2_time_indices, ic4_time_indices, lc2_time_indices, lc4_time_indices, baseline_time_indices = calculate_stimulus_periods(
                start_edges, stimulus_data, total_frames, cfg
            )
            
            # 新增：计算空白屏幕时间段 (只取刺激开始前)
            blank_screen_indices = calculate_blank_screen_periods(
                start_edges, total_frames, cfg.exp_info["t_stimulus"], cfg.exp_info["l_stimulus"]
            )
            
            # 保存时间段信息（现在包含六个时间段）
            save_stimulus_periods(ic2_time_indices, ic4_time_indices, lc2_time_indices, lc4_time_indices, 
                                 baseline_time_indices, blank_screen_indices, periods_file)
        else:
            print("时间段信息文件已存在且完整，跳过计算。")


    # 4. RR 神经元筛选 (使用分类别筛选并取并集的新逻辑)
    
    rr_enhanced_neurons, rr_inhibitory_neurons = rr_selection_by_class(segments, np.array(labels))
    rr_enhanced_neurons = np.array(sorted(set(rr_enhanced_neurons)), dtype=int)
    rr_inhibitory_neurons = np.array(sorted(set(rr_inhibitory_neurons)), dtype=int)
    
    # 提取兴奋性 RR 神经元的数据
    enhanced_segments = segments[:, rr_enhanced_neurons, :] if rr_enhanced_neurons.size > 0 else np.empty((segments.shape[0], 0, segments.shape[2]))
    enhanced_neuron_pos_rr = neuron_pos_filtered[:, rr_enhanced_neurons] if rr_enhanced_neurons.size > 0 else np.empty((2, 0))
    print(f"\n兴奋性 RR 神经元: {len(rr_enhanced_neurons)} 个, 位置数据形状: {enhanced_neuron_pos_rr.shape}")

    # 提取抑制性 RR 神经元的数据
    inhibitory_segments = segments[:, rr_inhibitory_neurons, :] if rr_inhibitory_neurons.size > 0 else np.empty((segments.shape[0], 0, segments.shape[2]))
    inhibitory_neuron_pos_rr = neuron_pos_filtered[:, rr_inhibitory_neurons] if rr_inhibitory_neurons.size > 0 else np.empty((2, 0))
    print(f"抑制性 RR 神经元: {len(rr_inhibitory_neurons)} 个, 位置数据形状: {inhibitory_neuron_pos_rr.shape}")

    # ========== 新增：保存原始神经元索引 ==========
    print(f"\n兴奋性 RR 神经元相对索引总数 {len(rr_enhanced_neurons)}: {rr_enhanced_neurons.tolist()}")
    print(f"抑制性 RR 神经元相对索引总数 {len(rr_inhibitory_neurons)}: {rr_inhibitory_neurons.tolist()}")
    
    # 将相对索引转换为原始索引
    rr_enhanced_original = keep_idx[rr_enhanced_neurons] if rr_enhanced_neurons.size > 0 else np.array([], dtype=int)
    rr_inhibitory_original = keep_idx[rr_inhibitory_neurons] if rr_inhibitory_neurons.size > 0 else np.array([], dtype=int)
    
    print(f"兴奋性 RR 神经元原始索引: {rr_enhanced_original.tolist()}")
    print(f"抑制性 RR 神经元原始索引: {rr_inhibitory_original.tolist()}")
    
    # 保存原始索引到文件
    rr_index_path_original = os.path.join(cfg.data_path, "rr_neuron_original_indices.csv")
    rr_original_df = pd.DataFrame({
        "neuron_index": np.concatenate([rr_enhanced_original, rr_inhibitory_original]),
        "category": (["exc"] * len(rr_enhanced_original)) + (["inh"] * len(rr_inhibitory_original))
    })
    rr_original_df.to_csv(rr_index_path_original, index=False, encoding="utf-8-sig")
    print(f"RR 神经元原始索引已保存到: {rr_index_path_original}")
    
    # 同时保存相对索引（向后兼容）
    rr_index_path = os.path.join(cfg.data_path, "rr_neuron_indices.csv")
    rr_index_df = pd.DataFrame({
        "neuron_index": np.concatenate([rr_enhanced_neurons, rr_inhibitory_neurons]),
        "category": (["exc"] * len(rr_enhanced_neurons)) + (["inh"] * len(rr_inhibitory_neurons))
    })
    rr_index_df.to_csv(rr_index_path, index=False, encoding="utf-8-sig")
    print(f"RR 神经元相对索引已保存到: {rr_index_path}")
    
    # 保存并集的原始索引供SVM使用
    rr_union_original = np.concatenate([rr_enhanced_original, rr_inhibitory_original])
    np.save(os.path.join(cfg.data_path, "all_stimuli_rr_union_original_indices.npy"), rr_union_original)
    print(f"RR 神经元并集原始索引已保存到: {os.path.join(cfg.data_path, 'all_stimuli_rr_union_original_indices.npy')}")
    
    # 分别保存兴奋性和抑制性的原始索引
    np.save(os.path.join(cfg.data_path, "all_stimuli_rr_excitatory_original_indices.npy"), rr_enhanced_original)
    np.save(os.path.join(cfg.data_path, "all_stimuli_rr_inhibitory_original_indices.npy"), rr_inhibitory_original)
    print(f"兴奋性和抑制性 RR 神经元原始索引已分别保存")

    print("\n🎉 RR筛选完成！所有时间段信息已保存，可用于网络分析。")
    
    # 最后再次检查时间段文件是否生成
    print("\n检查时间段文件生成情况:")
    files_to_check = [
        "stimulus_periods.mat",
        "stimulus_periods_ic2.npy",
        "stimulus_periods_ic4.npy",
        "stimulus_periods_lc2.npy",
        "stimulus_periods_lc4.npy",
        "stimulus_periods_baseline.npy",
        "stimulus_periods_blank_screen.npy"
    ]
    
    for file in files_to_check:
        file_path = os.path.join(cfg.data_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")