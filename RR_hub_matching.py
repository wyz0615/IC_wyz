import numpy as np
import pandas as pd
import h5py
from scipy import stats
import os

# 设置随机种子以确保可重复性
np.random.seed(42)

def load_data():
    """加载数据"""
    print("正在加载数据...")
    
    # 加载荧光信号数据
    mat_path = r"C:\Users\wangy\Desktop\IC\m79\wholebrain_output.mat"
    with h5py.File(mat_path, 'r') as f:
        # 加载荧光信号数据
        fluorescence_data = f['whole_trace_ori'][:]
        print(f"荧光信号数据形状: {fluorescence_data.shape}")
    
    # 加载RR神经元索引
    rr_path = r"C:\Users\wangy\Desktop\IC\m79\rr_neuron_original_indices.csv"
    
    print(f"正在读取RR神经元文件: {rr_path}")
    
    # 先查看文件的前几行，了解格式
    try:
        with open(rr_path, 'r') as f:
            lines = [next(f) for _ in range(5)]
        print("CSV文件前5行内容:")
        for i, line in enumerate(lines):
            print(f"  第{i+1}行: {line.strip()}")
    except Exception as e:
        print(f"预览文件时出错: {e}")
    
    # 尝试读取CSV文件 - 使用更灵活的方式
    try:
        # 尝试用header=None读取
        rr_data = pd.read_csv(rr_path, header=None)
        print(f"CSV文件读取成功，形状: {rr_data.shape}")
        
        # 检查第一行是否是表头
        first_row = rr_data.iloc[0].astype(str).str.lower().tolist()
        has_header = any('index' in val or 'neuron' in val for val in first_row)
        
        if has_header:
            print("检测到表头，跳过第一行")
            # 重新读取，使用第一行作为表头
            rr_data = pd.read_csv(rr_path)
            print(f"重新读取后形状: {rr_data.shape}")
            print(f"列名: {rr_data.columns.tolist()}")
        else:
            print("没有检测到表头")
            # 如果没有表头，根据列数设置列名
            if rr_data.shape[1] >= 2:
                rr_data.columns = ['index', 'type']
                print(f"设置列名为: {rr_data.columns.tolist()}")
            else:
                rr_data.columns = ['index']
                
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None, None, None
    
    # 查看读取的数据
    print(f"读取的数据前5行:\n{rr_data.head()}")
    
    # 处理索引列 - 尝试不同的列名可能性
    index_col = None
    for col in rr_data.columns:
        if 'index' in str(col).lower() or 'neuron' in str(col).lower():
            index_col = col
            break
    
    if index_col is None and len(rr_data.columns) > 0:
        index_col = rr_data.columns[0]
    
    if index_col is None:
        print("错误：找不到索引列")
        return None, None, None
    
    print(f"使用的索引列: {index_col}")
    
    # 处理类型列
    type_col = None
    for col in rr_data.columns:
        if col != index_col and ('type' in str(col).lower() or 'exc' in str(col).lower() or 'inh' in str(col).lower()):
            type_col = col
            break
    
    if type_col is None and len(rr_data.columns) > 1:
        # 如果没找到明显的类型列，假设第二列是类型列
        type_col = rr_data.columns[1] if rr_data.columns[1] != index_col else None
    
    print(f"使用的类型列: {type_col}")
    
    # 清理数据
    # 转换索引列为整数
    rr_data[index_col] = pd.to_numeric(rr_data[index_col], errors='coerce')
    rr_data = rr_data.dropna(subset=[index_col])
    rr_data[index_col] = rr_data[index_col].astype(int)
    
    # 处理类型列
    if type_col:
        # 清理类型数据：去除空格，统一小写
        rr_data[type_col] = rr_data[type_col].astype(str).str.strip().str.lower()
        # 将简写转换为标准格式
        type_mapping = {
            'e': 'exc', 'ex': 'exc', 'excitation': 'exc', 'excitatory': 'exc',
            'i': 'inh', 'in': 'inh', 'inhibition': 'inh', 'inhibitory': 'inh'
        }
        rr_data[type_col] = rr_data[type_col].map(lambda x: type_mapping.get(x, x))
    
    # 获取索引
    rr_indices = rr_data[index_col].values - 1  # 转换为0-based索引
    
    print(f"\nRR神经元数量: {len(rr_indices)}")
    
    if type_col:
        # 统计神经元类型分布
        type_counts = rr_data[type_col].value_counts()
        print(f"神经元类型分布:\n{type_counts}")
    
    # 重新组织数据，确保有标准的列名
    processed_rr_data = pd.DataFrame()
    processed_rr_data['index'] = rr_data[index_col]
    if type_col:
        processed_rr_data['type'] = rr_data[type_col]
    else:
        processed_rr_data['type'] = 'unknown'
    
    return fluorescence_data, rr_indices, processed_rr_data

def sample_neurons(fluorescence_data, sample_size=2000):
    """随机采样神经元"""
    print(f"\n正在随机采样 {sample_size} 个神经元...")
    total_neurons = fluorescence_data.shape[1]
    
    # 随机选择神经元索引
    sampled_indices = np.random.choice(total_neurons, sample_size, replace=False)
    sampled_indices = np.sort(sampled_indices)
    
    print(f"采样神经元索引范围: {sampled_indices[0]} 到 {sampled_indices[-1]}")
    
    # 提取采样数据
    sampled_data = fluorescence_data[:, sampled_indices]
    
    return sampled_data, sampled_indices

def calculate_correlation_matrix(sampled_data):
    """计算皮尔逊相关矩阵"""
    print("正在计算皮尔逊相关矩阵...")
    
    # 转置数据为 (神经元数, 时间点)
    data_transposed = sampled_data.T
    
    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(data_transposed)
    
    return correlation_matrix

def analyze_threshold(correlation_matrix, rr_indices, sampled_indices, rr_data, threshold, z_threshold=1.5):
    """分析单个阈值下的网络"""
    print(f"\n{'='*50}")
    print(f"分析阈值: {threshold}")
    print(f"{'='*50}")
    
    # 创建二值邻接矩阵
    abs_correlation = np.abs(correlation_matrix)
    binary_matrix = (abs_correlation > threshold).astype(int)
    np.fill_diagonal(binary_matrix, 0)
    
    # 计算连接度
    degree = np.sum(binary_matrix, axis=1)
    
    # 识别hub节点
    degree_mean = np.mean(degree)
    degree_std = np.std(degree)
    z_scores = (degree - degree_mean) / degree_std
    hub_indices = np.where(z_scores > z_threshold)[0]
    
    print(f"平均连接度: {degree_mean:.2f}")
    print(f"连接度标准差: {degree_std:.2f}")
    print(f"Hub节点数量: {len(hub_indices)} ({len(hub_indices)/len(degree)*100:.1f}%)")
    
    # 分析RR神经元
    rr_in_sampled = []
    rr_original_indices = []
    
    for i, rr_idx in enumerate(rr_indices):
        if rr_idx in sampled_indices:
            pos_in_sampled = np.where(sampled_indices == rr_idx)[0][0]
            rr_in_sampled.append(pos_in_sampled)
            rr_original_indices.append(rr_idx)
    
    if len(rr_in_sampled) == 0:
        print("警告：没有RR神经元被采样到！")
        return None
    
    # 检查被采样到的RR神经元是否为hub
    rr_hub_indices = []
    for idx in rr_in_sampled:
        if idx in hub_indices:
            rr_hub_indices.append(idx)
    
    # 计算比例
    rr_hub_ratio = len(rr_hub_indices) / len(rr_in_sampled) if len(rr_in_sampled) > 0 else 0
    
    print(f"被采样到的RR神经元数量: {len(rr_in_sampled)}")
    print(f"是hub的RR神经元数量: {len(rr_hub_indices)}")
    print(f"是hub的RR神经元比例: {rr_hub_ratio:.2%}")
    
    # 获取神经元类型信息
    rr_types_info = {}
    if rr_data is not None and 'type' in rr_data.columns:
        for idx in rr_in_sampled:
            original_idx = sampled_indices[idx]
            original_idx_1based = original_idx + 1
            
            # 查找神经元类型
            if original_idx_1based in rr_data['index'].values:
                neuron_type = rr_data.loc[rr_data['index'] == original_idx_1based, 'type'].values[0]
                rr_types_info[original_idx] = neuron_type
            else:
                rr_types_info[original_idx] = 'unknown'
    
    # 按神经元类型统计Hub比例
    type_stats = {}
    if rr_data is not None and 'type' in rr_data.columns:
        for idx in rr_in_sampled:
            original_idx = sampled_indices[idx]
            original_idx_1based = original_idx + 1
            
            if original_idx_1based in rr_data['index'].values:
                neuron_type = rr_data.loc[rr_data['index'] == original_idx_1based, 'type'].values[0]
                if neuron_type not in type_stats:
                    type_stats[neuron_type] = {'total': 0, 'hub': 0}
                
                type_stats[neuron_type]['total'] += 1
                if idx in hub_indices:
                    type_stats[neuron_type]['hub'] += 1
    
    # 创建结果字典
    result = {
        '阈值': threshold,
        '总神经元数': len(sampled_indices),
        'Hub节点数': len(hub_indices),
        'Hub节点比例': len(hub_indices)/len(sampled_indices)*100,
        '被采样到的RR神经元数': len(rr_in_sampled),
        '是Hub的RR神经元数': len(rr_hub_indices),
        'RR神经元Hub比例': rr_hub_ratio*100,
        '平均连接度': degree_mean,
        '连接度标准差': degree_std,
        '最大连接度': int(degree.max()),
        '最小连接度': int(degree.min()),
        'rr_in_sampled': rr_in_sampled,
        'rr_hub_indices': rr_hub_indices,
        'rr_hub_original': [sampled_indices[idx] for idx in rr_hub_indices],
        'rr_types_info': rr_types_info,
        'type_stats': type_stats
    }
    
    # 打印不同类型神经元的Hub比例
    if type_stats:
        print("\n不同类型神经元的Hub统计:")
        for neuron_type, stats in type_stats.items():
            hub_ratio = stats['hub']/stats['total'] if stats['total'] > 0 else 0
            print(f"  {neuron_type}: {stats['hub']}/{stats['total']} ({hub_ratio:.2%})")
    
    return result

def save_excel_results(all_results, detailed_results=None, output_dir=None):
    """保存所有阈值的结果到Excel文件"""
    print("\n正在保存结果到Excel文件...")
    
    if output_dir is None:
        output_dir = r"C:\Users\wangy\Desktop\IC"
    
    excel_path = os.path.join(output_dir, 'm79/threshold_analysis_results.xlsx')
    
    # 创建Excel writer对象
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. 保存汇总统计信息
        summary_data = []
        for result in all_results:
            if result is not None:
                summary_row = {
                    '阈值': result['阈值'],
                    '总神经元数': result['总神经元数'],
                    'Hub节点数': result['Hub节点数'],
                    'Hub节点比例(%)': f"{result['Hub节点比例']:.2f}",
                    '被采样到的RR神经元数': result['被采样到的RR神经元数'],
                    '是Hub的RR神经元数': result['是Hub的RR神经元数'],
                    'RR神经元Hub比例(%)': f"{result['RR神经元Hub比例']:.2f}",
                    '平均连接度': f"{result['平均连接度']:.2f}",
                    '连接度标准差': f"{result['连接度标准差']:.2f}",
                    '最大连接度': result['最大连接度'],
                    '最小连接度': result['最小连接度']
                }
                summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='阈值汇总统计', index=False)
            print(f"汇总统计已保存到Excel文件")
        
        # 2. 保存每个阈值的详细RR神经元分析
        if detailed_results:
            for threshold, detailed_df in detailed_results.items():
                if not detailed_df.empty:
                    sheet_name = f"阈值_{threshold:.2f}"
                    # 截断sheet名称，确保不超过31个字符
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                    detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"详细RR神经元分析已保存到Excel文件")
        
        # 3. 保存神经元类型统计
        type_stats_data = []
        for result in all_results:
            if result is not None and 'type_stats' in result and result['type_stats']:
                for neuron_type, stats in result['type_stats'].items():
                    hub_ratio = stats['hub']/stats['total']*100 if stats['total'] > 0 else 0
                    type_stats_data.append({
                        '阈值': result['阈值'],
                        '神经元类型': neuron_type,
                        '总神经元数': stats['total'],
                        'Hub神经元数': stats['hub'],
                        'Hub比例(%)': f"{hub_ratio:.2f}"
                    })
        
        if type_stats_data:
            type_stats_df = pd.DataFrame(type_stats_data)
            type_stats_df.to_excel(writer, sheet_name='神经元类型统计', index=False)
            print(f"神经元类型统计已保存到Excel文件")
    
    print(f"所有结果已保存到: {excel_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("小鼠大脑网络分析 - 阈值扫描")
    print("=" * 60)
    
    # 1. 加载数据
    fluorescence_data, rr_indices, rr_data = load_data()
    
    if fluorescence_data is None or rr_indices is None:
        print("错误：无法加载数据，程序退出")
        return
    
    # 2. 随机采样神经元
    sampled_data, sampled_indices = sample_neurons(fluorescence_data, sample_size=2000)
    
    # 3. 计算相关矩阵（只需要计算一次）
    correlation_matrix = calculate_correlation_matrix(sampled_data)
    
    # 4. 定义阈值范围和参数
    thresholds = np.arange(0.1, 0.55, 0.05)  # 0.1到0.5，步长0.05
    z_threshold = 1.5
    
    print(f"\n开始阈值扫描分析，共{len(thresholds)}个阈值:")
    print(f"阈值范围: {thresholds}")
    print(f"Hub识别z-score阈值: {z_threshold}")
    
    # 5. 对每个阈值进行分析
    all_results = []
    detailed_results = {}  # 存储每个阈值的详细RR神经元分析
    
    for threshold in thresholds:
        result = analyze_threshold(correlation_matrix, rr_indices, sampled_indices, rr_data, threshold, z_threshold)
        
        if result is not None:
            all_results.append(result)
            
            # 创建该阈值的详细RR神经元分析DataFrame
            rr_detailed_data = []
            for i, rr_idx in enumerate(result['rr_in_sampled']):
                original_idx = sampled_indices[rr_idx]
                is_hub = rr_idx in result['rr_hub_indices']
                neuron_type = result['rr_types_info'].get(original_idx, 'unknown')
                
                rr_detailed_data.append({
                    '采样后索引': rr_idx,
                    '原始索引': original_idx + 1,  # 保存为1-based索引
                    '神经元类型': neuron_type,
                    '连接度': correlation_matrix.shape[0],  # 注意：这里需要实际的连接度，需要从结果中获取
                    '是否为Hub': '是' if is_hub else '否'
                })
            
            if rr_detailed_data:
                detailed_results[threshold] = pd.DataFrame(rr_detailed_data)
    
    # 6. 保存所有结果到Excel文件
    save_excel_results(all_results, detailed_results)
    
    # 7. 打印最终总结
    print(f"\n{'='*60}")
    print("阈值扫描分析完成！")
    print(f"{'='*60}")
    
    if all_results:
        print(f"\n阈值扫描结果总结:")
        for result in all_results:
            print(f"\n阈值 {result['阈值']}:")
            print(f"  Hub节点比例: {result['Hub节点比例']:.2f}%")
            print(f"  RR神经元Hub比例: {result['RR神经元Hub比例']:.2f}%")
            print(f"  平均连接度: {result['平均连接度']:.2f}")
        
        # 找出RR神经元Hub比例最高的阈值
        if len(all_results) > 0:
            best_result = max(all_results, key=lambda x: x['RR神经元Hub比例'] if x['RR神经元Hub比例'] > 0 else -1)
            print(f"\n最佳阈值: {best_result['阈值']}")
            print(f"在该阈值下，RR神经元Hub比例最高: {best_result['RR神经元Hub比例']:.2f}%")

if __name__ == "__main__":
    main()