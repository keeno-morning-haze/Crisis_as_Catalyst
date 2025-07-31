import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import os
import json
import re
import sys
import openai
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# 获取主文件夹路径
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(main_dir)

import api

# 定义统计分析函数
def perform_statistical_analysis(raw_data_file, analysis_dir, results_file):
    """
    执行单位分配的统计分析
    :param raw_data_file：原始数据文件路径
    :param analysis_dir: 分析结果目录
    :param results_file: 结果文件路径
    """
    # 读取数据
    raw_data = pd.read_csv(raw_data_file)

    # 对每个model分别进行分析
    models = raw_data['model'].unique()
    
    # 创建分析结果目录
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # 打开文件进行写入
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('============= 单位分配统计分析结果 =============\n\n')
        
        for model in models:
            f.write(f'□ 模型: {model}\n')
            f.write('=' * 50 + '\n\n')
            
            model_data = raw_data[raw_data['model'] == model].copy()
            
            # 计算标准化得分
            # 获取Day开头的列
            day_columns = [col for col in model_data.columns if col.startswith('Day')]
            
            # 计算r_Blue：统计每行中取值为1-6的天数在总天数中的排序总和
            def get_r_blue(row):
                # 获取蓝色单位(1-6)的天数
                blue_days = [(i+1, val) for i, val in enumerate(row[day_columns]) if 1 <= val <= 6]
                # 计算这些天数的排序总和
                return sum(day_idx for day_idx, _ in blue_days)
                
            r_blue = model_data.apply(get_r_blue, axis=1)
            model_data['normalized_score'] = 1 - 2 * ((r_blue - 21) / 36)
            
            # 描述统计
            f.write('■ 描述统计:\n')
            f.write('-' * 50 + '\n\n')
            
            for scenario in ['A', 'B', 'C', 'D']:
                scenario_data = model_data[model_data['scenario'] == scenario]
                description = scenario_data['normalized_score'].describe()
                f.write(f'● 场景 {scenario}:\n')
                
                # 格式化输出描述统计
                stats = description.to_dict()
                f.write(f'  样本数: {stats["count"]:.0f}\n')
                f.write(f'  平均值: {stats["mean"]:.4f}\n')
                f.write(f'  标准差: {stats["std"]:.4f}\n')
                f.write(f'  最小值: {stats["min"]:.4f}\n')
                f.write(f'  Q1值:   {stats["25%"]:.4f}\n')
                f.write(f'  中位数: {stats["50%"]:.4f}\n')
                f.write(f'  Q3值:   {stats["75%"]:.4f}\n')
                f.write(f'  最大值: {stats["max"]:.4f}\n\n')
            
            # Friedman检验
            scenario_scores = []
            for scenario in ['A', 'B', 'C', 'D']:
                scenario_scores.append(
                    model_data[model_data['scenario'] == scenario]['normalized_score'].values
                )
            
            # 检查场景数据是否足够
            if all(len(scores) >= 3 for scores in scenario_scores):
                friedman_result = friedmanchisquare(*scenario_scores)
                f.write('■ Friedman检验结果:\n')
                f.write('-' * 50 + '\n\n')
                f.write(f'  统计量 = {friedman_result.statistic:.4f}\n')
                f.write(f'  p值    = {friedman_result.pvalue:.4f}\n')
                
                # 显示检验结果的解释
                if friedman_result.pvalue < 0.05:
                    f.write(f'  结论: 在α=0.05的显著性水平下，拒绝原假设，各场景之间存在显著差异\n\n')
                else:
                    f.write(f'  结论: 在α=0.05的显著性水平下，未能拒绝原假设，无法证明各场景之间存在显著差异\n\n')
            else:
                f.write('■ Friedman检验结果:\n')
                f.write('-' * 50 + '\n\n')
                f.write('  样本量不足，无法进行Friedman检验（至少需要3个样本）\n\n')
            
            # Wilcoxon符号秩检验
            f.write('■ Wilcoxon符号秩检验结果:\n')
            f.write('-' * 50 + '\n\n')
            
            scenarios = ['A', 'B', 'C', 'D']
            for i in range(len(scenarios)):
                for j in range(i+1, len(scenarios)):
                    scenario1 = scenarios[i]
                    scenario2 = scenarios[j]
                    scores1 = model_data[model_data['scenario'] == scenario1]['normalized_score']
                    scores2 = model_data[model_data['scenario'] == scenario2]['normalized_score']
                    
                    f.write(f'● {scenario1} vs {scenario2}:\n')
                    
                    # 检查样本量
                    if len(scores1) < 3 or len(scores2) < 3:
                        f.write(f'  样本量不足（至少需要3个样本）: {len(scores1)} vs {len(scores2)}\n\n')
                        continue
                    
                    # 检查是否存在完全相同的数据
                    if scores1.equals(scores2):
                        f.write(f'  两组数据完全相同，无需进行检验\n\n')
                        continue
                    
                    # 确保两组数据长度相同
                    min_length = min(len(scores1), len(scores2))
                    if len(scores1) != len(scores2):
                        f.write(f'  警告: 样本量不同，将截取至相同长度 {min_length}\n')
                        scores1 = scores1.iloc[:min_length]
                        scores2 = scores2.iloc[:min_length]
                    
                    # 计算差值并检查零差值的数量
                    differences = scores1.values - scores2.values
                    zero_diffs = np.sum(differences == 0)
                    if zero_diffs > 0:
                        f.write(f'  警告: 有 {zero_diffs} 个零差值\n')
                    
                    # 如果所有差值都为零，则跳过检验
                    if zero_diffs == len(differences):
                        f.write(f'  所有差值都为零，无法进行检验\n\n')
                        continue
                    
                    try:
                        # 使用正态近似法并采用 pratt 方法处理零差值
                        stat, p = wilcoxon(scores1, scores2, zero_method='pratt', alternative='two-sided', mode='approx')
                        f.write(f'  统计量 = {stat:.4f}\n')
                        f.write(f'  p值    = {p:.4f}\n')
                        
                        # 显示检验结果的解释
                        if p < 0.05:
                            f.write(f'  结论: 在α=0.05的显著性水平下，两场景之间存在显著差异\n\n')
                        else:
                            f.write(f'  结论: 在α=0.05的显著性水平下，两场景之间不存在显著差异\n\n')
                    except ValueError as e:
                        f.write(f'  无法计算（样本数据可能不足或其他问题）: {str(e)}\n\n')
            
            # 计算A12效应量
            f.write('■ A12效应量分析结果:\n')
            f.write('-' * 50 + '\n\n')
            
            def compute_a12(group1, group2):
                """计算A12效应量"""
                m = len(group1)
                n = len(group2)
                r = 0
                for i in range(m):
                    for j in range(n):
                        if group1.iloc[i] > group2.iloc[j]:
                            r += 1
                        elif group1.iloc[i] == group2.iloc[j]:
                            r += 0.5
                return r / (m * n)

            for i in range(len(scenarios)):
                for j in range(i+1, len(scenarios)):
                    scenario1 = scenarios[i]
                    scenario2 = scenarios[j]
                    scores1 = model_data[model_data['scenario'] == scenario1]['normalized_score']
                    scores2 = model_data[model_data['scenario'] == scenario2]['normalized_score']
                    
                    f.write(f'● {scenario1} vs {scenario2}:\n')
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        a12 = compute_a12(scores1, scores2)
                        
                        # 解释效应大小
                        interpretation = "无明显效应"
                        if a12 > 0.71:
                            interpretation = "大效应"
                        elif a12 > 0.64:
                            interpretation = "中等效应"
                        elif a12 > 0.56:
                            interpretation = "小效应"
                        elif a12 < 0.29:
                            interpretation = "大效应（相反方向）"
                        elif a12 < 0.36:
                            interpretation = "中等效应（相反方向）"
                        elif a12 < 0.44:
                            interpretation = "小效应（相反方向）"
                        
                        f.write(f'  A12 = {a12:.3f} ({interpretation})\n\n')
                    else:
                        f.write(f'  样本不足，无法计算A12效应量\n\n')
            
            f.write('\n' + '=' * 50 + '\n\n')

    print(f"分析完成，结果已保存到 {results_file}")

# ================ 使用大模型分析motivation决策考虑因素 ================

def analyze_motivation_with_llm(model_name="deepseek-v3", motivation_file=None, analysis_dir=None, llm_analysis_file=None):
    """
    使用大模型API判断motivation文件中决策是否考虑了熟悉度和未来期望
    :param model_name: 要使用的分析模型名称
    :param motivation_file: 动机文件路径
    :param analysis_dir: 分析结果目录路径
    :param llm_analysis_file: 分析结果文件路径
    """
    # 读取motivation.json文件
    try:
        with open(motivation_file, 'r', encoding='utf-8') as f:
            motivations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取motivation文件出错: {str(e)}")
        return
    
    # 按模型分组
    model_motivations = {}
    for item in motivations:
        model = item.get('model')
        if model not in model_motivations:
            model_motivations[model] = []
        model_motivations[model].append(item)
    
    # 准备分析结果
    llm_analysis_results = {}
    
    # 系统提示词
    system_prompt = """
    您是判断危机决策考虑因素的专家。请判断文本是否考虑了以下因素:
    1. 熟悉度：对两支小队的了解程度/熟悉程度/信任程度
    2. 未来合作期望：是否期望在未来继续与红色小队协作
    """
    
    # 用户提示词
    user_prompt_template = """
    决策动机: {motivation}

    请回答做出决策时是否考虑了以下因素(回答"是"或"否")，
    回答格式：
    1. 考虑了熟悉度：是/否
    2. 考虑了未来合作期望：是/否
    """
    
    # 创建分析结果目录
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    with open(llm_analysis_file, 'w', encoding='utf-8') as f:
        f.write('================ 大模型分析动机考虑因素结果 ================\n\n')
        
        # 对每个模型进行分析
        for model, items in model_motivations.items():
            print(f"正在分析 {model} 的决策动机...")
            f.write(f'□ 模型: {model}\n')
            f.write('=' * 50 + '\n\n')
            
            model_results = {
                'total_analyzed': 0,
                'familiarity_considered': 0,  # 考虑了熟悉度的数量
                'ecfi_considered': 0,  # 考虑了未来合作期望的数量
                'scenario_stats': {'A': {}, 'B': {}, 'C': {}, 'D': {}}
            }
            
            # 初始化场景统计
            for scenario in ['A', 'B', 'C', 'D']:
                model_results['scenario_stats'][scenario] = {
                    'total': 0,
                    'familiarity_considered': 0,
                    'ecfi_considered': 0
                }
            
            # 设置模型
            api.set_model(model_name)  # 使用指定的模型进行分析
            
            # 构建LLM
            llm = build_agent_llm()
            
            # 显示总数量
            total_items = len(items)
            print(f"共有 {total_items} 条数据需要分析...")
            
            # 分析每个动机文本
            for idx, item in enumerate(items):
                motivation = item.get('motivation', '')
                scenario = item.get('scenario', '')
                test_id = item.get('test_id', '')
                
                # 跳过错误记录
                if motivation == 'ERROR_RETRIEVING_MOTIVATION':
                    continue
                
                # 显示进度
                print(f"进度: {idx+1}/{total_items} ({(idx+1)/total_items*100:.1f}%)", end='\r')
                
                model_results['total_analyzed'] += 1
                model_results['scenario_stats'][scenario]['total'] += 1
                
                # 使用LLM分析
                try:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt_template.format(
                            motivation=motivation
                        ))
                    ]
                    
                    response = llm.invoke(messages)
                    analysis = response.content
                    
                    # 检查是否考虑了熟悉度
                    if "考虑了熟悉度：是" in analysis or "熟悉度:是" in analysis:
                        model_results['familiarity_considered'] += 1
                        model_results['scenario_stats'][scenario]['familiarity_considered'] += 1
                    
                    # 检查是否考虑了未来合作期望
                    if "考虑了未来合作期望：是" in analysis or "未来合作期望:是" in analysis:
                        model_results['ecfi_considered'] += 1
                        model_results['scenario_stats'][scenario]['ecfi_considered'] += 1
                    
                except Exception as e:
                    print(f"\n分析错误: {str(e)}")
                    continue
            
            print("\n分析完成！")
            
            # 写入分析结果
            total = model_results['total_analyzed']
            if total == 0:
                f.write('  没有可分析的记录\n\n')
                continue
            
            # 写入总体统计
            f.write('■ 总体统计:\n')
            f.write('-' * 50 + '\n\n')
            
            fam_count = model_results['familiarity_considered']
            ecfi_count = model_results['ecfi_considered']
            
            f.write(f'  分析总数: {total}\n')
            f.write(f'  考虑了熟悉度的数量: {fam_count} ({fam_count/total*100:.1f}%)\n')
            f.write(f'  考虑了未来合作期望的数量: {ecfi_count} ({ecfi_count/total*100:.1f}%)\n\n')
            
            # 按场景分析考虑情况
            f.write('■ 各场景考虑情况:\n')
            f.write('-' * 50 + '\n\n')
            
            for scenario in ['A', 'B', 'C', 'D']:
                scenario_stats = model_results['scenario_stats'].get(scenario, {})
                scenario_total = scenario_stats.get('total', 0)
                
                if scenario_total > 0:
                    fam_scenario = scenario_stats.get('familiarity_considered', 0)
                    ecfi_scenario = scenario_stats.get('ecfi_considered', 0)
                    
                    f.write(f'  场景{scenario} (共{scenario_total}条):\n')
                    f.write(f'    考虑熟悉度: {fam_scenario}/{scenario_total} ({fam_scenario/scenario_total*100:.1f}%)\n')
                    f.write(f'    考虑未来合作期望: {ecfi_scenario}/{scenario_total} ({ecfi_scenario/scenario_total*100:.1f}%)\n')
                    f.write('\n')
                else:
                    f.write(f'  场景{scenario}: 无数据\n\n')
            
            f.write('\n' + '=' * 50 + '\n\n')
    
    print(f"分析完成，结果已保存到 {llm_analysis_file}")

def build_agent_llm():
    """
    构建LLM实例用于简单判断任务
    """
    if api.current_config['provider'] == 'azure':
        llm = AzureChatOpenAI(
            azure_endpoint=api.current_config['api_base'],
            api_key=api.current_config['api_key'],
            api_version=api.current_config['azure_config']['api_version'],
            deployment_name=api.current_config['azure_config']['deployment_name'],
            temperature=0.0,  # 使用0温度以获取一致判断
            top_p=1.0, 
            timeout=20,
            max_tokens=256  # 减少token，因为只需要简单回答
        )
    else:
        # 对于其他提供商
        openai.api_base = api.current_config['api_base']
        openai.api_key = api.current_config['api_key']
        
        llm = ChatOpenAI(
            api_key=api.current_config['api_key'],
            model_name=api.current_config['model_name'],
            temperature=0.0,  # 使用0温度以获取一致判断
            top_p=1.0,
            presence_penalty=0,
            frequency_penalty=0,
            max_tokens=256,  # 减少token，因为只需要简单回答
            openai_api_base=api.current_config['api_base'],
            openai_api_key=api.current_config['api_key']
        )
    return llm

# ================ 文件路径配置 ================
# 所有文件路径相关的配置都集中在这里

def get_file_paths():
    """返回程序中使用的所有文件路径"""
    # 基础路径
    base_folder = 'data/assignunit'
    
    # 数据文件路径
    raw_data_file = os.path.join(base_folder, 'raw_responses_assignunit_simpcot.csv')
    motivation_file = os.path.join(base_folder, 'raw_responses_assignunit_simpcot_motivation.json')
    
    # 分析结果路径
    analysis_dir = os.path.join('results/assignunit', 'analysis_0')
    results_file = os.path.join(analysis_dir, 'analysis_results_assignunit.txt')
    llm_analysis_file = os.path.join(analysis_dir, 'llm_motivation_analysis.txt')
    
    return {
        'raw_data_file': raw_data_file,
        'motivation_file': motivation_file,
        'analysis_dir': analysis_dir,
        'results_file': results_file,
        'llm_analysis_file': llm_analysis_file
    }

# 执行大模型动机分析
if __name__ == "__main__":
    # 获取文件路径配置
    paths = get_file_paths()
    
    # 执行统计分析
    perform_statistical_analysis(
        raw_data_file=paths['raw_data_file'],
        analysis_dir=paths['analysis_dir'],
        results_file=paths['results_file']
    )
    
    # 如果需要分析motivation关键词，取消下面这行注释
    #analyze_motivation_with_llm(
    #    model_name="deepseek-v3",
    #    motivation_file=paths['motivation_file'],
    #    analysis_dir=paths['analysis_dir'],
    #    llm_analysis_file=paths['llm_analysis_file']
    #)
