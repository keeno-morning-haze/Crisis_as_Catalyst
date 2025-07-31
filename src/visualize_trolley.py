import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_responses(raw_file, analysis_file):
    """
    分析问卷回答并生成统计结果
    """
    # 读取CSV文件，指定列名
    columns = ['model'] + [f'Q{i+1}' for i in range(13)]
    df = pd.read_csv(raw_file, names=columns)
    
    # 文本分析结果
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("道德机器问题统计分析结果:\n\n")
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            if len(model_data) > 0:  # 只输出有数据的模型结果
                f.write(f"\n模型 {model} 的统计结果:\n")
                
                for col in df.columns:
                    if col != 'model':
                        stats = model_data[col].value_counts()
                        total = len(model_data)
                        f.write(f"\n问题 {col}:\n")
                        for choice in ['A', 'B']:
                            count = stats.get(choice, 0)
                            percent = (count / total) * 100
                            f.write(f"选择 {choice}: {count}次 ({percent:.1f}%)\n")

def plot_model_comparison(raw_file, output_file=None):
    """
    绘制不同模型的选择分布对比图
    """
    if output_file is None:
        output_file = os.path.join(script_dir, 'model_comparison_trolley.png')
    
    # 读取数据
    df = pd.read_csv(raw_file)
    
    # 计算每个模型在每个问题上选择A的比例
    model_stats = []
    for q in range(1, 14):
        col = f'Q{q}'
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            if len(model_data) > 0:  # 只处理有数据的模型
                a_percent = (model_data[col] == 'A').mean() * 100
                model_stats.append({
                    'Question': col,
                    'Model': model,
                    'A_Percentage': max(a_percent, 0.5)  # 设置最小高度为0.5%
                })
    
    # 创建数据框
    stats_df = pd.DataFrame(model_stats)
    
    # 设置图表样式
    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 8))
    
    # 创建柱状图
    sns.barplot(data=stats_df, x='Question', y='A_Percentage', hue='Model')
    
    # 设置y轴范围，确保0.5%的最小高度可见
    plt.ylim(-1, 101)
    
    # 设置图表标题和标签
    plt.title('Different Models\' Choice Distribution (Percentage of A)')
    plt.xlabel('Question Number')
    plt.ylabel('Percentage choosing A (%)')
    
    # 调整x轴标签角度
    plt.xticks(rotation=45)
    
    # 添加图例
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

def plot_question_distribution(raw_file, output_dir=None):
    """
    为每个问题创建单独的分布图
    """
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'question_plots')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(raw_file)
    
    # 为每个问题创建图表
    for q in range(1, 14):
        col = f'Q{q}'
        
        plt.figure(figsize=(10, 6))
        
        # 计算每个模型的选择分布
        model_choices = []
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            choices = model_data[col].value_counts().reindex(['A', 'B']).fillna(0)
            choices_pct = (choices / len(model_data)) * 100
            model_choices.append({
                'Model': model,
                'A': choices_pct['A'],
                'B': choices_pct['B']
            })
        
        # 创建数据框
        choices_df = pd.DataFrame(model_choices)
        
        # 创建堆叠条形图
        choices_df.plot(x='Model', y=['A', 'B'], kind='bar', stacked=True)
        
        plt.title(f'Question {q} Choice Distribution by Model')
        plt.xlabel('Model')
        plt.ylabel('Percentage (%)')
        plt.legend(title='Choice')
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'question_{q}_distribution.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    raw_file = os.path.join("data/trolley", "raw_responses_trolley.csv")
    analysis_file = os.path.join("results/trolley", "analysis_results_trolley.txt")
    
    # 生成分析结果
    analyze_responses(raw_file, analysis_file)
    
    # 生成模型对比图
    plot_model_comparison(raw_file)
    
    # 生成每个问题的分布图
    plot_question_distribution(raw_file) 