"""
生成VLM模型测试结果的Excel报告
"""
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime

def create_excel_report():
    """创建Excel报告"""
    
    # 数据准备
    data = {
        '排名': [1, 2, 3, 4, 5, 6],
        '模型': ['gpt-4o', 'gemma3-27b', 'qwen2.5-vl-32b', 'gemini-2.5-flash', 'gemma3-12b', 'qwen2.5-vl-7b'],
        '总准确率': [75.00, 67.74, 67.06, 66.27, 64.29, 16.67],  # 更新qwen7b的数据
        'Easy准确率': [78.57, 58.33, 70.24, 65.48, 63.10, 14.29],
        'Medium准确率': [73.81, 59.52, 75.00, 69.05, 67.86, 16.67],
        'Hard准确率': [72.62, 57.14, 55.95, 64.29, 61.90, 19.05],
        'Tokens': [19013, 18409, 22869, 13145, 15902, 529],
        '预估成本($)': [0.19, 0.04, 0.05, 0.07, 0.03, 0.00]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 创建Excel文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"VLM_模型测试报告_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 写入主数据表
        df.to_excel(writer, sheet_name='模型排名', index=False)
        
        # 获取工作表对象
        worksheet = writer.sheets['模型排名']
        
        # 设置样式
        setup_worksheet_style(worksheet, len(df))
        
        # 创建按难度排名的表
        create_difficulty_rankings(writer, df)
        
        # 创建统计摘要表
        create_summary_sheet(writer, df)
    
    print(f"✅ Excel报告已生成: {filename}")
    return filename

def setup_worksheet_style(worksheet, num_rows):
    """设置工作表样式"""
    
    # 定义样式
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    center_alignment = Alignment(horizontal='center', vertical='center')
    
    # 设置标题行样式
    for col in range(1, 9):  # A到H列
        cell = worksheet.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
        cell.border = border
    
    # 设置数据行样式
    for row in range(2, num_rows + 2):
        for col in range(1, 9):
            cell = worksheet.cell(row=row, column=col)
            cell.border = border
            cell.alignment = center_alignment
            
            # 为排名列设置特殊样式
            if col == 1:  # 排名列
                if row == 2:  # 第一名
                    cell.fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
                elif row == 3:  # 第二名
                    cell.fill = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")
                elif row == 4:  # 第三名
                    cell.fill = PatternFill(start_color="CD7F32", end_color="CD7F32", fill_type="solid")
            
            # 为准确率列设置颜色渐变
            if col in [3, 4, 5, 6]:  # 准确率列
                value = cell.value
                if isinstance(value, (int, float)):
                    if value >= 70:
                        cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                    elif value >= 60:
                        cell.fill = PatternFill(start_color="FFFFE0", end_color="FFFFE0", fill_type="solid")
                    elif value >= 50:
                        cell.fill = PatternFill(start_color="FFE4B5", end_color="FFE4B5", fill_type="solid")
                    else:
                        cell.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
    
    # 调整列宽
    column_widths = [8, 20, 12, 12, 12, 12, 12, 12]
    for i, width in enumerate(column_widths, 1):
        worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = width

def create_difficulty_rankings(writer, df):
    """创建按难度排名的表"""
    
    difficulties = ['Easy', 'Medium', 'Hard']
    
    for difficulty in difficulties:
        # 按难度排序
        if difficulty == 'Easy':
            sorted_df = df.sort_values('Easy准确率', ascending=False)
            accuracy_col = 'Easy准确率'
        elif difficulty == 'Medium':
            sorted_df = df.sort_values('Medium准确率', ascending=False)
            accuracy_col = 'Medium准确率'
        else:  # Hard
            sorted_df = df.sort_values('Hard准确率', ascending=False)
            accuracy_col = 'Hard准确率'
        
        # 创建排名
        difficulty_df = pd.DataFrame({
            '排名': range(1, len(sorted_df) + 1),
            '模型': sorted_df['模型'].values,
            f'{difficulty}准确率': sorted_df[accuracy_col].values,
            '总准确率': sorted_df['总准确率'].values
        })
        
        # 写入工作表
        difficulty_df.to_excel(writer, sheet_name=f'{difficulty}难度排名', index=False)
        
        # 设置样式
        worksheet = writer.sheets[f'{difficulty}难度排名']
        setup_worksheet_style(worksheet, len(difficulty_df))

def create_summary_sheet(writer, df):
    """创建统计摘要表"""
    
    summary_data = {
        '指标': [
            '测试模型数',
            '每个模型样本数',
            '总测试次数',
            '最佳模型',
            '最佳总准确率',
            'Easy最佳模型',
            'Easy最佳准确率',
            'Medium最佳模型',
            'Medium最佳准确率',
            'Hard最佳模型',
            'Hard最佳准确率',
            '平均准确率',
            '总Tokens',
            '总预估成本($)'
        ],
        '数值': [
            6,
            252,
            1512,
            df.loc[df['总准确率'].idxmax(), '模型'],
            f"{df['总准确率'].max():.2f}%",
            df.loc[df['Easy准确率'].idxmax(), '模型'],
            f"{df['Easy准确率'].max():.2f}%",
            df.loc[df['Medium准确率'].idxmax(), '模型'],
            f"{df['Medium准确率'].max():.2f}%",
            df.loc[df['Hard准确率'].idxmax(), '模型'],
            f"{df['Hard准确率'].max():.2f}%",
            f"{df['总准确率'].mean():.2f}%",
            df['Tokens'].sum(),
            f"${df['预估成本($)'].sum():.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
    
    # 设置样式
    worksheet = writer.sheets['统计摘要']
    setup_worksheet_style(worksheet, len(summary_df))

def main():
    """主函数"""
    print("="*60)
    print("📊 生成VLM模型测试Excel报告")
    print("="*60)
    
    try:
        filename = create_excel_report()
        print(f"\n✅ 报告生成成功!")
        print(f"📁 文件位置: {filename}")
        print(f"📋 包含工作表:")
        print(f"   - 模型排名")
        print(f"   - Easy难度排名")
        print(f"   - Medium难度排名")
        print(f"   - Hard难度排名")
        print(f"   - 统计摘要")
        
    except Exception as e:
        print(f"❌ 生成报告时出错: {e}")

if __name__ == "__main__":
    main()
