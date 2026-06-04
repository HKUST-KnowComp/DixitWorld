# -*- coding: utf-8 -*-
"""
将Markdown评估文档转换为HTML，方便在浏览器中查看
"""

def md_to_html(md_file, html_file, title):
    """将Markdown转换为HTML"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建HTML模板
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 40px;
        }}
        h3 {{
            color: #2980b9;
            background-color: #ecf0f1;
            padding: 10px;
            border-left: 4px solid #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table th, table td {{
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        img {{
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }}
        img:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .case-container {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .clue {{
            background-color: #fff9e6;
            padding: 15px;
            border-left: 4px solid #f39c12;
            margin: 10px 0;
            font-style: italic;
            font-size: 1.1em;
        }}
        .eval-area {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .eval-area input[type="number"] {{
            width: 60px;
            padding: 5px;
            font-size: 1em;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        .eval-area input[type="text"] {{
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        .game-result {{
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
        }}
        .success {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .failure {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .checkmark {{
            color: #27ae60;
            font-size: 1.5em;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        hr {{
            border: none;
            border-top: 2px dashed #bdc3c7;
            margin: 30px 0;
        }}
        .overview-table {{
            margin: 20px 0;
        }}
        .meta-info {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
"""
    
    # 简单的Markdown到HTML转换
    lines = content.split('\n')
    in_table = False
    in_eval = False
    case_open = False
    
    for i, line in enumerate(lines):
        # 跳过已经是HTML的table标签
        if line.strip().startswith('<table>') or line.strip().startswith('</table>') or line.strip().startswith('<tr>') or line.strip().startswith('</tr>') or line.strip().startswith('<td'):
            html += line + '\n'
            continue
        
        # 标题
        if line.startswith('# '):
            html += f'<h1>{line[2:]}</h1>\n'
        elif line.startswith('## '):
            if case_open:
                html += '</div>\n'
                case_open = False
            html += f'<h2>{line[3:]}</h2>\n'
        elif line.startswith('### '):
            if case_open:
                html += '</div>\n'
            html += '<div class="case-container">\n'
            html += f'<h3>{line[4:]}</h3>\n'
            case_open = True
        # 加粗
        elif '**' in line:
            # 处理线索
            if '说书人线索' in line or "Storyteller's Clue" in line:
                clue = line.split('**: "')[1].rstrip('"')
                html += f'<div class="clue">{clue}</div>\n'
            # 处理游戏结果
            elif '游戏结果' in line or 'Game Result' in line:
                if '✅' in line:
                    result = line.split('**: ')[1]
                    html += f'<div class="game-result success">{result}</div>\n'
                else:
                    result = line.split('**: ')[1]
                    html += f'<div class="game-result failure">{result}</div>\n'
            # 处理评估区域
            elif '评估区域' in line or 'Evaluation Area' in line:
                html += '<div class="eval-area">\n<strong>评估区域：</strong>\n'
                in_eval = True
            else:
                # 一般加粗处理
                line = line.replace('**', '<strong>').replace('</strong>', '</strong>', 1)
                if '<strong>' in line and '</strong>' not in line:
                    line += '</strong>'
                html += f'<p>{line}</p>\n'
        # 评估项
        elif in_eval and ('- [ ]' in line or '- Clarity' in line or '- Creativity' in line or '- Ambiguity' in line or '- Notes' in line or '- 备注' in line):
            if 'Clarity' in line or '清晰度' in line:
                html += '<label><strong>Clarity（清晰度）</strong>: <input type="number" min="1" max="5" placeholder="1-5"> / 5</label><br>\n'
            elif 'Creativity' in line or '创意性' in line:
                html += '<label><strong>Creativity（创意性）</strong>: <input type="number" min="1" max="5" placeholder="1-5"> / 5</label><br>\n'
            elif 'Ambiguity' in line or '模糊度' in line:
                html += '<label><strong>Ambiguity（模糊度）</strong>: <input type="number" min="1" max="5" placeholder="1-5"> / 5</label><br>\n'
            elif 'Notes' in line or '备注' in line:
                html += '<label><strong>备注 / Notes</strong>: <input type="text" placeholder="请输入备注..."></label>\n'
                html += '</div>\n'
                in_eval = False
        # 分隔线
        elif line.strip() == '---':
            html += '<hr>\n'
        # 列表项
        elif line.startswith('- ') or line.startswith('   - '):
            html += f'<p style="margin-left: 20px;">{line[2:]}</p>\n'
        # 表格
        elif line.startswith('|') and not in_table:
            # 开始表格
            if '|---' not in line:
                html += '<table class="overview-table">\n<thead><tr>\n'
                cells = [c.strip() for c in line.split('|')[1:-1]]
                for cell in cells:
                    html += f'<th>{cell}</th>\n'
                html += '</tr></thead>\n<tbody>\n'
                in_table = True
        elif line.startswith('|') and in_table:
            if '|---' in line:
                continue
            html += '<tr>\n'
            cells = [c.strip() for c in line.split('|')[1:-1]]
            for cell in cells:
                html += f'<td>{cell}</td>\n'
            html += '</tr>\n'
        elif in_table and not line.startswith('|'):
            html += '</tbody>\n</table>\n'
            in_table = False
        # 普通段落
        elif line.strip():
            html += f'<p class="meta-info">{line}</p>\n'
        else:
            html += '<br>\n'
    
    if case_open:
        html += '</div>\n'
    
    html += """
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML文档已生成: {html_file}")

def main():
    print("🔄 转换中文评估文档为HTML...")
    md_to_html('HUMAN_EVALUATION_CN.md', 'HUMAN_EVALUATION_CN.html', 'VLM-Dixit 真人评估数据集')
    
    print("🔄 转换英文评估文档为HTML...")
    md_to_html('HUMAN_EVALUATION_EN.md', 'HUMAN_EVALUATION_EN.html', 'VLM-Dixit Human Evaluation Dataset')
    
    print("\n🎉 转换完成！")
    print("📄 可以在浏览器中打开HTML文件进行评估")
    print("   - HUMAN_EVALUATION_CN.html")
    print("   - HUMAN_EVALUATION_EN.html")

if __name__ == '__main__':
    main()
