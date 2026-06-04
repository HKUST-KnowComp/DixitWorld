# -*- coding: utf-8 -*-
"""
生成可交互的HTML评估表单
支持在线填写并导出JSON结果
"""

import json

def generate_interactive_html(storyteller_rounds, output_file, lang='cn'):
    """生成可交互的HTML评估表单"""
    
    is_cn = (lang == 'cn')
    title = "VLM-Dixit 真人评估表单" if is_cn else "VLM-Dixit Human Evaluation Form"
    
    html = f"""<!DOCTYPE html>
<html lang="{"zh-CN" if is_cn else "en"}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        .header-info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .case-card {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 25px;
            margin: 30px 0;
            transition: all 0.3s;
        }}
        .case-card.evaluated {{
            border-color: #28a745;
            background: #d4edda;
        }}
        .case-number {{
            background: #3498db;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .clue-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            font-size: 1.1em;
            font-style: italic;
        }}
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .image-item {{
            text-align: center;
            padding: 10px;
            border: 3px solid #dee2e6;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .image-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .image-item.selected {{
            border-color: #007bff;
            background: #cfe2ff;
        }}
        .image-item img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 8px;
        }}
        .evaluation-form {{
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .form-group {{
            margin: 15px 0;
        }}
        .form-group label {{
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
        }}
        .radio-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .radio-option {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .radio-option input[type="radio"] {{
            width: 20px;
            height: 20px;
            cursor: pointer;
        }}
        input[type="number"] {{
            width: 80px;
            padding: 8px;
            font-size: 1em;
            border: 2px solid #ced4da;
            border-radius: 5px;
        }}
        input[type="text"], textarea {{
            width: 100%;
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ced4da;
            border-radius: 5px;
            font-family: inherit;
        }}
        textarea {{
            min-height: 80px;
            resize: vertical;
        }}
        .answer-reveal {{
            background: #d1ecf1;
            border: 2px solid #bee5eb;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }}
        .answer-reveal.show {{
            display: block;
        }}
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }}
        .btn-primary {{
            background: #007bff;
            color: white;
        }}
        .btn-primary:hover {{
            background: #0056b3;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .btn-success {{
            background: #28a745;
            color: white;
        }}
        .btn-success:hover {{
            background: #218838;
        }}
        .btn-export {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 30px;
            font-size: 1.1em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
        }}
        .nav-buttons {{
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }}
        .stats {{
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
        }}
        @media (max-width: 768px) {{
            .images-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .container {{
                padding: 15px;
            }}
            h1 {{
                font-size: 1.5em;
            }}
            .btn-export {{
                bottom: 15px;
                right: 15px;
                padding: 12px 20px;
                font-size: 1em;
            }}
            .image-item img {{
                max-height: 150px;
                object-fit: cover;
            }}
        }}
        @media (max-width: 480px) {{
            body {{
                padding: 10px;
            }}
            .container {{
                padding: 10px;
            }}
            h1 {{
                font-size: 1.3em;
            }}
            .header-info {{
                font-size: 0.9em;
            }}
            .header-info ol {{
                padding-left: 20px;
            }}
            .case-card {{
                padding: 15px;
            }}
            .images-grid {{
                gap: 8px;
            }}
            .btn-export {{
                bottom: 10px;
                right: 10px;
                padding: 10px 15px;
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 {title}</h1>
        
        <div class="header-info">
            <p><strong>{"评估说明" if is_cn else "Instructions"}:</strong></p>
            <ol>
                <li>{"阅读线索并查看4张候选图片" if is_cn else "Read the clue and view 4 candidate images"}</li>
                <li>{"点击您认为正确的图片" if is_cn else "Click the image you think is correct"}</li>
                <li>{"填写评分（1-5分）" if is_cn else "Fill in ratings (1-5 points)"}</li>
                <li>{"点击'显示答案'查看正确答案" if is_cn else "Click 'Show Answer' to reveal the correct answer"}</li>
                <li>{"评估完所有案例后，点击右下角'导出结果'按钮" if is_cn else "After evaluating all cases, click 'Export Results' button"}</li>
            </ol>
        </div>
        
        <div class="stats">
            <strong>{"进度" if is_cn else "Progress"}: <span id="evaluated-count">0</span> / <span id="total-count">0</span></strong>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill">0%</div>
        </div>
        
        <div id="cases-container"></div>
        
        <button class="btn btn-success btn-export" onclick="exportResults()">
            📥 {"导出结果" if is_cn else "Export Results"}
        </button>
    </div>
    
    <script>
        const cases = """ + json.dumps(storyteller_rounds, ensure_ascii=False) + """;
        const lang = '""" + lang + """';
        let evaluationData = {};
        
        function initializeCases() {
            const container = document.getElementById('cases-container');
            document.getElementById('total-count').textContent = cases.length;
            
            cases.forEach((caseData, index) => {
                const caseDiv = createCaseCard(caseData, index);
                container.appendChild(caseDiv);
            });
            
            updateProgress();
        }
        
        function createCaseCard(caseData, index) {
            const div = document.createElement('div');
            div.className = 'case-card';
            div.id = `case-${index}`;
            
            const isCorrectText = lang === 'cn' ? '是否正确' : 'Correct?';
            const clarityText = lang === 'cn' ? 'Clarity（清晰度）' : 'Clarity';
            const creativityText = lang === 'cn' ? 'Creativity（创意性）' : 'Creativity';
            const ambiguityText = lang === 'cn' ? 'Ambiguity（模糊度）' : 'Ambiguity';
            const notesText = lang === 'cn' ? '备注' : 'Notes';
            const showAnswerText = lang === 'cn' ? '显示答案' : 'Show Answer';
            const correctAnswerText = lang === 'cn' ? '正确答案' : 'Correct Answer';
            const positionText = lang === 'cn' ? '位置' : 'Position';
            
            div.innerHTML = `
                <div class="case-number">${lang === 'cn' ? '案例' : 'Case'} ${index + 1}</div>
                <p style="color: #6c757d; font-size: 0.9em;">
                    ${lang === 'cn' ? '模型' : 'Model'}: ${caseData.storyteller_model} | 
                    Match ${caseData.match_number}, ${caseData.phase}, Round ${caseData.round_number}
                </p>
                
                <div class="clue-box">
                    <strong>${lang === 'cn' ? '线索' : 'Clue'}:</strong> "${caseData.clue}"
                </div>
                
                <div class="images-grid">
                    ${caseData.candidate_images.map((img, i) => `
                        <div class="image-item" onclick="selectImage(${index}, ${i})">
                            <img src="images/${img}" alt="${img}">
                            <div><strong>${i + 1}. ${img}</strong></div>
                        </div>
                    `).join('')}
                </div>
                
                <div class="evaluation-form">
                    <div class="form-group">
                        <label>${lang === 'cn' ? '您的选择：' : 'Your Choice:'}</label>
                        <input type="hidden" id="choice-${index}" value="">
                        <div id="choice-display-${index}" style="font-size: 1.2em; color: #007bff; font-weight: bold;">
                            ${lang === 'cn' ? '请点击上方图片选择' : 'Click an image above to select'}
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>${clarityText} (1-5):</label>
                        <input type="number" min="1" max="5" id="clarity-${index}" 
                               onchange="saveEvaluation(${index})">
                    </div>
                    
                    <div class="form-group">
                        <label>${creativityText} (1-5):</label>
                        <input type="number" min="1" max="5" id="creativity-${index}"
                               onchange="saveEvaluation(${index})">
                    </div>
                    
                    <div class="form-group">
                        <label>${ambiguityText} (1-5):</label>
                        <input type="number" min="1" max="5" id="ambiguity-${index}"
                               onchange="saveEvaluation(${index})">
                    </div>
                    
                    <div class="form-group">
                        <label>${notesText}:</label>
                        <textarea id="notes-${index}" onchange="saveEvaluation(${index})"></textarea>
                    </div>
                    
                    <button class="btn btn-primary" onclick="revealAnswer(${index})">
                        ${showAnswerText}
                    </button>
                    
                    <div class="answer-reveal" id="answer-${index}">
                        <strong>✅ ${correctAnswerText}:</strong> ${caseData.target_image} 
                        (${positionText} ${caseData.target_position + 1})
                        <div id="result-${index}" style="margin-top: 10px; font-size: 1.1em;"></div>
                    </div>
                </div>
            `;
            
            return div;
        }
        
        function selectImage(caseIndex, imageIndex) {
            // Remove previous selection
            document.querySelectorAll(`#case-${caseIndex} .image-item`).forEach(item => {
                item.classList.remove('selected');
            });
            
            // Add new selection
            document.querySelectorAll(`#case-${caseIndex} .image-item`)[imageIndex].classList.add('selected');
            
            // Save choice
            document.getElementById(`choice-${caseIndex}`).value = imageIndex + 1;
            document.getElementById(`choice-display-${caseIndex}`).innerHTML = 
                `<strong>${lang === 'cn' ? '已选择' : 'Selected'}: ${imageIndex + 1}</strong>`;
            
            saveEvaluation(caseIndex);
        }
        
        function revealAnswer(caseIndex) {
            const answerDiv = document.getElementById(`answer-${caseIndex}`);
            answerDiv.classList.add('show');
            
            const choice = parseInt(document.getElementById(`choice-${caseIndex}`).value);
            const correctPos = cases[caseIndex].target_position + 1;
            const isCorrect = choice === correctPos;
            
            const resultDiv = document.getElementById(`result-${caseIndex}`);
            if (choice) {
                if (isCorrect) {
                    resultDiv.innerHTML = `<span style="color: green;">✓ ${lang === 'cn' ? '选择正确！' : 'Correct!'}</span>`;
                } else {
                    resultDiv.innerHTML = `<span style="color: red;">✗ ${lang === 'cn' ? '选择错误' : 'Incorrect'}. ${lang === 'cn' ? '您选了' : 'You chose'} ${choice}, ${lang === 'cn' ? '正确答案是' : 'correct answer is'} ${correctPos}</span>`;
                }
            } else {
                resultDiv.innerHTML = `<span style="color: orange;">${lang === 'cn' ? '您未选择' : 'No choice made'}</span>`;
            }
            
            saveEvaluation(caseIndex);
        }
        
        function saveEvaluation(caseIndex) {
            const choice = document.getElementById(`choice-${caseIndex}`).value;
            const clarity = document.getElementById(`clarity-${caseIndex}`).value;
            const creativity = document.getElementById(`creativity-${caseIndex}`).value;
            const ambiguity = document.getElementById(`ambiguity-${caseIndex}`).value;
            const notes = document.getElementById(`notes-${caseIndex}`).value;
            
            const correctPos = cases[caseIndex].target_position + 1;
            const isCorrect = choice ? (parseInt(choice) === correctPos) : null;
            
            evaluationData[caseIndex] = {
                case_number: caseIndex + 1,
                model: cases[caseIndex].storyteller_model,
                match: cases[caseIndex].match_number,
                phase: cases[caseIndex].phase,
                round: cases[caseIndex].round_number,
                clue: cases[caseIndex].clue,
                target_image: cases[caseIndex].target_image,
                target_position: correctPos,
                your_choice: choice ? parseInt(choice) : null,
                is_correct: isCorrect,
                clarity: clarity ? parseInt(clarity) : null,
                creativity: creativity ? parseInt(creativity) : null,
                ambiguity: ambiguity ? parseInt(ambiguity) : null,
                notes: notes
            };
            
            // Check if fully evaluated
            if (choice && clarity && creativity && ambiguity) {
                document.getElementById(`case-${caseIndex}`).classList.add('evaluated');
            }
            
            updateProgress();
        }
        
        function updateProgress() {
            const total = cases.length;
            const evaluated = Object.values(evaluationData).filter(e => 
                e.your_choice && e.clarity && e.creativity && e.ambiguity
            ).length;
            
            document.getElementById('evaluated-count').textContent = evaluated;
            
            const percentage = (evaluated / total * 100).toFixed(0);
            const progressFill = document.getElementById('progress-fill');
            progressFill.style.width = percentage + '%';
            progressFill.textContent = percentage + '%';
        }
        
        function exportResults() {
            const evaluatedCount = Object.values(evaluationData).filter(e => 
                e.your_choice && e.clarity && e.creativity && e.ambiguity
            ).length;
            
            if (evaluatedCount === 0) {
                alert(lang === 'cn' ? '请至少完成一个评估再导出！' : 'Please complete at least one evaluation before exporting!');
                return;
            }
            
            const results = {
                evaluator: prompt(lang === 'cn' ? '请输入您的姓名或ID：' : 'Please enter your name or ID:') || 'Anonymous',
                timestamp: new Date().toISOString(),
                total_cases: cases.length,
                evaluated_cases: evaluatedCount,
                evaluations: Object.values(evaluationData).filter(e => e.your_choice)
            };
            
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `evaluation_results_${{results.evaluator}}_${{new Date().toISOString().split('T')[0]}}.json`;
            link.click();
            
            alert(lang === 'cn' ? 
                `成功导出 ${{evaluatedCount}} 个评估结果！\\n感谢您的参与！` : 
                `Successfully exported ${{evaluatedCount}} evaluations!\\nThank you for your participation!`);
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', initializeCases);
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 交互式HTML表单已生成: {output_file}")

def prepare_cases_data(storyteller_rounds, target_total=120):
    """准备案例数据"""
    import random
    random.seed(42)
    
    models_order = sorted(storyteller_rounds.keys())
    num_models = len(models_order)
    
    min_rounds = min(len(rounds) for rounds in storyteller_rounds.values())
    
    samples_per_model = {}
    if min_rounds < 20:
        models_with_few = [m for m in models_order if len(storyteller_rounds[m]) < 20]
        models_with_enough = [m for m in models_order if len(storyteller_rounds[m]) >= 20]
        
        used_by_few = sum(len(storyteller_rounds[m]) for m in models_with_few)
        remaining = target_total - used_by_few
        
        if models_with_enough:
            per_enough = remaining // len(models_with_enough)
            for m in models_with_few:
                samples_per_model[m] = len(storyteller_rounds[m])
            for m in models_with_enough:
                samples_per_model[m] = per_enough
        else:
            for m in models_order:
                samples_per_model[m] = len(storyteller_rounds[m])
    else:
        per_model = target_total // num_models
        for m in models_order:
            samples_per_model[m] = per_model
    
    all_cases = []
    for model in models_order:
        rounds = storyteller_rounds[model]
        num_to_sample = samples_per_model[model]
        
        if len(rounds) > num_to_sample:
            selected_rounds = random.sample(rounds, num_to_sample)
        else:
            selected_rounds = rounds
        
        selected_rounds.sort(key=lambda x: (x['match_number'], x['round_number']))
        all_cases.extend(selected_rounds)
    
    return all_cases

def main():
    # 导入数据收集函数
    import sys
    sys.path.append('.')
    from generate_human_evaluation import collect_storyteller_rounds
    
    print("🔍 收集说书人回合数据...")
    storyteller_rounds = collect_storyteller_rounds('parallel_batch_test_progress.json')
    
    print("📦 准备案例数据...")
    cases_data = prepare_cases_data(storyteller_rounds, target_total=120)
    
    print(f"\n📝 生成交互式HTML表单...")
    print(f"   总案例数: {len(cases_data)}")
    
    generate_interactive_html(cases_data, 'EVALUATION_FORM_CN.html', lang='cn')
    generate_interactive_html(cases_data, 'EVALUATION_FORM_EN.html', lang='en')
    
    print("\n🎉 完成！")
    print("📄 可交互的HTML表单:")
    print("   - EVALUATION_FORM_CN.html (中文版)")
    print("   - EVALUATION_FORM_EN.html (英文版)")
    print("\n💡 使用说明:")
    print("   1. 在浏览器中打开HTML文件")
    print("   2. 点击图片进行选择")
    print("   3. 填写评分")
    print("   4. 点击'显示答案'查看结果")
    print("   5. 完成后点击'导出结果'保存JSON文件")

if __name__ == '__main__':
    main()
