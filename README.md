# LLMNER

基于大型语言模型（Large Language Model, LLM）的命名实体识别（Named Entity Recognition, NER）工具与实验仓库。

> **状态**：示例 README（请根据项目实际文件/脚本名调整命令与说明）

---

## 项目简介

LLMNER 是一个用来探索和评估大型语言模型在命名实体识别任务上表现的实验性仓库。它包含基于 prompt 的方法（零样本/少样本推理）、基于 LLM 的分类器封装、以及辅助的训练/评估脚本。


---

## 仓库结构（示例）

```text
LLMNER/
├── data/                # 示例数据、数据集转换脚本
├── llm-cls/             # 基于 LLM 的分类/推理实现
├── train/               # 训练或微调相关脚本（若有）
├── chat/                # prompt 模板、交互脚本
├── result/              # 输出预测与评估结果
├── requirements.txt     # Python 依赖
├── README.md            # 本文件
```

---

## 安装

建议使用虚拟环境来隔离依赖：

```bash
git clone https://github.com/dreamer-tx/LLMNER.git
cd LLMNER
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 评估指标

推荐计算并保存以下指标：

* 全局：Precision、Recall、F1（micro / macro）
* 按实体类型：每类的 Precision / Recall / F1

---

## 可扩展方向

* 增加更多 prompt 模板并做自动化对比。
* 支持多语言或专有领域（医疗、法律）实体集。
---
