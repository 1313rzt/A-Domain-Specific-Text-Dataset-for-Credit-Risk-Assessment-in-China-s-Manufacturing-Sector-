# A Domain-Specific Text Dataset for Credit Risk Assessment in China’s Manufacturing Sector

This repository contains the **Manufacturing Credit Risk Labeled Sentences (MCRLS)** dataset — a large-scale, sentence-level Chinese financial text dataset specifically designed for **credit risk analysis** in the manufacturing industry. It includes **36,838 expert-labeled sentences** extracted from **19,479 annual reports** of **2,593 A-share listed manufacturing companies** between **2013 and 2022**.

The dataset captures both risk-related and non-risk textual signals and is intended for training and evaluating large language models (LLMs), developing text-based credit scoring systems, and enhancing financial risk analytics.

---

## 📁 Repository Contents

MCRLS-dataset/
│
├── data/
│ ├── MCRLS_dataset.xlsx # The labeled sentence dataset
│ └── keywords.txt # List of 396 TF-IDF + policy-derived keywords
│
├── scripts/
│ ├── pdf_to_txt.py # PDF parser for annual reports
│ ├── extract_sections.py # Extract key sections using regex
│ ├── tfidf_keywords.py # TF-IDF keyword extractor
│ └── filter_sentences.py # Sentence-level filtering and sampling
│
├── annotation_guidelines.pdf # Labeling instructions and category definitions
├── README.md # Project overview and instructions
└── LICENSE # CC-BY 4.0 license

---

## 🧾 Dataset Overview

| Field     | Type    | Description |
|-----------|---------|-------------|
| `sentence` | string  | A sentence extracted from a Chinese annual report |
| `label`    | integer | Credit risk label: -1 = Risky, 0 = Neutral, 1 = Positive |

- **Total Sentences**: 36,838  
- **Language**: Simplified Chinese  
- **Years Covered**: 2013–2022  
- **Industry**: Manufacturing sector (A-share listed firms in China)

All sentences were extracted from key sections of annual reports:  
> “Important Risk Warning”, “Management Discussion and Analysis (MD&A)”, “Board Reports”  
and filtered using a 396-word financial risk keyword dictionary.

---

## 🟩 Annotation Guidelines

**Label Categories:**

| Label | Description |
|-------|-------------|
| `-1`  | Risky — reflects financial pressure, debt, instability, supply chain risk, etc. |
| `0`   | Neutral — general descriptions, factual statements, not directly related to risk |
| `1`   | Positive — reflects financial strength, strategic mitigation, financing, or partnerships |

**Examples**:

- **Risky (-1)**:  
  “公司存在应收账款回收周期较长的风险。”  
  “市场需求下滑导致库存积压。”

- **Neutral (0)**:  
  “2022年营业收入同比增长8.2%。”  
  “本公司位于江苏省南京市。”

- **Positive (1)**:  
  “公司通过债券融资获得新资金用于技术改造项目。”  
  “与上下游建立稳定合作关系，有效降低供应链风险。”

**Annotation Process**:

- Annotators: Financial experts and postgraduate students  
- Method: Manual labeling in Excel  
- Quality Assurance: Multi-round arbitration + 10% cross-validation  
- Agreement Rate: >90%



 Usage Instructions

```python
import pandas as pd

# Load the dataset
df = pd.read_excel("data/MCRLS_dataset.xlsx", engine="openpyxl")

# Preview
print(df.sample(5))
You can fine-tune BERT, ERNIE, or other LLMs using this dataset for sentence-level risk classification.

💡 Applications
Fine-tuning financial NLP models (BERT, ERNIE, RoBERTa)

Sentence-level credit risk classification

Forward-looking risk signal extraction

Supply chain risk modeling


Regulatory risk monitoring (RegTech)

🔓 License
This dataset is released under the Creative Commons Attribution 4.0 (CC BY 4.0) license.
You are free to use, share, adapt, and build upon this dataset with appropriate credit.

📄 Citation
If you use this dataset, please cite the following work:

Zhangwei, Renzhitao, et al. (2025).
A Domain-Specific Text Dataset for Credit Risk Assessment in China’s Manufacturing Sector.
Scientific Data (under review).
Dataset available at: [https://github.com/yourusername/MCRLS-dataset](https://github.com/1313rzt/A-Domain-Specific-Text-Dataset-for-Credit-Risk-Assessment-in-China-s-Manufacturing-Sector-)
