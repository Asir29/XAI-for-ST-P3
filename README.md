## 🚗 Forgetting to Care, Forgetting to See: Interpreting ST-P3 Planning via LLM Explanations and Collision Visualizations

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asir29/XAI-for-ST-P3/blob/main/XAI_Scripts/Colab_Evaluation_on_mini_set%20_Ollama.ipynb)


This repository provides an explainable evaluation pipeline for the **ST-P3** motion planning model on the **nuScenes mini-set**, integrated with the **Ollama** large language model for interpretability.

---

## 🧪 What’s Inside?

🔹 A Colab-ready notebook:  
📍 `XAI_Scripts/Colab_Evaluation_on_mini_set_Ollama.ipynb`

This notebook performs the full setup and evaluation:

✅ Sets up a Conda environment with all required packages using `condacolab`  
✅ Downloads and installs the **ST-P3** repository  
✅ Fetches the **pre-trained model image**  
✅ Downloads the **nuScenes mini-set** dataset  
✅ Configures the **Ollama LLM** framework  
✅ Runs the **evaluation** process end-to-end

---

## 🧠 Explainability Settings

### 🛑 "Forgetting to Care"
To simulate the model **ignoring specific semantic concepts** (e.g., obstacles, lane dividers), edit the concept mask logic in:
"ST-P3/stp3/cost.py"

(inserting 0 to deactivate the concept or 1 to activate it)

### 💬 Adjusting LLM Prompting
To change how the LLM focuses on the cost explanation logic, modify the prompt section in:
the function "explain_with_ollama"
in the file "ST-P3/evaluate.py"

### 👁️ "Forgetting to See" 
To simulate the model **missing visual input of certain elements** (e.g., pedestrians, obstacles), intervene on the occupancy map predictions by zeroing out or modifying specific class predictions in "ST-P3/evaluate.py"




(This enables fine-tuning of the explanations for more insightful or focused outputs.)
