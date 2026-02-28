# NLI Text Classifier

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/eyad222/nli-app1)

---

## 📌 Project Description

The **NLI Text Classifier** is an application that performs **Natural Language Inference (NLI)** to determine whether a given text **entails**, **contradicts**, or is **neutral** with respect to a specified label. For example, given a text about Linux distributions and the label `linux distro comparison`, the app predicts if the text supports (entailment), contradicts (contradiction), or is neutral.

This project was built using:

- **PyTorch** and **Hugging Face Transformers** for NLP modeling  
- **Streamlit** for interactive GUI  
- **Docker** for containerization and deployment  

The project was developed **under the guidance of Prof. Ashraf Mohamed and Eng. Mostafa Touny**, focusing on bridging theoretical NLP assumptions to practical, real-world usage.

---

## ⚠️ Problems in the Original Project

1. **Static Inputs**: The original code required hardcoded text and label inputs.  
2. **Inefficient Model Loading**: The model loaded every time a prediction was made, causing delays.  
3. **Limited Output**: Only probabilities were shown, with no human-readable label.  
4. **No GUI**: Users had to modify and run Python scripts manually.  
5. **Not Real-World Ready**: The code assumed ideal theoretical conditions (i.i.d data, full batch access, etc.) and could not handle streaming inputs or dynamic use cases.  

---

## 🏭 Real-World Assumptions

| Theoretical Assumption      | Reality Implementation                          |
|-----------------------------|------------------------------------------------|
| i.i.d data                  | Handles user streaming input                   |
| Fixed graph                 | Dynamic text inputs and labels                 |
| Full batch access           | Real-time single-example inference             |
| Exact inference             | Approximate inference with optimized caching  |

---

## 🔧 How I Tackled the Problems & Enhancements

### 1. Dynamic Input with Streamlit GUI

Original code (static inputs):

```python
sequence = "Hardcoded example text"
label = "linux distro comparison"
prob_label_is_true = classify(sequence, label)
print(prob_label_is_true)
