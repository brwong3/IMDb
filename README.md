# 🎬 IMDB Sentiment Classifier  

A binary sentiment analysis project that classifies IMDB movie reviews as **positive** or **negative** using traditional machine learning models and text feature extraction.

---

## 🧠 Overview  

This project builds and evaluates multiple classifiers on 50,000 IMDB movie reviews.  
We transformed text into numeric vectors using **Bag-of-Words (CountVectorizer)** and trained several models to compare performance:  

- **k-Nearest Neighbors (kNN)**  
- **Logistic Regression**  
- **Multi-Layer Perceptron (MLP)**  
- **Linear SVM (LinearSVC)**  

Hyperparameter sweeps and validation experiments were performed to reduce overfitting and optimize bias–variance tradeoffs.  

---

## 🧩 Key Results  

- **Best accuracy:** Logistic Regression and MLP  
- **Most stable model:** Linear SVM (`C=0.1` after tuning)  
- **Insight:** Lowering SVM regularization reduced overfitting and improved generalization.  
- **Observation:** All models struggled with ambiguous or mixed-sentiment reviews — motivating exploration of contextual embeddings (e.g., BERT).  

---

## ⚙️ Tools and Libraries  

- **Language:** Python  
- **Libraries:** scikit-learn, NumPy, pandas, matplotlib  
- **Techniques:** CountVectorizer, model selection, hyperparameter tuning, learning-curve analysis  

---

## 👩‍💻 My Contribution  

- Implemented and tuned **LinearSVC**, optimizing `C=0.1` for best bias/variance balance.  
- Co-authored the **Results** and **Insights** sections of the report.  
- Analyzed overfitting and regularization trends through learning curves and parameter sweeps.  

---

## 🚀 How to Run  

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook project.ipynb
