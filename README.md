# 📊 **Cross-Asset Correlation Anomaly Detector**

### *A professional tool for dynamic correlation analysis, anomaly detection, clustering & cointegration across financial assets.*

# 🇬🇧 **English Version**

## 🚀 Overview

The **Cross-Asset Correlation Anomaly Detector** is a robust analytics application built in Python and Streamlit that allows users to:

* Analyze **rolling correlations** between multiple assets
* Detect **correlation anomalies** using Z-score deviations
* Visualize a **correlation heatmap**
* Perform **dynamic pairwise correlation analysis**
* Compute **hierarchical clustering** of assets based on correlation structures
* Test for **cointegration relationships** and visualize spreads
* Export all results (matrices, heatmaps, spreads) in **Excel / PDF**

Designed for portfolio analysts, quantitative researchers, traders, and data-driven investors.

---

## 🧩 Key Features

### ✅ **1. Overview & Heatmap**

* Downloads historical prices from Yahoo Finance
* Computes log returns
* Builds a correlation matrix
* Shows a ranked anomaly table (Z-scores)
* Displays a clean heatmap with export options

---

### 🔄 **2. Dynamic Correlation Viewer**

Analyze rolling pairwise correlations:

* Choose any two assets
* Visualize rolling correlation across time
* Overlay Z-scores
* Detect periods of correlation instability
* Download the chart as PDF or PNG

---

### 🧬 **3. Clustering (Hierarchical)**

Groups assets based on correlation similarity:

* Distance metric: `1 − correlation`
* Ward linkage hierarchical clustering
* Reordered correlation matrix revealing asset clusters
* Ideal for risk-parity, diversification or factor grouping
* Includes exportable clustered matrix and heatmap

---

### 🔗 **4. Cointegration Testing**

Test long-term equilibrium relationships:

* Engle-Granger cointegration test
* Regression spread and Z-score
* Cointegration interpretation guidance
* Downloadable spread series and Z-scores

---

## 🛠️ Tech Stack

* **Python 3.11+**
* **Streamlit** (UI framework)
* **Pandas / NumPy** (data manipulation)
* **Matplotlib / Seaborn** (visualizations)
* **Scipy / Statsmodels** (statistical modelling)
* **Scikit-Learn** (clustering)
* **Yahoo Finance API** (market data)

---

## 📦 Installation

```bash
git clone https://github.com/marcbagur20/cross-asset-correlation-detector.git
cd cross-asset-correlation-detector
pip install -r requirements.txt
```

---

## ▶️ Run the application

```bash
streamlit run src/app.py
```

The app will launch automatically in your browser.

---

## 📁 Project Structure

```
cross-asset-correlation-detector/
│
├── src/
│   ├── app.py                 # Main Streamlit app
│   ├── utils.py               # Data processing & analytics functions
│
├── notebooks/
│   └── exploratory/           # Research notebooks
│
├── assets/                    # (Optional) Images for README
│
└── README.md
```

---

## 📤 Export Options

| Feature            | Excel | PDF |
| ------------------ | ----- | --- |
| Correlation Matrix | ✅     | —   |
| Clustered Matrix   | ✅     | —   |
| Heatmap            | —     | ✅   |
| Spread & Z-Series  | ✅     | —   |

---

## 🧭 Use Cases

✔ Portfolio diversification analysis
✔ Detecting structural market regime changes
✔ Stress-testing correlation assumptions
✔ Pair selection for stat-arb strategies
✔ Factor clustering and risk decomposition
✔ Quant research and market studies

---

## 🤝 Contributing

Pull requests are welcome!
If you want to extend the tool (e.g., PCA factors, volatility clustering, or ETFs vs indices), feel free to open an issue.

---

## 📜 License

This project is licensed under the **MIT License**, meaning you're free to use and modify it with attribution.

---
