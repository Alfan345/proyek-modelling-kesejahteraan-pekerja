# 📊 Proyek Modelling Kesejahteraan Pekerja Indonesia

> Machine learning analysis untuk mengklasifikasikan dan mengelompokkan tingkat kesejahteraan pekerja berdasarkan data sosial ekonomi Indonesia (2002-2022)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)

---

## 📌 **Project Overview**

Proyek ini menganalisis **kesejahteraan pekerja di Indonesia** menggunakan **machine learning** dengan 2 pendekatan:

### **1️⃣ Clustering (Unsupervised Learning)**
Mengelompokkan provinsi berdasarkan karakteristik kesejahteraan pekerja menggunakan algoritma clustering untuk menemukan pola tersembunyi dalam data.

### **2️⃣ Classification (Supervised Learning)**
Memprediksi kategori kesejahteraan pekerja berdasarkan fitur-fitur sosial ekonomi seperti upah, pengeluaran, dan garis kemiskinan.

---

## 🎯 **Business Problem**

Bagaimana cara mengidentifikasi dan mengklasifikasikan tingkat kesejahteraan pekerja di berbagai provinsi Indonesia untuk membantu pengambilan keputusan kebijakan yang lebih tepat sasaran?

**Pertanyaan Kunci:**
- Provinsi mana yang memiliki pola kesejahteraan serupa?
- Faktor apa yang paling mempengaruhi kesejahteraan pekerja?
- Bagaimana memprediksi kategori kesejahteraan berdasarkan indikator ekonomi?

---

## 🚀 **Key Features**

✅ **Multi-year Analysis** - Data time series 2002-2022 (20 tahun)  
✅ **Comprehensive Dataset** - UMP, upah harian, pengeluaran, garis kemiskinan  
✅ **Two ML Approaches** - Clustering & Classification  
✅ **Exploratory Data Analysis** - Visualisasi insight mendalam  
✅ **Feature Engineering** - Pembuatan fitur baru dari data mentah  
✅ **Model Evaluation** - Metrics lengkap untuk validasi model  

---

## 📂 **Dataset Information**

### **1. minUpah.csv** (20 KB)
Upah Minimum Provinsi (UMP) per tahun
- **Columns:** `provinsi`, `tahun`, `ump`
- **Periode:** 2002-2022
- **Coverage:** Seluruh provinsi Indonesia
- **Sample:**
  ```csv
  provinsi,tahun,ump
  ACEH,2022,3166460
  SUMATERA UTARA,2022,2522610
  DKI JAKARTA,2022,4641854
  ```

### **2. rataRataUpah.csv** (7.5 KB)
Rata-rata upah harian pekerja
- **Columns:** `provinsi`, `tahun`, `upah`
- **Periode:** 2015-2022
- **Unit:** Rupiah per hari
- **Sample:**
  ```csv
  provinsi,tahun,upah
  ACEH,2022,16772
  JAWA BARAT,2022,15234
  ```

### **3. pengeluaran.csv** (243 KB)
Pengeluaran masyarakat per bulan
- **Columns:** `provinsi`, `daerah`, `jenis`, `tahun`, `peng`
- **Kategori Jenis:** MAKANAN, NONMAKANAN, TOTAL
- **Kategori Daerah:** PERDESAAN, PERKOTAAN
- **Periode:** 2007-2022

### **4. garisKemiskinan.csv** (342 KB)
Garis kemiskinan per kapita per bulan
- **Columns:** `provinsi`, `jenis`, `daerah`, `tahun`, `periode`, `gk`
- **Periode:** MARET, SEPTEMBER
- **Tahun:** 2015-2022

### **5. data_klasifikasi_kesejahteraan.csv** (1.4 MB)
Dataset hasil preprocessing untuk classification model
- **Rows:** ~10,000+
- **Features:** Multi-dimensi socioeconomic indicators
- **Target:** Kategori kesejahteraan (low/medium/high)

---

## 🔬 **Methodology**

### **Part 1: Clustering Analysis**

#### **Workflow:**
```
1. Data Collection & Integration
   ├── Merge 4 datasets (UMP, upah, pengeluaran, garis kemiskinan)
   ├── Handle missing values
   └── Feature aggregation per provinsi

2. Exploratory Data Analysis (EDA)
   ├── Distribusi UMP antar provinsi
   ├── Trend upah dari waktu ke waktu
   ├── Perbandingan perkotaan vs perdesaan
   └── Korelasi antar variabel

3. Feature Engineering
   ├── Rasio upah terhadap pengeluaran
   ├── Perbandingan UMP dengan garis kemiskinan
   ├── Growth rate UMP per tahun
   └── Standardization/Normalization

4. Clustering Algorithms
   ├── K-Means Clustering
   ├── Hierarchical Clustering
   ├── DBSCAN (optional)
   └── Elbow method untuk optimal clusters

5. Cluster Interpretation
   ├── Profiling setiap cluster
   ├── Visualisasi dengan PCA/t-SNE
   └── Business insights per cluster
```

#### **Expected Clusters:**
- **Cluster 1:** Provinsi dengan kesejahteraan tinggi (DKI Jakarta, dll)
- **Cluster 2:** Provinsi dengan kesejahteraan menengah
- **Cluster 3:** Provinsi dengan kesejahteraan rendah

---

### **Part 2: Classification Model**

#### **Workflow:**
```
1. Data Preparation
   ├── Load preprocessed dataset
   ├── Label encoding untuk target variable
   └── Train-test split (80:20)

2. Feature Selection
   ├── Correlation analysis
   ├── Feature importance ranking
   └── Remove redundant features

3. Model Training
   ├── Logistic Regression (baseline)
   ├── Decision Tree
   ├── Random Forest
   ├── Gradient Boosting
   └── XGBoost

4. Hyperparameter Tuning
   ├── GridSearchCV
   ├── RandomizedSearchCV
   └── Cross-validation (5-fold)

5. Model Evaluation
   ├── Accuracy, Precision, Recall, F1-Score
   ├── Confusion Matrix
   ├── ROC-AUC Curve
   └── Feature Importance Analysis

6. Model Deployment (Optional)
   ├── Save best model (pickle/joblib)
   └── Inference pipeline
```

---

## 💻 **How to Use**

### **Prerequisites**
```bash
- Python 3.8+
- Jupyter Notebook / Google Colab
```

### **1. Clone Repository**
```bash
git clone https://github.com/Alfan345/proyek-modelling-kesejahteraan-pekerja.git
cd proyek-modelling-kesejahteraan-pekerja
```

### **2. Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Or create requirements.txt:**
```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
jupyter==1.0.0
```

### **3. Run Notebooks**

**A. Clustering Analysis:**
```bash
jupyter notebook "[Clustering]_Submission_Akhir_BMLP_Alfan_(Updated).ipynb"
```

**B. Classification Model:**
```bash
jupyter notebook "[Klasifikasi]_Submission_Akhir_BMLP_Alfan (1).ipynb"
```

**Or use Google Colab:**
1. Upload notebook ke Google Drive
2. Buka dengan Google Colab
3. Upload datasets ke Colab environment

---

## 📊 **Expected Results**

### **Clustering Insights**
```
Cluster 0 (Kesejahteraan Tinggi):
- UMP rata-rata: > 3,500,000
- Upah harian: > 150,000
- Pengeluaran/Income ratio: < 0.6
- Provinsi: DKI Jakarta, Banten, Kalimantan Timur

Cluster 1 (Kesejahteraan Menengah):
- UMP rata-rata: 2,500,000 - 3,500,000
- Upah harian: 120,000 - 150,000
- Provinsi: Jawa Barat, Jawa Timur, Sumatera Utara

Cluster 2 (Kesejahteraan Rendah):
- UMP rata-rata: < 2,500,000
- Upah harian: < 120,000
- Provinsi: NTT, Maluku, Papua
```

### **Classification Performance**
```
Expected Model Performance:
- Accuracy: 85-92%
- F1-Score: 0.83-0.90
- ROC-AUC: 0.88-0.95

Top Features:
1. UMP (Upah Minimum Provinsi)
2. Rasio upah terhadap pengeluaran
3. Garis kemiskinan
4. Pengeluaran non-makanan
5. Trend pertumbuhan upah
```

---

## 📈 **Visualizations**

Kedua notebook menghasilkan berbagai visualisasi:

### **Exploratory Data Analysis:**
- 📊 Bar chart UMP per provinsi
- 📈 Line chart trend upah 2002-2022
- 🗺️ Heatmap korelasi features
- 📉 Box plot distribusi pengeluaran

### **Clustering Visualizations:**
- 🎯 Scatter plot clusters (PCA/t-SNE)
- 📊 Elbow curve optimal K
- 🌳 Dendrogram hierarchical clustering
- 📈 Cluster profile comparison

### **Classification Visualizations:**
- 🎯 Confusion matrix
- 📊 Feature importance chart
- 📈 ROC-AUC curve
- 🔥 Learning curve

---

## 🎓 **Skills Demonstrated**

| Category | Skills |
|----------|--------|
| **Data Science** | Exploratory Data Analysis, Statistical Analysis, Data Visualization |
| **Machine Learning** | Clustering (K-Means, Hierarchical), Classification (RF, XGBoost) |
| **Feature Engineering** | Feature creation, Transformation, Scaling, Encoding |
| **Model Evaluation** | Cross-validation, Metrics interpretation, Hyperparameter tuning |
| **Python Libraries** | Pandas, NumPy, Matplotlib, Seaborn, scikit-learn |
| **Domain Knowledge** | Socioeconomic analysis, Welfare indicators, Policy implications |
| **Communication** | Data storytelling, Insight presentation, Business recommendations |

---

## 🔮 **Business Impact**

### **Actionable Insights:**

1. **Kebijakan Upah Minimum**
   - Identifikasi provinsi yang perlu penyesuaian UMP
   - Benchmark UMP terhadap cost of living

2. **Program Kesejahteraan Sosial**
   - Targeting provinsi dengan kesejahteraan rendah
   - Alokasi bantuan sosial yang lebih efektif

3. **Investasi Infrastruktur**
   - Prioritas pembangunan di cluster kesejahteraan rendah
   - Peningkatan akses ekonomi di daerah tertinggal

4. **Monitoring & Evaluation**
   - Track perubahan cluster dari tahun ke tahun
   - Early warning system untuk penurunan kesejahteraan

---

## 📚 **Technical Details**

### **Clustering Algorithms Used:**
```python
# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_matrix = linkage(X_scaled, method='ward')
```

### **Classification Models:**
```python
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200)
gb.fit(X_train, y_train)
```

---

## 🔧 **Future Enhancements**

- [ ] Tambahkan data inflasi untuk normalisasi upah
- [ ] Implementasi model Deep Learning (Neural Networks)
- [ ] Dashboard interaktif dengan Streamlit/Dash
- [ ] Prediksi time series untuk UMP tahun berikutnya
- [ ] Geospatial analysis dengan peta interaktif
- [ ] API deployment untuk real-time prediction
- [ ] Incorporate data pendidikan dan kesehatan
- [ ] Multi-class classification dengan lebih banyak kategori

---

## 📖 **Data Sources**

Data bersumber dari:
- **BPS (Badan Pusat Statistik)** - Official government statistics
- **Kementerian Ketenagakerjaan** - Ministry of Manpower data
- **Open Data Indonesia** - Public socioeconomic datasets

**Note:** Data telah dibersihkan dan diproses untuk keperluan analisis.

---

## 📝 **Project Structure**

```
proyek-modelling-kesejahteraan-pekerja/
│
├── dataset/                                          # Raw datasets
│   ├── minUpah.csv                                  # Provincial minimum wage
│   ├── rataRataUpah.csv                             # Average daily wage
│   ├── pengeluaran.csv                              # Household expenditure
│   └── garisKemiskinan.csv                          # Poverty line data
│
├── [Clustering]_Submission_Akhir_BMLP_Alfan_(Updated).ipynb
│   ├── Data Integration
│   ├── EDA & Visualization
│   ├── Feature Engineering
│   ├── K-Means Clustering
│   ├── Hierarchical Clustering
│   └── Cluster Interpretation
│
├── [Klasifikasi]_Submission_Akhir_BMLP_Alfan (1).ipynb
│   ├── Data Preprocessing
│   ├── Feature Selection
│   ├── Model Training (RF, GB, XGB)
│   ├── Hyperparameter Tuning
│   ├── Model Evaluation
│   └── Feature Importance Analysis
│
├── data_klasifikasi_kesejahteraan.csv               # Processed dataset
└── README.md                                         # Documentation
```

---

## 🎯 **Key Findings (Expected)**

### **Clustering Analysis:**
✅ Terdapat 3 cluster utama kesejahteraan pekerja di Indonesia  
✅ DKI Jakarta dan provinsi dengan SDA kaya berada di cluster atas  
✅ Gap kesejahteraan antara perkotaan dan perdesaan signifikan  
✅ Trend UMP meningkat rata-rata 8-12% per tahun  

### **Classification Model:**
✅ Random Forest memberikan akurasi terbaik (88-92%)  
✅ UMP adalah prediktor terkuat kesejahteraan  
✅ Model dapat memprediksi kategori dengan F1-score > 0.85  
✅ Feature importance: UMP > Pengeluaran > Garis Kemiskinan  

---

## 👤 **Author**

**Alfanah Muhson**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/alfanah-muhson)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Alfan345)

---

## 📝 **License**

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 **Acknowledgments**

- **Badan Pusat Statistik (BPS)** untuk data sosial ekonomi Indonesia
- **scikit-learn team** untuk machine learning library
- **Pandas & NumPy community** untuk data manipulation tools
- **Matplotlib & Seaborn** untuk visualization library

---

## 📞 **Contact & Support**

Untuk pertanyaan, saran, atau kolaborasi:
- LinkedIn: [alfanah-muhson](https://linkedin.com/in/alfanah-muhson)
- GitHub Issues: [Create an issue](https://github.com/Alfan345/proyek-modelling-kesejahteraan-pekerja/issues)

---

**⭐ If you find this project useful, please give it a star!**

---

## 📖 **References**

1. BPS - Statistik Upah Minimum Provinsi
2. Kementerian Ketenagakerjaan RI - Data Ketenagakerjaan
3. Scikit-learn Documentation - [https://scikit-learn.org](https://scikit-learn.org)
4. Pandas User Guide - [https://pandas.pydata.org](https://pandas.pydata.org)

---

## 🎓 **Learning Outcomes**

Proyek ini mendemonstrasikan:
- ✅ **End-to-end ML workflow** dari data mentah hingga insights
- ✅ **Multiple ML paradigms** (supervised & unsupervised)
- ✅ **Real-world social impact** analysis
- ✅ **Production-quality documentation**
- ✅ **Business acumen** dalam interpretasi hasil

---

**💡 Tip:** Jalankan notebook secara berurutan untuk hasil terbaik. Pastikan semua dataset tersedia di folder `dataset/`.
