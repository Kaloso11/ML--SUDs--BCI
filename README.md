# Unsupervised Learning for Substance Use Disorders (SUDs)

This repository contains research code for analyzing substance use disorders (SUDs) using unsupervised machine learning techniques. The work integrates data from **clinical assessments** and **EEG experiments** to uncover latent patterns and relationships between brain activity and SUD severity.

---

## 🧠 Project Structure

```
Unsupervised-Learning--SUDs/
│
├── clinical_assessments/       # Scripts for processing NSDUH clinical data
├── eeg_experiments/            # EEG signal processing and feature extraction
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── visualization/
│   └── analysis/
├── modeling/                   # Clustering, dimensionality reduction, etc.
├── common/                     # Shared utilities and helper functions
├── experiments/                # Experimental pipelines and scripts
├── notebooks/                  # (Optional) Notebooks for exploration
├── data/                       # (Optional) Raw and processed EEG/clinical data
└── docs/                       # (Optional) Project documentation and diagrams
```

---

## 🛠 Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Or create a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔍 Key Modules

- **clinical_assessments**: Parses and analyzes NSDUH-based clinical data.
- **eeg_experiments**: Handles EEG signal processing, DWT, PSD, ICA, ERP waveform analysis, and topographic mapping.
- **modeling**: Applies unsupervised models (e.g., k-means, DBSCAN, PCA) to clustered outputs.
- **experiments**: Main experimental pipelines integrating features and models.
- **common**: General utilities for batching, preprocessing, and shared tasks.

---

## 📊 Results

Results include:
- Feature maps from EEG signals
- Cluster groupings based on clinical indicators
- Comparisons across AUD, CaUD, CoUD

(You can include plots or metrics here.)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
