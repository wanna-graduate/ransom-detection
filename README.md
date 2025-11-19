# Early Detection of Ransomware via Multi-Source Fusion with Hardware Performance Counters

A ransomware detection and analysis toolkit based on multiple data sources, including HPC performance counters, disk I/O behavior, and ransom note detection.

---

## 📁 Repository Structure

This repository currently contains the following modules:

- `classifier for HPC data/`  
  Code related to classifiers that use **hardware performance counter (HPC) data** for malicious behavior detection (feature extraction, training, testing scripts, etc.).

- `classifier for disk-io/`  
  Code related to ransomware detection models based on **disk I/O behavior features**.

- `dataset/`  
  Datasets and their organization:
  - `perf/`: **HPC-related data**, including  
    - Training and test sets of known ransomware;  
    - Training and test sets of unknown ransomware;  
    - Corresponding benign samples.  
  - `blk/`: **Block I/O–related data**, used to train and evaluate the disk I/O behavior classifier.

- `ransom note detection/FsFilter1/`  
  A **ransom note detection module** based on a File System Filter driver, used to intercept or log ransom-note-related operations at the system level.

- `others/`  
  Auxiliary tools for **data collection and preprocessing**, such as collecting HPC counters, recording I/O logs, format conversion, and feature cleaning.

- `LICENSE`  
  The open-source license for this project.

- `README.md`  
  This document.

---

## ✨ Features

1. **HPC Behavior Classifier**
   - Trains classification models using CPU performance counter data (e.g., cache misses, branch mispredictions);
   - Supports binary classification (e.g., Benign vs. Ransomware);
   - Suitable for performance-sensitive high-performance computing (HPC) scenarios.

2. **Disk I/O Behavior Classifier**
   - Extracts features from I/O request patterns, file access frequency, write patterns (sequential/random, massive small-file writes, etc.);
   - Detects typical ransomware behaviors such as bulk encryption and abnormal write patterns.

3. **Ransom Note Detection (FsFilter)**
   - Monitors file creation and modification via a file system filter driver;
   - Identifies suspicious ransom notes (file name keywords, directory locations, content patterns, etc.), which can be used for early warning or forensic support.

---

## 🔧 Environment & Dependencies

> **Note:** The following is an example. Please adjust the dependency list and versions according to your actual implementation.

### Common Dependencies

- Python ≥ 3.8  
- Typical data science / machine learning libraries (depending on your code), such as:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `pytorch` 

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
