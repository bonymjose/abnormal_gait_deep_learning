# Exploring the Potential of Deep Learning in Pathological Gait Detection

## Overview
This repository contains the code for my research on utilizing Deep Learning algorithms, including Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and their hybrids, for gait classification. The project focuses on analyzing skeletal and silhouette images, as well as a multimodal approach combining both, to identify normal and abnormal gaits.

# Research Questions Explored

This project aims to investigate several critical aspects of gait classification models using Deep Learning. The key research questions addressed are:

1. **Data Volume for Training**: What is the ideal amount of data needed to train a gait classification model that balances both accuracy and efficiency effectively?
2. **Optimal Input Form**: Among skeletal images, silhouette images, and their combination, which type of input provides the most precise results in gait classification models?
3. **Model Comparison - LSTM vs. CNNs**: In the realm of publicly available gait datasets, how does the effectiveness of LSTM or other RNNs on sequential data compare to the use of CNNs on image data?
4. **Hybrid Model Performance**: Does a hybrid approach that merges LSTM and CNN capabilities significantly improve the efficacy of gait classification models?


## Dataset
The models were trained and tested on publicly available datasets. 

## Key Findings
- **Optimal Data Volume**: Identified that approximately 20 skeletal images per folder per individual is an efficient sample size for training, balancing accuracy and data economy.
- **Efficacy of Image Types**: Demonstrated that skeletal images outperform silhouette and multimodal methods in diagnostic accuracy.
- **Model Performance Comparison**: Found CNN-based models to be more efficient and accurate compared to LSTM-based and hybrid models.
- **Hybrid Model Exploration**: Investigated hybrid models combining CNNs and LSTMs, which showed potential but did not consistently outperform pure CNN or LSTM models.
- **Limitations**: The study encountered limitations including dataset representativeness, data imbalances, computational resources, and potential exploration of other Deep learning architectures.

