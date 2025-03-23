# Classification of DDoS Detection and Mitigation dataset Using Deep Learning

## 📄Overview of DDoS Attacks
A Distributed Denial of Service (DDoS) attack is a malicious attempt to disrupt normal traffic by overwhelming a target system, such as a server or network, with a flood of traffic. These attacks exploit vulnerabilities in network protocols, application services, or infrastructure, leading to service disruption and financial losses.


## 📊About the Dataset
The **DDoS Detection and Mitigation Dataset** is designed to help in the development of machine learning models for detecting and mitigating DDOS attacks. It contains both **benign network traffic** and **various types of DDoS attack traffic**, collected using **Mininet** and an **SDN Controller**.

### Features of the dataset:
- Simulated attack and normal traffic data
- Packet-level features extracted for analysis
- Suitable for training machine learning models to classify network traffic

## Data Collection Process
The dataset was collected using a **Software-Defined Networking (SDN)** environment:
1. **Mininet** was used to create a virtual network topology.
2. **An SDN Controller** managed traffic flow.
3. **Custom attack scripts** simulated various DDoS attacks.
4. **Traffic monitoring tools** (like Wireshark) captured network flow data.
5. **Feature extraction** was done to prepare the dataset for machine learning.

## Preprocessing Steps
1. Load the dataset from a CSV file.
2. Drop non-numeric and unnecessary columns.
3. Handle missing and infinite values by replacing them with NaN and dropping them.
4. Encoded non numeric columns and picked important features based on their correlation.
5. Normalize numerical features using MinMaxScaler.
6. Shuffle the dataset for better generalization.
7. Split the dataset into training (80%) and testing (20%) sets.

## Getting started
To set up the environment and install dependencies, use the following:

### Requirements
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Installation
Run the following command to install the required libraries:
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
```
## Model Development
   Deep learning models are implemented using TensorFlow/Keras, which includes:  
   - **LSTM (Long Short-Term Memory)**   
   - **CNN (Convolutional Neural Network)**  
   - **ANN (Artificial Neural Network)**
## Training Process
- Loss function: Binary Cross-Entropy
- Metrics: Accuracy, Precision, Recall
- Training for 5 epochs with a batch size of 128
- ANN was tested on the original test set, adversarial test set, and combined dataset
## FGSM Adversarial Samples Generation
- Base model was trained to generate adversarial Samples using FGSM with ε=0.1. The adversarial dataset was saved as adversarial_dataset.csv
- Original and adversarial datasets were combined into combined_dataset.csv
## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Visualization
## Visualizations
- Confusion Matrices (Original Test Set, Adversarial Dataset, Combined Dataset)
- ROC curves for the original, adversarial, and combined datasets
- Bar charts comparing performance metrics across datasets

## Results

![WhatsApp Image 2025-03-22 at 15 55 33_33278812](https://github.com/user-attachments/assets/d536bb55-d667-4f34-8e46-520b6693ef6c)

![Screenshot 2025-03-21 191217](https://github.com/user-attachments/assets/3b54aab7-138b-4305-ac33-dd0d288f042c)

![Screenshot 2025-03-21 190554](https://github.com/user-attachments/assets/ea3cc793-2a7f-4154-b482-bb914bee8d7a)




## Contributors

<p align="left">
  <a href="https://github.com/sruthi070">
    <img src="https://avatars.githubusercontent.com/u/154976021?v=4" width="50px" height="45px" style="border-radius: 50%;" />
  </a>
  <a href="https://github.com/sravyapalla">
    <img src="https://avatars.githubusercontent.com/u/143865378?v=4" width="50px" height="45px" style="border-radius: 50%;" />
  </a>
  <a href="https://github.com/Siddu230">
    <img src="https://avatars.githubusercontent.com/u/155234070?v=4" width="50px" height="45px" style="border-radius: 50%;" />
  </a>
</p>

