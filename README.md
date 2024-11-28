# CS6235KO Knowledge Obsolescence Structured Project

**Authors:** Hongzhen Liang, Huanzhi Pu  
**Project Objective:** Explore and address the issue of knowledge obsolescence in automated fake news detection using advanced machine learning pipelines and the Snorkel framework.

---

## Project Overview

This project involves the structured training, integration, and evaluation of machine learning models to detect fake news effectively. It leverages multiple datasets and the Snorkel pipeline to assemble and improve model performance through label integration.

---

## Steps and Instructions

### Step 1: **Dataset Setup (Skipped)**  
The datasets required for this project are pre-included. No additional setup is necessary. Relevant datasets:  

- **NELA Dataset:** [Access here](https://gtvault-my.sharepoint.com/personal/khu83_gatech_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkhu83%5Fgatech%5Fedu%2FDocuments%2FNELA&ga=1&LOF=1)  
- **Old FNC Dataset:** [Access here](https://gtvault-my.sharepoint.com/personal/khu83_gatech_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkhu83%5Fgatech%5Fedu%2FDocuments%2FOld%20FNC&ga=1&LOF=1)  

---

### Step 2: **Train Four Models on NELA and Old FNC Datasets**

1. Train four distinct models on two datasets:  
   - **2 Models** on the NELA dataset.  
   - **2 Models** on the Old FNC dataset.  

2. Detailed instructions for training are provided in the [Follow2/README.md](Follow2/README.md). Ensure proper preprocessing and configuration as described.

---

### Step 3: **Integrate Four Models Using Snorkel**  

1. Use the **Snorkel framework** to assemble the four models trained in Step 2.  
2. The integration process combines the predictive capabilities of the individual models into a unified labeling pipeline.  

Refer to the detailed guidance in the [Follow3/README.md](Follow3/README.md).

---

### Step 4: **Train Two Models with New FNC Labels**  

1. Train two additional models using different labeling strategies:  
   - One with **earlier New FNC labels** (produced from Snorkel integration).  
   - One with **manually verified labels** for higher accuracy and reliability.  

2. For implementation details, refer to [Follow4/README.md](Follow4/README.md).

---

### Step 5: **Develop Two Advanced Snorkel Pipelines**  

1. Assemble the four models from Step 2 and combine them with one of the models from Step 4.  
2. Build and evaluate **two advanced pipelines** using Snorkel for improved performance and adaptability.  

For precise instructions, refer to [Follow5/README.md](Follow5/README.md).

---

## Additional Notes

- Ensure that all datasets are preprocessed correctly before training to avoid discrepancies in model performance.
- Utilize the provided `README.md` files in each Followup folder for detailed implementation guidelines.
- Monitor the performance of each pipeline to validate improvements in label accuracy and detection efficiency.

This README provides a structured roadmap for successfully completing the CS6235KO Knowledge Obsolescence project.
