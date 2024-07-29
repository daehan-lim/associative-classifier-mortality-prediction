# Associative Classifier for Patient Mortality Prediction

## Overview

This project implements a novel associative classifier designed to predict patient mortality using electronic health records (EHRs). The classifier is specifically tailored to handle highly unbalanced datasets, a common challenge in healthcare data analysis. The implementation outperforms comparable methods such as CBA, CMAR, and decision tree classifiers in terms of F1 score and ROC AUC on a real-world hospital dataset.

## Key Features

- Custom associative classifier implementation
- Effective handling of class imbalance
- Rule pruning strategy for improved interpretability
- Performance comparison with established methods (CBA, CMAR, Decision Trees)
- Application to real-world healthcare data (eICU Collaborative Research Database)

## Background: Associative Classifiers

Associative classifiers are a type of supervised learning algorithm that combines association rule mining with classification. They are particularly valuable in healthcare applications due to their high interpretability and readability. Unlike "black box" models such as neural networks, associative classifiers generate rules that can be easily understood and validated by domain experts.

The general process of an associative classifier involves:

1. Mining frequent itemsets from the training data
2. Generating classification rules from these itemsets
3. Pruning the rule set to remove irrelevant or redundant rules
4. Using the remaining rules to classify new instances

This implementation introduces novel techniques in rule generation and pruning to handle class imbalance and improve overall performance.

## Algorithm Overview

The classifier works in two main phases:

1. **Training Phase:**
   - Generates frequent itemsets using the Apriori algorithm
   - Creates classification rules based on an interestingness measure
   - Prunes rules using a novel strategy based on rule strength and F1 score

2. **Prediction Phase:**
   - Applies the top k rules that cover a given instance
   - Predicts the class based on the accumulated scores of these rules

## Dataset

This project uses data derived from the eICU Collaborative Research Database. Specifically, the experiments were conducted on a dataset obtained from a single hospital within the eICU database, identified as Hospital ID 420. The dataset consists of:

- 4,615 patient records
- 147 drug features
- Binary class label (0 for survival, 1 for death)

The data was preprocessed to include, for each patient, a record of all the drugs they received during their hospital stay, along with the binary class label denoting the patient's survival outcome upon discharge. The dataset exhibits significant class imbalance, with approximately 90.6% of patients surviving and 9.4% not surviving.

### Accessing and Preparing the Data

To reproduce this work:

1. Access the eICU Collaborative Research Database \[1\]:
   - Visit [https://eicu-crd.mit.edu/](https://eicu-crd.mit.edu/)
   - Follow the instructions to complete the required training course and request access

2. Once you have access, follow these steps to prepare the dataset:
   - Select data from Hospital ID 420
   - Extract patient records, including all administered drugs
   - Create a binary class label for each patient (0 for survival, 1 for death)
   - Format the data into a CSV file titled `dataset.csv`, where each row represents a patient, columns represent drugs (binary indicators for administration), and the last column is the class label

Note: Due to data use restrictions, the preprocessed dataset used in this study cannot be directly shared. However, following the steps above will allow you to create an identical dataset for reproduction or further research.

## Requirements

To run this project, Python 3.7+ and the following libraries are needed:

- pandas
- numpy
- scikit-learn
- tabulate
- daehan_mlutil
- mlxtend

These dependencies can be installed using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Code Structure

- `main.py`: Script for running the full evaluation (10 runs with different random seeds)
- `model.py`: Contains the core implementation of the associative classifier
- `apriori_mlx.py`: Modified version of mlxtend's Apriori algorithm
- `associative_classifier_demo.ipynb`: Jupyter notebook demonstrating the classifier on a single run

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/daehan-lim/associative-classifier-mortality-prediction.git
   cd associative-classifier-mortality-prediction
   ```

2. Create a folder named `data/` in the project structure and move the `dataset.csv` file (prepared as described in the Dataset section) to this folder.

3. Install the required dependencies as described in the Requirements section.

4. To run the full evaluation (10 runs with different random seeds):
   ```
   python main.py
   ```

5. For a quick demonstration of the classifier on a single run:
   ```
   jupyter notebook associative_classifier_demo.ipynb
   ```
   Open and run all cells in the notebook.

## Results

The classifier achieved the following performance metrics on the test set:

- F1 Score: 0.5396
- ROC AUC: 0.7553

These results outperform CBA, CMAR, and a decision tree classifier on the same dataset.

## Interpretability

One of the key advantages of this approach is the interpretability of the generated rules. For example, a rule like `{ACETAMINOPHEN, GLUCOSE} → survived` suggests that patients administered both Acetaminophen and Glucose are more likely to survive. This type of insight can be invaluable for healthcare professionals in understanding and tailoring treatment plans.

For a detailed analysis of the results and a comparison with other methods, please refer to the original paper [2].

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The eICU Collaborative Research Database for providing the dataset
- The authors of CBA and CMAR for their foundational work in associative classification

## Contact

For any questions or feedback, please open an issue on this GitHub repository.

## References

[1] T. J. Pollard, A. E. W. Johnson, J. Raffa, and O. Badawi, “The eICU Collaborative Research Database.” physionet.org, 2017.

[2] P. A. Eng Lim and C. H. Park, “Associative Classifier Applied to Medical Data”, presented at the Proceedings of the Korea Computer Congress, Jun. 2023.
