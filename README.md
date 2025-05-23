## An empirical study of the effect of background data size on the stability of SHapley Additive exPlanations (SHAP) for deep learning models
### Please read our [article](https://arxiv.org/pdf/2204.11351) for further information.
- [Background](#background)
- [Results](#results)
- [Datasets](#datasets)
- [Functions and pipeline](#functions-and-pipeline)

### Background
Nowadays, the interpretation of why a machine learning (ML) model makes certain inferences is as crucial as the accuracy of such inferences. Some ML models like the decision tree possess inherent interpretability that can be directly comprehended by humans. Others like artificial neural networks (ANN), however, rely on external methods to uncover the deduction mechanism. SHapley Additive exPlanations (SHAP) is one of such external methods, which requires a background dataset when interpreting ANNs. Generally, a background dataset consists of instances randomly sampled from the training dataset. However, the sampling size and its effect on SHAP remain to be unexplored. 
### Results
In our empirical study on the MIMIC-III dataset, we show that the two core explanations - SHAP values and variable rankings fluctuate when using different background datasets acquired from random sampling, indicating that users cannot unquestioningly trust the one-shot interpretation from SHAP. Luckily, such fluctuation decreases with increase of the background dataset size. Also, we notice an U-shape in the stability assessment of SHAP variable rankings, demonstrating that SHAP is more reliable in ranking the most and least important variables compared to moderately important ones. Overall, our results suggest that users should take into account how background data affects SHAP results, with improved SHAP stability as the background sample size increases.
### Datasets
Two simulated datasets (21 features and 1 binary outcome) are generated for model development, SHAP explanation, and istability quantification. `train_data` works for model development and background data sampling. `valid_data` is used to determine the optimal model and involves all samples to be explained by SHAP.
### Functions and pipeline
Here we provide open-source code for SHAP fluctuations caused by different background data with the same sample size.
The four python files `MLP`, `SHAP`, `BLUE`, and `Jaccard` constitute the 4-step reprofuction process for developing a MLP model on a simulated tabular data, generating SHAP-based global feature rankings, and quantifying such instabilities by BLUE and Jaccard index.
- STEP (i): Run the `MLP` file. In this file, we will establish a three-layer MLP based on the simulated `train_data`. Cross entropy loss and Adam optmizer are introdueced for model training. After training iterations, we will get 100 candidate models and select the optimal one based on the `valid_data`.
- STEP (ii): Run the `SHAP` file. In this file, SHAP will be used to interpret the optimal model's behavior on the `valid_data`. As indicated above, a background dataset is necessary for the SHAP calculation. Generally, people randomly sample some data from the training data, which leads to the fluctuations of SHAP-based global feature rankings. Here we samples 100 sets of background data and in each set, there are 50 samples. Based on each set, we will have a global feature ranking. Finally, we will get 100 rankings from 100 sets of background data.
- STEP (iii): Run the `BLUE` file. Based on the 100 rankings from STEP (ii), we will quantify the instabilities of SHAP-based feature rankings by the whole rankings-based BLUE or the quartile-based BLUE. BLUE is applied for exact comparison between two rankings.
- STEP (iv): Run the `Jaccard` file. Based on the 100 rankings from STEP (ii), we will quantify the instabilities of SHAP-based feature rankings by the quartile-based Jaccard index. Different from BLUE, Jaccard index is used for evaluating the fuzzy similarity between variable rankings.
