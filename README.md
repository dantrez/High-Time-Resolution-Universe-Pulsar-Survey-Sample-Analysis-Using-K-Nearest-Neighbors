# [Project 1: High Time Resolution Universe Pulsar Survey Sample Analysis Using K Nearest Neighbors]
  Pulsars are very rapidly rotating neutron stars which emit beams of electromagnetic radiation from their magnetic poles. This results in "pulses" in the radio spectrum when seen from Earth. General relativity and alternative theories of gravity can be tested with these massive, compact objects. 
  Data (HTRU2) was obtained from the the High Time Resolution Universe South Low Latitude (HTRU-S LowLat) pulsar survey, conducted with the Parkes 64-m Radio Telescope in Australia. Candidate signsls are averaged over many rotations of the pulsar, which is determined by the length of an observation. These signals are mainly composed of RFI noise, making the discovery of real pulsars difficult. 
  Evaluations of the initial survey data, coupled with the performance characteristics of K-Nearest Neighbours are presented in a binary classification solution. The true pulsar examples of the catalog sample encompass a minority positive class (1,639 out of 16,259 total observations) due to radio frequency interference. The independent variables comprise the mean, standard deviation, excess kurtosis and skewness of the integrated profile. Hyperparameter tuning included K-Fold Cross Validation and GridSearchCV (with best parameters for n_neighbors). 

Summary:

  Initial data analyses using Seaborn showed effective correlations between Skewness (60) and Kurtosis (up to 7.5). Other observable correlations with Skewness, to a lesser extent, were found (Mean, Std_Dev, Mean_DM, Kurtosis_DM, Dev_DM). 
  After the dataset was split into Training and Test (80/20 split), feature scaling was performed to normalize the range of independent variables. 
KNeighborsClassifier was used with a starting point of 5. An accuracy score of 98.15642458100559 was obtained.
Hyperparameter tuning was initialized to avoid overfitting. The performance measure reported by k-fold cross-validation resulted in a mean score of 97.04997330052368 per cent. 
GridSearchCV resulted in best parameters for "n_neighbors" of 17. This was subsequently applied to a K-Neighbors classifier with a resultant score of 97.32931540044529; an optimization of .28700715583204 per cent. 
  Of the  three non-parametric classifiers, Random Forest (RF), k-Nearest Neighbor (kNN), and Support Vector Machine (SVM), kNN was found to produce the highest accuracy for this dataset. 

Citation:

R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656 
