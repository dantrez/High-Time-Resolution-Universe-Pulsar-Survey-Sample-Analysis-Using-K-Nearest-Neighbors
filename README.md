# [Project 1: High Time Resolution Universe Pulsar Survey Sample Analysis Using K Nearest Neighbors](https://github.com/dantrez/dantrez_projects/blob/main/High%20Time%20Resolution%20Universe%20Pulsar%20Survey%20Sample%20Analysis%20Using%20K-Nearest%20Neighbors.ipynb)
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

![](https://github.com/dantrez/dantrez_projects/blob/main/images/Vela_Pulsar_jet.jpg?raw=true)

# [Project 2: Artificial Neural Network for Classification and Inspection of SDSS Dataset with Stars, Galaxies, and Quasars](https://github.com/dantrez/dantrez_projects/blob/main/ANN%20SDSS%20Astronomy%20Classification%20Model%20of%20Quasar%20Stars%20and%20Galaxies.ipynb)
The Sloan Digital Sky Survey uses a dedicated 2.5 m wide-angle optical telescope in New Mexico, United States, to conduct spectroscopic surveys started in 1998. The dataset used in this model is the 16th, released after August 2018. Astrophysicists of the Sloan Digital Sky Survey published the largest, most detailed 3D map of the universe so far, filled a gap of 11 billion years in its expansion history, and provided data which supports the theory of a flat geometry of the universe. It confirms that different regions seem to be expanding at different speeds.
  Each row of CCD has a different optical filter with average wavelengths for u, g, r, i, z bands.

Summary:

  Data cleaning (cleansing) was performed initially, with the "class" column moved into the last column. An optimization-based approach was augmented to handle missing values with the Python SimpleImputer. Mandatory Feature Scaling (for voluminous SDSS data in a neural network) resulted in a convergence of the weights for improved performance. LabelEncoder transformed the dependent variables into dummy variables with the one hot encoding process.  
  Observations fit a Gaussian distribution (bell curve) with a well behaved mean and standard deviation. Seaborn data analysis was performed. Pairplots showed correlations or lack thereof and an interactive 3D surface plot was provided to explore interactions amongst the different filter curves of u, g, r, i, and z bands.
  A SELU activation function was utilized in the first layer, resulting in a slightly improved accuracy over Rectified Linear, Sigmoid, . SELU multiplies scale to ensure a slope greater than 1 for positive inputs. Softmax was used as the activation for the last layer of a classification network as the result could be interpreted as a probability distribution. It is computed as exp(x) / tf.reduce_sum(exp(x)). 
  Nadam optimization is an improved-upon version of Adam, employing Nesterov momentum. It improved epochal accuracy by approximately .015 per cent.

  Overall performance of this simplified neural network is comparable to that of random forest, multiple logistic regression, or decision tree. Test accuracy of 98.77% and 3X3 confusion matrix scored 19,755 correct predictions out of 20,000 objects. Because of noise and the difficult spectroscopic boundaries inherent in astronomy and the pointlike or extended (galaxy) objects observed, Increasing the number of nodes
in a layer, adjusting the optimizer, and epoch adaption resulted in a well-performing model useful for classification studies. 

Citation:

    Funding for the Sloan Digital Sky Survey (SDSS) has been provided by the Alfred P. Sloan Foundation, the Participating Institutions, the National Aeronautics and Space Administration, the National Science Foundation, the U.S. Department of Energy, the Japanese Monbukagakusho, and the Max Planck Society. The SDSS Web site is http://www.sdss.org/.

    The SDSS is managed by the Astrophysical Research Consortium (ARC) for the Participating Institutions. The Participating Institutions are The University of Chicago, Fermilab, the Institute for Advanced Study, the Japan Participation Group, The Johns Hopkins University, Los Alamos National Laboratory, the Max-Planck-Institute for Astronomy (MPIA), the Max-Planck-Institute for Astrophysics (MPA), New Mexico State University, University of Pittsburgh, Princeton University, the United States Naval Observatory, and the University of Washington.
    
![](https://github.com/dantrez/dantrez_projects/blob/main/images/699444main_QSO.jpg?raw=true)

# [Project 3: Exploratory Data Analysis (EDA) Using Financial Data In A Portfolio](https://github.com/dantrez/dantrez_projects/blob/main/Exploratory%20Data%20Analysis%20(EDA)%20Using%20Financial%20Data%20In%20A%20Portfolio.ipynb)

