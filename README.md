# [Project 1: NDVI Landsat 8 Rasterio crops image and ecosystem analysis of Saginaw Bay region](https://github.com/dantrez/dantrez_projects/blob/main/NDVI%20Landsat%208%20Rasterio%20crops%20image%20and%20ecosystem%20analysis%20of%20Saginaw%20Bay%20region.ipynb)
  Satellite-based multispectral imagery is a major source of data enabling computation of NDVI and other vegetation indices for large cultivated areas. Live green plants appear relatively dark in the RED wavelengths and relatively bright in the NIR (near-infrared). By contrast, clouds and snow tend to be rather bright in the red (as well as other visible wavelengths) and quite dark in the near-infrared. The pigment in plant leaves, chlorophyll, strongly absorbs visible light (from 400 to 700 nm) for use in photosynthesis. The cell structure of the leaves can strongly reflect near-infrared light (from 700 to 1100 nm). The more leaves a plant has, the more these wavelengths of light are affected. 

  NDVI (Normalized Difference Vegetation Index) is a valuable and popular way to understand vegetation health and land use using satellites or remote sensing. It is calculated as follows:

    NDVI = ( NIR − Red ) / ( NIR + Red ) 

  where Red and NIR stand for the spectral reflectance measurements acquired in the RED (visible) and near-infrared (NIR) regions. These spectral reflectances are themselves ratios of the reflected radiation to the incoming radiation in each spectral band individually; hence they take on values between 0 and 1. By design, the NDVI itself thus varies between -1 and +1. NDVI is functionally equivalent to the simple infrared/red ratio (NIR/VIS). 

The model, as a helpful tool, uses Python libraries rasterio and GDAL, designed to work with geospatial raster data. Several software programs use the GDAL/OGR libraries to allow them to read and write multiple GIS formats. A dataset download from the U.S. Geological Survey covers the Saginaw Bay region of Michigan for August 2021 from 203085 meters to 440115 meters, left to right, and 4662885 meters to 4903515 meters bottom to top. The Landsat8 Band 4 is in the 0.64 - 0.67 µm while Band 5 is in 0.85 - 0.88 µm. The images are 30-meter multi-spectral spatial resolutions along a 185 km (115 mi) swath and the satellite has a 16-day repeat cycle with an equatorial crossing time of 10:00 a.m. +/- 15 minutes. This system is used for mapping areas in the Northern Hemisphere. After downloading, cleaning, inspection, and analysis of two bands, namely Band4 for RED wavelengths and Band5 for NIR, NDVI plot boundaries show “EPSG 32617” identifies a particular coordinate reference system: UTM zone 17N. These coordinate values are relative to the origin of the dataset’s coordinate reference system (CRS). 

Affine transformation was performed on the Band4 image to illustrate its typical use to correct for geometric distortions or deformations that occur with non-ideal camera angles. For example, satellite imagery uses affine transformations to correct for wide angle lens distortion, panorama stitching, and image registration. Transforming and fusing the images to a large, flat coordinate system is desirable to eliminate distortion. This enables easier interactions and calculations. The affine transformation matrix maps pixel locations in (row, col) coordinates to (x, y) spatial positions. The product of this matrix and (0, 0), the row and column coordinates of the upper left corner of the dataset, is the spatial position of the upper left corner. Empty cells or nodata cells are reported as 0 to change to fractional decimals in float64 format.

Summary: 

Coniferous regions are significantly easier to predict than crop areas due to their homogeneous and regular evolution. Additionally, the examples given show field areas where vegetation was wilting or non-existent, coupled with water, soil, and clouds that do not reflect NIR well. Also, high reflectance of NIR results in the images represents intuitive green areas of healthy vegetation (brighter green is better). These bright green areas contains dense, live green vegetation. 

This whole-farm approach to crop analysis can use information technology, satellite positioning data, remote sensing and correlated data gathering in understanding crop yields or ecosystem changes. The latest technological use of advanced artificial intelligence (AI) algorithms coupled with remote sensing can also provide management-zone driven results for: nitrogen, phosphorus, potassium, magnesium, iron and other minerals.  In addition, targeted field analysis, as well as continental or global-scale vegetation can be monitored. 

![](https://github.com/dantrez/dantrez_projects/blob/main/images/NDVI%20composite%20of%20NIR%20and%20RED%20bands%20in%20false%20color-%20lighter%20is%20healthier-.jpg?raw=true)
![](https://github.com/dantrez/dantrez_projects/blob/main/images/Sample%20image%20of%20NDVI%20-%20zoomed%20false%20color.jpg?raw=true))


# [Project 2: Pix2Pix DCGAN CNN for Image-to-Image Translation of Satellite Images to Google Maps Images](https://github.com/dantrez/dantrez_projects/blob/main/Pix2Pix%20DCGAN%20CNN%20for%20Image-to-Image%20Translation%20of%20Satellite%20Images%20to%20Google%20Maps%20Images.ipynb)
 This Pix2Pix GAN will train a Deep Convolutional Neural Network (DCGAN) to perform image-to-image translation tasks and will allow for the generation of large images. In this case, it will convert satellite photos to maps. 
  The model comprises a generator for generating synthetic "map" images from complex satellite imagery. In an adversarial process, the discriminator model will identify "real" from "fake" data before sending the data back to the generator for image improvement. The image must be a plausible generated image; thus the Pix2Pix moniker. It uses L1 loss measurements to consistently refine the target image.
  Current uses are image-to-image translation tasks such as converting maps to satellite photographs (or vice-versa), black and white photographs to color, and sketches of products to product photographs for artistic rendering or production. 
  The dataset is comprised of 39 images pairs of New York satellite imagery 1200 pixels wide and 600 pixels tall, cleaned (cleansed). The training image sets were saved in a Numpy array and converted to 256x256 size. The output is dependent on the size of the image and each value is a probability for the likelihood that a patch in the input image is real. These values can be averaged to give an overall likelihood or classification score if needed.
  The generator is an encoder-decoder model utilizing a U-Net architecture with skip-connections added to avoid overfitting. A downsampling of the input image is sent to a bottleneck, emerging in an output section of equal size. The Tanh activation function is used in the output layer with pixel values in the generated image in the range [-1,1]. 
  The composite model is updated with two targets: one indicating generated images are real (cross entropy loss), forcing large weight updates in the generator (toward generating more realistic image), and the executed real translation of the image, which is compared against the output of the generator model (L1 loss).
  The Keras functional API connected the generator and discriminator. The model was saved to generate sample image-to-image translations periodically during training, such as every 10 training epochs. Image quality inspection was used at the end of training to choose a final model. 
  The number of epochs is set at 100 to keep training times down. A batch size of 1 is used as is recommended in the paper. Training involved a fixed number of training iterations. 
  The generator was saved and evaluated every 10 epochs.  The training dataset comprised 39 images to keep CPU costs down, resulting in approximately 90 minutes of CPU time on a Ryzen Vega system. 
  
  Summary:
  
  Results show the generated image captures various features well. Due to the random probability nature of running machine learning models, the author's output may vary compared to another user's. 

Citation:
 	arXiv:1611.07004 [cs.CV]
  	(or arXiv:1611.07004v3 [cs.CV] for this version)
        
  	
https://doi.org/10.48550/arXiv.1611.07004

Jason Brownlee, Ph.D.

![](https://github.com/dantrez/dantrez_projects/blob/main/images/H5%20files.jpg?raw=true)

# [Project 3: High Time Resolution Universe Pulsar Survey Sample Analysis Using K Nearest Neighbors](https://github.com/dantrez/dantrez_projects/blob/main/High%20Time%20Resolution%20Universe%20Pulsar%20Survey%20Sample%20Analysis%20Using%20K-Nearest%20Neighbors.ipynb)
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

# [Project 4: Artificial Neural Network for Classification and Inspection of SDSS Dataset with Stars, Galaxies, and Quasars](https://github.com/dantrez/dantrez_projects/blob/main/ANN%20SDSS%20Astronomy%20Classification%20Model%20of%20Quasar%20Stars%20and%20Galaxies.ipynb)
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

# [Project 5: Exploratory Data Analysis (EDA) Using Financial Data In A Portfolio](https://github.com/dantrez/dantrez_projects/blob/main/Exploratory%20Data%20Analysis%20(EDA)%20Using%20Financial%20Data%20In%20A%20Portfolio.ipynb)
 A useful template for financial portfolio analysis, this model highlights the following core analyses:

 

Standard deviation

CAPM

Sharpe Ratio

Mean Returns

Bollinger Bands

Portfolio diversification

Risk analysis

Correlation and covariance

 

  Because of the scope and nature of stock and fund data, real-time data gathering was used whenever possible. This reduced data cleaning (cleansing) significantly yet NaN values were replaced with a mean.

  Given a mathematical basis in order to determine risk, a standard deviation assumes the higher the risk, the more potential reward and provides a quantified estimate of the uncertainty of future returns. Financial time series are known to be non-stationary, whereas the statistical calculations such as standard deviation apply only to stationary series.

  Risk measurement and standard financial management strategies can include the Sharpe Ratio which is derived from the previous version called ex-ante Sharpe Ratio. The ratio is computed by differences between the returns of the investment and the risk-free return, divided by the standard deviation of the investment returns. It represents the additional amount of return that an investor receives per unit of increase in risk. Sharpe ratios, along with Treynor ratios and Jensen's alphas, can rank the performance of portfolio or mutual fund managers.

  CAPM, or otherwise known as Capital Asset Pricing Model, makes use of the security market line (SML) and its relation to expected return and systematic risk (beta) to show how the market must price individual securities in relation to their security risk class. The SML enables calculations of reward-to-risk for any security in relation to that of the overall market.

  Further, Bollinger Bands displays a graphical band (the envelope maximum and minimum of moving averages) and volatility (expressed by the width of the envelope) in one two-dimensional chart. This included graph exhibits graphical analysis of selected stocks to compare and contrast in a diversified portfolio setting. By tuning the parameters to a particular asset for a particular market environment, the out-of-sample trading signals can be improved compared to the default parameters.

  Because this model's time-series is stationary (start, end times), correlation and covariance mathematical constructs are similar in describing the degree to which two random variables (or sets of random variables) tend to deviate from their expected values.

  Note: this tool is not intended to provide financial advice and investors should always consult a professional regarding investments.

![](https://github.com/dantrez/dantrez_projects/blob/main/images/bollinger.jpg?raw=true)

# [Project 6: NLP Sentiment Analysis of Hotel Reviews Using NLTK  and VADER](https://github.com/dantrez/dantrez_projects/blob/main/NLP%20Sentiment%20Analysis%20of%20Hotel%20Reviews%20Using%20NLTK%20%20and%20VADER.ipynb)
  Natural Language Processing, or NLP in this form of Sentiment Analysis, is the supervised learning computational treatment of opinions, sentiments and subjectivity of text performed by a machine learning model. Recently instituted algorithms, methods, and enhancements are investigated and presented briefly in this model. Document-level Sentiment Analysis aims to classify an opinion as a summary. In addition, easy-reference of the data is provided in the model itself. 

  The cleansed (cleaned) dataset supplied provides a large sample of 20,491 Trip Advisor hotel reviews comprising one column and a scoring of 1-5 for each record in the other, with 5 being a positive score. An investigative analysis of text data is done to assist an organization to make data-driven decisions. 

  The model is rule-based, using the Python 3 Pandas, Matplotlib, and Numpy. In addition, NLTK (Natural Language ToolKit) is the de facto standard platform for building Python programs to work with human language data. It contains easy-to-use interfaces to many corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning. 

  NLTK supplies "stopwords", which are very common words that appear in the text carrying little meaning; they serve only a syntactic function but do not indicate subject matter. These are eliminated from the model to provide more efficiency.

  Stemming algorithms work by cutting off the end or the beginning of the word, taking into account a list of common prefixes and suffixes that can be found in an inflected word.  NLTK uses PorterStemmer; rules contained in this algorithm are divided in five different phases numbered from 1 to 5. The purpose of these rules is to reduce the words to the root.
Noise will be reduced and the results provided on the information retrieval process will be more accurate.

  Naïve Bayes is a probabilistic machine learning algorithm based on the Bayes Theorem, used in a wide variety of classification tasks. The classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other features that are inherently codependent on one another. Multinomial Naïve Bayes, Gaussian, and the Bernoulli Naïve Bayes using Scikit Learn are three variations. The Multinomial Naive Bayes algorithm (MultinomialNB) used here implements the Naive Bayes algorithm for multinomial distributed data. MultinomialNB suits our purposes as the data are typically represented as word vector counts. 

  VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. The latest update incorporated refactoring for Python 3 compatibility, improved modularity, and fused into NLTK. A valance score was added to the model to further assist in the scoring of data. It can be noted Rating did not correlate well with the sentiment from this social media accumulation. For example, a Rating of 1 coupled with a POSITIVE sentiment in contradiction. This is where VADER shows a more comprehensive study of the single record. 

  Summary

  The model shows an overall positive sentiment for the hotel studied. The relative "mood", from the deeper compound and 1-5 ratings scale combined, show favorable conditions for a traveller in the single hotel studied.  
  
![](https://github.com/dantrez/dantrez_projects/blob/main/images/sentiment%20analysis%20graphic.jpg?raw=true)

# [Project 7: Association Rule Data Mining Basket Analysis Calculations And Analysis Using FPGrowth](https://github.com/dantrez/dantrez_projects/blob/main/Association%20Rule%20Data%20Mining%20Basket%20Analysis%20Calculations%20And%20Analysis%20Using%20FPGrowth.ipynb)
  As one of the first applications of data mining, Market Basket Analysis identifies items that typically occur together in purchase transactions. 
Of most interest is the discovery of unexpected associations, which may open new avenues for marketing or research. Further, the discovery of sequential patterns, i.e. sequences of errors or warnings that precede an equipment failure may be used to schedule preventative maintenance or may provide insight into a design flaw. 

  Preprocessing included the cleaning (cleansing) of the dataset by means of converting the dataset to a list of transactions with no header columns and initializing it to an empty list. It was then sequentially populated up to 20 transactions for each transaction record, spanning 7500 records in total.

  FPGrowth and mlxtend libraries were specifically utilized to develop association rules which indicate the combinations of factors most commonly observed in retail purchases. FPGrowth finds patterns that frequently occur in a dataset in an association rule mining setting. Cross-marketing, catalog design, sale campaign analysis, and DNA sequencing are well suited for this library. 

  One-hot encoding is performed and the list is transformed into an array of True-False tuples. This array of transactions is then converted into a dataframe. 

  Minimum support values are set and can be easily adjusted, depending on use needs. Similarly, association rules thresholds may be easily set with regards to support, confidence, lift, leverage, and conviction.  The support tells us the number of times, or percentage, that the products co-occur. The confidence tells us the probability of the second item being purchased given the first item. The lift gives us the strength of association. Leverage is the difference between the ratio of coverage of both factors correlating and lift is the ratio probability of them, showing the likelihood of the association. 

  For example, the model shows  a setting for lift of 6.0. resulting in the antecedents "eggs, burger sauce" itemset strongly influencing the consequents purchase of "chicken" with a lift value of 9.72. Note how a comprehensive data mining approach can be used in this model by altering the thresholds in this manner. 

![](https://github.com/dantrez/dantrez_projects/blob/main/images/apriori.jpg?raw=true)

# Project 8: Popular Competition for House Prices Results

  A popular house-buying competition for Data Scientists to learn and utilize creative feature engineering and advanced regression techqniques was attempted. Several regression models, pipelines, and preprocessing attempts were made to fine-tune an acceptable result for the author.

  Submissions were evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. It was found that a Ridge-Regression pipeline model worked the best. Preprocessing included an EDA of the dataset, thorough analysis of the inter-relationships of the data, and imputing values while taking into account outliers resulted in the author's personal best RMSE score of .15389 (less is better), which placed at the top 63%. The median value was 12.348.   

# [Project 9b: Low Income Countries Crop Analysis Using Tableau Dashboard](https://github.com/dantrez/dantrez_projects/blob/main/Low%20income%20crops%20text.txt)

   This visualization shows food and agriculture statistics for the world's low-income countries for the period 2019-2021. The source datasets were provided by the Food and Agriculture Organization of the United Nations, providing crop production totals and the resulting values. 
  The yearly output for all countries combined increased from 2019-2020 by 2.93%. From 2020 to 2021, the increase was .8%. This result stems, in part, from swarms of desert locusts ravaging countries across the Greater Horn of Africa, the Arabian Peninsula, and Southwest Asia. Another factor was COVID-19 impacting production and farm families, along with the 2020 Atlantic hurricane season producing 30 storm systems, surpassing the usual 12-storm annual average. Effective Disaster Risk Reduction (DRR) policies can influence agriculatural production systems but are at the mercy of drought, storms, pest, diseases and wildfires. 
  A map of each country and their resultant 2019-2021 crop totals is given, along with each named crop for a given year. All crops are given although some countries will not produce a given crop, while a country nearby will. For example, South Sudan does not produce sugar cane in measurable amounts, but South Sudan does. 
  This data and visualization is in the interest of assessment of the state of food security and progress towards achieving the hunger and food insecurity targets, progress towards the global nutrition targets defined by the World Health Assembly, and bringing the prevalence of undernourishment, and the prevalence of moderate or severe food insecurity based on the Food Insecurity Experience Scale into light. 

![](https://github.com/dantrez/dantrez_projects/blob/main/images/Low%20Income%20Crops%20Dashboard.jpg?raw=true)

# [Project 10: QGIS & Cesium Ion reveal crucial insights for informed site selection and conceptual design decisions](https://github.com/dantrez/dantrez_projects/blob/main/images/Reg%20combo%20OSM%20and%20Ggle.gif)

  For a LinkedIn project, I conducted a hypothetical 3D urban analysis, seeking to identify potential sites for a new mixed-use building project. Using an OpenStreetMap layer containing street names and classifications, I was able to orient within the target neighborhood and zoom to potential locations of interest. 

Cesium Ion provided Google's 3D building tiles, enabling an immersive 3D view within QGIS software. This animated 3D Map view allowed for photo-realistic building rendering and fly-through capabilities to assess building massing and blocking characteristics from varying vantage points. 

Street-level, building height, and roof width are a sample of measurements taken for a faux real-estate analysis. 

Overall, leveraging 3D modeling and analysis tools within a GIS framework could provide a client invaluable visual and spatial insights to help inform their site selection and conceptual design decisions.

#GIS #Cartography #RemoteSensing #SpatialAnalysis #DataVisualization #LocationIntelligence #Geodatics #RasterAnalysis #VectorDataManagement #TemporalAnalysis #WebMapping #GeospatialDataProcessing
#OpenStreetMap

![](https://github.com/dantrez/dantrez_projects/blob/main/images/Reg%20combo%20OSM%20and%20Ggle.gif?raw=true)
![](https://github.com/dantrez/dantrez_projects/blob/main/images/Reg%20Measurements.gif?raw=true)

# [Project 11: Animated REM or Detrended Digital Elevation Model (DEM) for flood potential]

  Here is a baseline understanding of a random Canadian river topography area of interest and potential flooding zones. 
In order to enhance the analysis, features such as low-lying areas, benches, and terraces are in a colored (darker colors signify elevation height) 3D view using high-resolution (1-meter) data. This allows a visualization of the elevations above the riverbed more accurately, helping identify which areas are likely to remain dry during flood events.

After measuring the centerline of the river and sampling its elevation points, an interpolation of the sampled elevations outwards from the river was subtracted from the original DEM. Sample points were weighted during interpolation such that the influence of one point relative to another declines with distance (inverse weighting) and accurately associates the river elevation with every point in the AOI. 

To visualize the results, a 3D animation was created showing benches, terraces, and hills above the actual river. This animation provides a clear understanding of how these features may impact potential flood events.

![](https://github.com/dantrez/dantrez_projects/blob/main/images/Animation6.gif?raw=true)

# [Project 12: Unsupervised Deep Learning Land Cover Segmentation Analysis ]

  I'm fired up to share the results of my unsupervised image classification project! I used high-resolution aerial NAIP imagery from the USDA as ground truth to classify land cover types in Utah, USA. For this analysis, I tested using a segmentor in an ONNX model called Deepness and used high-res aerial imagery from 2023. This Open Neural Network Exchange has DeepLabV3+ model with tu-semnasnet_100 backend and FocalDice as a loss function. Then I let the machine learn to identify land cover classes all on its own - no supervision required. Our unsupervised model segmented the data into different land cover types. I then used clustering algorithms to group the data into distinct classes like Roads, Buildings, Woodlands, and Water. The model achieved an overall accuracy of 82% compared to ground truth data - not too shabby for zero human supervision! It nailed Buildings with 94% user's accuracy. But it did struggle a bit with Water and Woodlands, only getting 79% producer's accuracy there. This was probably due to confusion with other impervious surfaces. Roads attained a healthy 91%. The key takeaway? Yes, minor supervised edits post-processing is always needed. Unsupervised learning can achieve pretty darn good results without any labeled training data. It's an absolute game-changer, opening the door to automated mapping for areas we've never tackled before. Unsupervised AI can revolutionize how we extract insights from imagery across industries. 

![](https://github.com/dantrez/dantrez_projects/blob/main/images/Unsup_Class.gif?raw=true)

# [Project 13: SAR Interferogram Analysis at Popocatepetl Volcano, Mexico City Region from December 2021 ]

  I am pleased and honored today as I delve into the latest insights from my Synthetic Aperture Radar (SAR) interferometric analysis of the Popocatepetl volcanic region near Mexico City. My focus is on two SAR images captured by Sentinel-1, providing us with a window to understand geological changes and potential hazards in this area.
  
Image Acquisition:

The data I’m analyzing comprises a pair of Single Complex Sentinel-1 Synthetic Aperture Radar (SAR) imagery acquired over the Popocatepetl volcano. These images were taken on December 7th, 2021; one image was captured two weeks later on December 19th.

Data Processing:

My processing chain utilized SNAP software. 
1.	Orbit Correction: I applied precise orbit files ensuring the images were accurately aligned in space.
2.	Coregistration: The pixels within each image set are matched precisely, allowing for exact comparison between corresponding areas of both datasets. 
3.	Interferogram Generation & Coherence Calculation 
By multiplying one SAR with its complex conjugate counterpart from the other date, I generated an interferogram. This mathematical operation allowed me to calculate and visualize the coherency, which is a measure of how similar two signals are in phase.
4.	Topography Phase Subtraction: 
The topographical elevation differences were subtracted out. By doing so I isolated phases due solely from surface displacement rather than terrain variations. 
5.	Noise Reduction (Goldstein Filtering):
I applied the Goldstein filter to minimize noise in my data, ensuring that only meaningful signals are preserved.
6.	Geometric Correction & Overlay: 
The final product was geometrically corrected and then overlaid onto an image within Google Earth for better visualization. 

Interferogram Analysis:

Upon reviewing these processed images: 
1.	Dense Fringes of Deformation - I observe distinct, dense fringes corresponding to areas experiencing deformations or displacements. 
2.	 Coherence Patterns
             High coherency is visible in the white regions indicating stable and less disturbed terrains. Conversely, low coherency—appearing darker on my map——indicates more unstable conditions possibly due to vegetation density which can disrupt radar signals.
         3.Vegetation Influence - The volume scattering from dense vegetation areas contributes significantly towards loss of coherent signal. 
         4.  Stable Areas
              I also notice regions with high coherency, indicating stable and less disturbed terrains. These are potential candidates for further detailed analysis or monitoring.

Next Steps:

My preliminary findings suggest the need to perform displacement calculations using unwrapped phases in order to precisely quantify surface movements over time. This will provide a clearer understanding of ongoing geological processes at Popocatepetl volcano, which is crucial information that can inform risk assessments and emergency response strategies for local communities.

www.linkedin.com/in/dantrezona

#SAR, #RemoteSensing, #GeospatialAnalysis, #ImageProcessing, #Interferometry, #SAR, #LIDAR, #Geodesy, #Satellite, #DataVisualization, #ChangeDetection, #TimeSeriesAnalysis, #EarthObservation, #NaturalResources, #DisasterManagement, #UrbanPlanning

![](https://github.com/dantrez/dantrez_projects/blob/main/images/Popocatepetl%20volcano%20SAR%20FINAL%20-%20Github.gif?raw=true)
