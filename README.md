# User Behavior Classification and Data Usage Prediction 
---
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Title.jpg" alt="Main Image" width="600" height="350">

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Data Source](#data-source)
- [Terminologies Used in Data](#terminologies-used-in-data)
- [Tools and Technology Used](#tools-and-technology-used)
- [Data Preparation](#data-preparation)
  - [1) Import Libraries](#import-libraries)
  - [2) Load Data](#load-data)
  - [3) Data Checking](#data-checking)
  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [1) Descriptive Statistics](#descriptive-statistics)
  - [2) Univariate Analysis](#univariate-analysis)
  - [3) Bivariate Analysis](#bivariate-analysis)
  - [4) Multivariate Analysis](#multivariate-analysis)
  
- [Classification Prediction for User Behavior](#classification-prediction-for-user-behavior)
  - [1) Feature Engineering for Classification](#feature-engineering-for-classification)
  - [2) Model Selection: Random Forest](#model-selection-random-forest)
  - [3) Classification Report](#classification-report)
  - [4) Confusion Matrix](#confusion-matrix)
  - [5) Important Features on Classification](#important-features-on-classification)
  
- [Regression Prediction for Data Usage](#regression-prediction-for-data-usage)
  - [1) Feature Engineering for Regression](#feature-engineering-for-regression)
  - [2) Model Selection: Random Forest](#model-selection-random-forest-1)
  - [3) Performance Metrics](#performance-metrics)
  - [4) Important Features on Regression](#important-features-on-regression)
  - [5) Refining Feature Engineering for Improved Regression Prediction Performance](#refining-feature-engineering-for-improved-regression-prediction-performance)
  - [6) Cross-Validation](#cross-validation)
  - [7) Final Prediction](#final-prediction)
  - [8) Important Features on Final Model](#important-features-on-final-model)
  - [9) Residual Analysis and Outlier Detection](#residual-analysis-and-outlier-detection)
  
- [Summary of Classification and Regression Outcomes](#summary-of-classification-and-regression-outcomes)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)


### Project Overview
This project focuses on analyzing a dataset that captures mobile device usage and user behavior patterns. The primary objective is to leverage data science techniques to gain insights into how users interact with mobile devices and to forecast their data usage accurately. This two approach involves both classification and regression models, executed through Python, to achieve comprehensive predictive insights. 

---

### Objectives
-	Build the classification model, we can categorize users into distinct behavior segments, which could serve as valuable input for understanding customer profiles, tailoring marketing strategies, or enhancing user experience.
-	Build regression model, we can estimate how much data a user is likely to consume, based on historical usage patterns and behavioral factors. Accurate predictions can aid in managing data plans, optimizing network resources, and personalizing data offerings.
-	Provide Actionable Insights for Telecommunications Providers, which offer data-driven recommendations that can aid in decision-making processes, including customer segmentation, data plan optimization, and resource allocation.

---

### Data Source
- [Kaggle](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset)

---

### Terminologies Used in Data
Dataset of user_behavior_dataset csv file:
-	User ID: Unique identifier for each user.
-	Device Model: Model of the device used by the user (e.g., Google Pixel 5, iPhone 12).
-	Operating System: The operating system of the device (e.g., Android and iOS).
-	App Usage Time (min/day): Average time a user spends using apps on their device per day, measured in minutes.
-	Screen On Time (hours/day): The average number of hours the screen is on per day.
-	Battery Drain (mAh/day): The daily battery usage in milliampere-hours.
-	Number of Apps Installed: Total number of apps installed on the user's device.
-	Data Usage (MB/day): Average data consumption per day in megabytes.
-	Age: Age of the user.
-	Gender: Gender of the user.
-	User Behavior Class: A classification code that likely indicates the user behavior pattern or type.

---

### Tools and Technology Used
- <img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Jupyter%20Notebook%20Icon.png" alt="Jupyter Notebook Icon" width="30" height="30"> Python: Data cleaning, transformation, exploratory data analysis (EDA), feature engineering, and model building in this project. Libraries like Pandas and NumPy facilitate efficient data handling, managing tasks such as loading datasets, handling encoding issues and adjusting data types. For EDA, visualization tools like Matplotlib and Seaborn help reveal data patterns, outliers, and relationships, while Pandas Profiling provides comprehensive automated reports. Feature engineering is supported by Scikit-learn with techniques for encoding, scaling, and creating new features to enhance data insights. For classification and regression, Scikit-learn offer a range of models, metrics and cross-validation ensure robust model evaluation.

---

### Data Preparation
#### 1. Import Libraries:
1.	<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Library_Pandas.png" alt="Jupyter Notebook Icon" width="30" height="30"> **Pandas (pd):** Used for data manipulation and analysis, particularly for handling data in DataFrames.
2.	<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Library_Numpy.png" alt="Jupyter Notebook Icon" width="30" height="30"> **NumPy (np):** Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
3.	<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Library_Matplotlib.png" alt="Jupyter Notebook Icon" width="30" height="30"> **Matplotlib (plt):** A plotting library for creating static, animated, and interactive visualizations in Python.
4.	<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Library_Seaborn.png" alt="Jupyter Notebook Icon" width="30" height="30"> **Seaborn (sns):** Built on top of Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics.
5.	<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Library_Warning.jpg" alt="Jupyter Notebook Icon" width="30" height="30"> **Warnings:** Used to control the display of warnings. In this case, set it to ignore certain warnings.
6.	<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Library_SKLearn.png" alt="Jupyter Notebook Icon" width="30" height="30"> **Scikit-Learn (sklearn):**
-	**train_test_split:** Splits data into training and testing sets for model evaluation.
-	**StandardScaler:** Standardizes features by removing the mean and scaling to unit variance.
-	**OneHotEncoder:** Encodes categorical variables as binary (0/1) vectors.
-	**ColumnTransformer:** Used for applying different preprocessing transformations to different columns in a dataset.
-	**Pipeline:** Allows to chain multiple data transformations and model training steps into a single workflow.
-	**classification_report and confusion_matrix:** Metrics for evaluating the performance of classification models.
-	**RandomForestClassifier:** An ensemble classifier that uses a multitude of decision trees for classification tasks.
-	**RandomForestRegressor:** Similar to the classifier but used for regression tasks.
-	**mean_absolute_error, mean_squared_error, r2_score:** Evaluation metrics for regression models.
-	**KMeans:** An algorithm used for clustering tasks.
-	**KFold:** Used for cross-validation, where the data is split into K subsets to evaluate model performance.

#### 2. Load Data:
- Read the [**user_behavior_dataset.csv**](https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Dataset/user_behavior_dataset.csv) and the table shown as following:
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Data%20Display.png" alt="Main Image" width="800" height="300">

#### 3. Data Checking:
- Access with the [**Result**](https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Analysis%20PDF/Data%20Preparation%20-%20Data%20Checking.pdf)
- The dataset contains 700 entries, each representing a unique user with 11 attributes. Key attributes include the User ID (unique identifier), Device Model (5 unique types), Operating System (3 types), App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of Apps Installed, Data Usage (MB/day), Age (42 unique values), Gender (likely binary), and User Behavior Class (5 unique classes). There are no missing values or duplicates, ensuring data completeness and quality. Additionally, the dataset of each attribute has a mix of categorical and numerical data, providing a broad range of information on user behavior and device usage patterns.

---

### EDA
#### 1. Descriptive Statistics:
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/EDA%20-%20Descriptive%20Statistic.png" width="800" height="300">
This descriptive statistic for all of the numerical numbers in this dataset exclude the User ID. Its summary for 700 users across several usage and demographic metrics. On average, users spend 271.13 minutes on apps and 5.27 hours with their screen on daily, resulting in an average battery drain of 1525.16 mAh. The typical user has around 50.68 apps installed, with data usage averaging 929.74 MB per day. Users range in age from 18 to 59, with an average of 38 years old. The "User Behavior Class" is ranging from 1 to 5, indicating varying levels of engagement. Notably, there is a wide spread in app usage time, battery drain, and data usage, as reflected in the standard deviations, showing diverse usage patterns within the user base.

#### 2. Univariate Analysis:
- Pie Chart: Device Model, Operating System, Gender
- Histogram: App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Data Usage (MB/day)
- KDE Plot: Number of Apps Installed
- Box Plot: Age
- Access with the [**Result**](https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Analysis%20PDF/Univariate%20Analysis.pdf)
- The univariate analysis of the dataset reveals a predominantly Android-using sample (79.14%) with a nearly balanced gender distribution (52% male, 48% female) and an age range centered around mid-adulthood (median age of 38 years). Device usage patterns vary significantly, with most users engaging in moderate app usage, screen time, and data usage. However, a small group of high-engagement users stands out, showing elevated app usage (up to 600 minutes per day), longer screen times (up to 12 hours), and higher battery drain (up to 3000 mAh per day). Data usage is similarly right-skewed, with most users consuming less than 500 MB per day, though some reach up to 2500 MB. The number of apps installed follows a multimodal distribution, indicating different user preferences or requirements for app diversity. Overall, this analysis highlights a diverse user base with distinct engagement levels, suggesting potential for segmentation based on usage intensity and user needs.

#### 3. Bivariate Analysis:
- Box Plot: [App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of App Installed, Data Usage (MB/day)] by User Behavior Class
- Scatter plot: [App Usage Time (min/day) vs Screen On Time (hours/day), App Usage Time (min/day) vs Data Usage (MB/day), App Usage Time (min/day) vs Battery Drain (mAh/day), Battery Drain (mAh/day) vs Data Usage (MB/day)] -- Label: User Behavior Class
- Scatter Plot: [Data Usage (MB/day) vs Age] -- Label: Operating System
- Violin Plot: Data Usage (MB/day) by Gender
- Count Plot: User Behavior Class by (Operating System, Gender)
- Swarm Plot: [Data Usage (MB/day), Battery Drain (mAh/day), Screen On Time (hours/day)] vs Gender
- Strip Plot: Age vs User Behavior Class
- Access with the [**Result**](https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Analysis%20PDF/Bivariate%20Analysis.pdf)
- The bivariate analysis reveals distinct categorizations of App Usage Time, Screen On Time, Battery Drain, and Data Usage into specific user behavior classes with positive interrelationships. The observation that Age shows no significant correlation with Data Usage suggests it may not be a valuable predictor, prompting a re-evaluation of the features used in modeling. Additionally, the similarities in Data Usage patterns by Gender and the distribution of User Behavior Classes by Operating System will guide targeted strategies for personalized user recommendations. Additionally, Data Usage, Battery Drain, and Screen On Time exhibit similar patterns between genders and the distribution of Age across User Behavior Classes is quite equivalent, showing that age groups are evenly represented within each class.

#### 4. Multivariate Analysis:
- Pair Plot: App Usage Time (min/day), Screen on Time (hours/day), Battery Drain (mAh/day), Data Usage (MB/day), Age
- Heatmap: Correlation Matrix of Numerical Features
- Access with the [**Result**](https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Analysis%20PDF/Multivariate%20Analysis.pdf)
- The multivariate analysis reveals strong positive relationships among key features such as App Usage Time, Screen On Time, Battery Drain, Number of Apps Installed, Data Usage, and User Behavior Class, with correlation coefficients ranging from 0.9 to 1. This suggests that as one of these features increases, the others tend to increase correspondingly. The only exception is Age, which shows no significant correlation with any of the other features. These insights will be crucial for accurately classifying user behaviors and predicting data usage, as understanding the interdependencies among these variables can enhance model performance and inform targeted interventions or recommendations for users based on their usage patterns.

---

### Classification Prediction of User Behavior
#### 1. Feature Engineering for Classification:
- In the feature engineering process, the ColumnTransformer is utilized to efficiently preprocess different types of data within the dataset. OneHotEncoder is applied to categorical columns—specifically, Device Model, Operating System, and Gender—to convert these categorical variables into a binary matrix, enabling the model to interpret them effectively. The StandardScaler is applied to the numerical columns, including App Usage Time, Screen On Time, Battery Drain, Number of Apps Installed, Data Usage, and Age, standardizing these features by removing the mean and scaling to unit variance. This ensures that all numerical features are on a similar scale, which is important for many machine learning algorithms. By using ColumnTransformer, streamline the preprocessing steps into a single object, allowing for a clean and organized workflow that can easily integrate with subsequent modelling processes.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classification%2CRegression%20-%20ColumnTransformer.png" width="350" height="150">

#### 2. Model Selection: Random Forest
- The Random Forest method will be chosen for classification prediction due to its ability to handle both categorical variables (such as Device Model, Operating System, and Gender) and continuous variables (such as App Usage Time and Data Usage) without the need for extensive preprocessing. This versatility makes Random Forest particularly suitable for managing diverse data types in user behavior classification.
- In summary, while Random Forest can handle various data types without extensive preprocessing, engaging in feature engineering can significantly enhance the model's predictive power and interpretability.
- The pipeline was created and it illustrated in the diagram showcases the structured workflow for preprocessing data and training a Random Forest classifier.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classification%20-%20Pipeline.png" width="350" height="150">

- In this prediction process, 20% of the data is allocated for testing, while the remaining 80% is used for training the model. This allocation is a standard practice that ensures there is sufficient data available for both model training and evaluation.

#### 3. Classification Report:
- The classification report provides a comprehensive overview of the Random Forest model's performance in predicting user behavior classes. Each class achieved perfect scores across the key metrics of accuracy, precision, recall, and F1-score, all recorded at 1.00. This indicates that the model was able to classify every instance accurately without any misclassifications. The support column, which reflects the number of true instances for each class, ranges from 27 to 29, demonstrating that the dataset was well-balanced across all categories. These results highlight the effectiveness of the Random Forest algorithm in handling the classification task.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classifiaction%20Report.png" width="350" height="200">

#### 4. Confusion Matrix:
- The confusion matrix serves as a visual representation of the model's classification performance, further corroborating the findings from the classification report. The diagonal values of the matrix, which are 27, 29, 29, 28, and 27, represent the number of correct predictions for each user behavior class, confirming that all instances were accurately classified. Notably, the off-diagonal cells contain zeros, indicating that there were no false positives or false negatives. This level of accuracy reinforces the model's robustness and suggests that it has successfully learned the underlying patterns in the data.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classification%20-%20Confusion%20Matrix.png" width="500" height="350">

#### 5. Important Features on Classification:
- The top features identified include App Usage Time (min/day), Data Usage (MB/day), and Battery Drain (mAh/day), which are shown to have the highest importance scores. These results suggest that variations in these features significantly influence the model's ability to differentiate between user behavior classes.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classification%20-%20Important%20Features.png" width="700" height="350">
In conclusion, the results from the Random Forest classification model demonstrate outstanding performance, with perfect classification metrics and a clear confusion matrix showing no misclassifications. The analysis of feature importance underscores the critical attributes that influence user behavior predictions. Overall, these findings indicate that the Random Forest model is highly effective for user behavior classification.

---

### Regression Prediction on Data Usage
#### 1. Feature Engineering for Regression:
- In the feature engineering process for regression prediction, ColumnTransformer is utilized to preprocess the dataset effectively. OneHotEncoder is applied to categorical columns same with the Classification prediction method and the StandardScaler is employed to standardize the numerical columns exclude the feature of Data Usage.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classification%2CRegression%20-%20ColumnTransformer.png" width="350" height="150">

#### 2. Model Selection: Random Forest
- By averaging the predictions of multiple decision trees, Random Forest reduces the risk of overfitting, ensuring that the model generalizes well to unseen data. This is crucial when predicting continuous outcomes like Data Usage, where overfitting can lead to poor performance on new data.
- The pipeline created, as illustrated in the accompanying diagram, demonstrates a structured workflow for preprocessing data and training a Random Forest regressor to accurately predict Data Usage.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Pipeline.png" width="350" height="150">

#### 3. Performance Metrics:
- Mean Absolute Error (MAE) is approximately 115.64, indicating that on average of the model's predictions deviate from the actual Data Usage values by this amount. The Mean Squared Error (MSE) is 23170.92, which provides a measure of the average squared difference between predicted and actual values, emphasizing the model's predictive accuracy. The R-squared value of 0.93 signifies that the model explains about 93% of the variance in Data Usage, demonstrating a strong fit and the model's effectiveness in capturing the underlying patterns in the data.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Performance%20Metrics%201.png" width="300" height="80">

#### 4. Important Features on Regression:
- The feature importance chart indicates that "Battery Drain (mAh/day)" and "Number of Apps Installed" are the most influential predictors, followed closely by "App Usage Time (min/day)" and "User Behavior Class." This information highlights that daily battery usage and app engagement are critical factors influencing data consumption, suggesting that users who install more apps and utilize their devices more extensively tend to have higher data usage.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Important%20Metrics%201.png" width="700" height="350">
Overall, the combination of robust feature importance insights and strong performance metrics validates the Random Forest regression approach for predicting Data Usage. This model not only identifies key factors influencing data consumption but also achieves high accuracy, making it a valuable tool for understanding user behavior and optimizing resource management in mobile applications.

#### 5. Refining Feature Engineering for Improved Regression Prediction Performance
- Random Forests are not sensitive to feature scaling because they operate by splitting data based on feature thresholds rather than on distances or magnitudes. Therefore, applying StandardScaler() to the numerical columns is generally unnecessary when using Random Forests.

Add New Features:
- App Usage Time (hours/day) = App Usage Time (min/day) / 60
- App Usage over Screen = App Usage Time (hours/day) / Time On Screen (hours/day)
- Data usage per App hour (MB/hour) = Data Usage (MB/day) / App Usage Time (hours/day)
- Data usage per App = Data Usage (MB/day) / Number of Apps Installed
- Battery Drain per age = Battery Drain (mAh/day) / Age
- Heavy app user = App Usage Time (hours/day) > 4.5hours        ---- (True/False)
- Screen time per app = Screen On Time (hours/day) / Number of Apps Installed
- Age group: (Group 1: Age<20, Group 2: 20<=Age<30, Group 3: 30<=Age<50, Group 4: Others)
- Correlation Matrix:
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Correlation%20Matrix.png" width="800" height="700">

- User segment: The K-Means clustering method was used in this column, it categorizes users into three clusters based on daily App Usage Time (hours/day), Screen On Time (hours/day), and Data Usage (MB/day). Each segment represents a distinct group of users with similar patterns in these behaviors, enabling targeted analysis or personalized services based on these usage characteristics.
- The scatter plot illustrates user segmentation by "Data Usage (MB/day)" and "App Usage Time (hours/day)," showing three distinct user groups. Users with low data usage and app time are clustered in blue (Segment 1), those with moderate usage in green (Segment 2), and high-usage users in red (Segment 0). The data demonstrates a clear trend where increased data usage correlates with higher app usage time, with each segment occupying a specific range of values, which indicating distinct user behavior patterns.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20K-Means%20Clustering.png" width="500" height="350">
Lastly, One Hot Encoder was using for the Categorical Columns, which are Device Model, Operating System and Gender.

#### 6. Cross-Validation:
- K-Fold Cross-Validation is an effective technique for evaluating machine learning models, especially useful for ensuring that a model, such as a Random Forest regressor, generalizes well to unseen data. By splitting the dataset into multiple folds, this method allows the model to be trained and validated on different subsets, providing a more reliable estimate of performance and helping to reduce overfitting. Each data point is utilized for both training and validation, maximizing the use of available data. In the implementation, after loading the dataset and defining features and the target variable, a Random Forest regressor is initialized. K-Fold Cross-Validation is then set up with 5 splits, and the model is evaluated using mean squared error (MSE), mean absolute error (MAE), and R-squared (R²) metrics.
- In conclusion, the K-Fold Cross-Validation results indicate that the Random Forest regression model effectively predicts data usage with high accuracy and reliability. Across five folds, the model demonstrates a strong performance, with the average Mean Absolute Error (MAE) of 32.74 and a Mean Squared Error (MSE) of 1887.98, indicating that the predictions are closely aligned with the actual values. The consistently high R-squared (R²) value of 0.9953 further reinforces the model's ability to explain a significant portion of the variance in data usage, showcasing its robustness and effectiveness in capturing the underlying relationships within the dataset.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20K-Fold%20Cross%20Validation.png" width="280" height="550">

#### 7. Final Prediction:
- The Final Random Forest regression model has been built and evaluated for predicting data usage based on various user behavior metrics. Overall, these results underscore the model's robustness and effectiveness in capturing the complex relationships within the dataset, making it a valuable tool for predicting data usage patterns and informing strategic decisions in user engagement and resource management.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Final%20Prediction.png" width="300" height="80">

#### 8. Important Features on Final Model:
- The most significant feature is Battery Drain (mAh/day), which has the highest importance score, indicating it plays a crucial role in predicting data usage. Following closely are User Behavior Class and User Segment, suggesting that user patterns is also significant predictors. Other important features include Number of Apps Installed, App Usage Time (hours/day), and Data Usage per App, which further emphasize the influence of user engagement metrics.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Important%20Metrics%202.png" width="700" height="350">

#### 9. Residual Analysis and Outlier Detection:
- The residual plot for the Random Forest regression model indicates a random distribution of residuals around the horizontal line at zero, suggesting that the model effectively captures the underlying trends in data usage without systematic bias. However, the analysis identifies four significant outliers, with residual values ranging from -109.99 to -209.39, indicating instances where the model underestimates actual data usage. While the overall performance of the model appears valid and reliable based on the random distribution of residuals, the presence of these outliers warrants further investigation to understand their impact on the model's predictions and data quality. Overall, the model is operational and demonstrates its capacity to provide dependable predictions for the majority of the dataset.
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Regression%20-%20Residual%20Plot%20and%20Outlier.png" width="500" height="550">

---

### Summary of classification and regression outcomes
- The classification report indicates an overall accuracy of 1.00 for the Random Forest model, meaning that the model correctly predicted the user behavior class for every instance in the dataset, achieving a perfect classification rate. This high accuracy reflects the model's exceptional performance in distinguishing between different user behavior classes, further validating the effectiveness of the Random Forest algorithm for this classification task.
- The regression analysis for predicting data usage, conducted using two different feature engineering methods, revealed that the second method significantly improved model performance. The first method yielded a Mean Absolute Error (MAE) of 115.64, a Mean Squared Error (MSE) of 23170.92, and an R-squared (R²) value of 0.9341. In contrast, the second method achieved a much lower MAE of 35.50, an MSE of 2328.62, and a high R² of 0.9934, demonstrating enhanced accuracy and greater consistency in predictions. This substantial improvement underscores the critical role of feature engineering in enhancing model performance, leading to more reliable and effective data usage predictions.

---

### Key Insights
-	App Usage and Data Usage: Analysis reveals a strong positive correlation between app usage time and data usage, indicating that users who spend more time on applications tend to consume more data. This insight emphasizes the need for monitoring app usage patterns to predict data consumption accurately.
-	User Behavior Class Impact: Different user behavior classes, such as heavy users or casual users, demonstrate distinct patterns in data usage. Users classified as heavy users are likely to show significantly higher data consumption, suggesting that targeted strategies can be developed for different segments to optimize data plans and manage usage effectively.
-	Screen On Time Influences Data Usage: The users who frequently interact with their devices may require more robust data management strategies to avoid overages.
-	Battery Drain as an Indicator of Usage Patterns: Features related to battery drain are correlated with both user behavior class and data usage. High battery drain often accompanies heavy app usage, which can help in predicting data consumption patterns more accurately.
-	Background Activity Contribution: Data usage driven by background activities is highlighted as a significant factor affecting overall data consumption. Identifying user behavior classes that engage in high background data activities can provide insights into optimizing app settings to minimize unnecessary data usage.

---

### Recommendations
-	Personalized Data Management Alerts: Implement features that notify users about their data usage patterns, especially for those identified as heavy users. Personalized alerts can help users manage their data consumption more effectively, encouraging them to stay within their data limits.
-	Optimize App Usage Guidelines: Develop and provide users with guidelines or recommendations for optimizing app usage, particularly for those with high screen on time. This can include suggestions to limit background app activity or to set usage limits on specific applications that consume excessive data.
-	User Segmentation for Targeted Offers: Utilize the classification of user behavior classes to tailor data plans and promotional offers. For example, heavy users may benefit from unlimited data plans or special deals on data packages that cater to their usage patterns.
-	Encourage Efficient App Development: Collaborate with app developers to promote efficient data usage and battery performance within applications. This could involve optimizing background processes and reducing the frequency of data syncing when the app is not actively used.

---

### Conclusion
- In conclusion, the insights derived from the classification of user behavior and data usage prediction provide valuable strategies for understanding and optimizing user engagement with their devices. By analyzing usage patterns and interactions, we can effectively categorize users into distinct behavior classes, allowing for tailored recommendations that enhance user experience and data management. 
- The predictive model for data usage further empowers users by forecasting their data consumption based on their behavior, enabling them to make informed decisions about their app usage and data plans. Collectively, these findings not only improve user satisfaction by providing personalized insights but also contribute to more efficient resource utilization, ensuring that users can maximize their device's capabilities while minimizing unnecessary data expenditure. Implementing these recommendations can lead to a more sustainable and user-friendly digital environment, fostering better management of both device resources and user behavior.

---

### MIT License
This project is released under the [MIT License](https://github.com/Zhiweikau/Laptop_Price_Data_Analysis-Python-Tableau/blob/main/LICENSE), permitting you to freely use, modify, and share the codebase, provided that the original license and copyright notice are retained.

### Connect with Me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Zhi%20Wei%20Kau-blue?logo=linkedin)](https://www.linkedin.com/in/zhi-wei-kau-945338243/)
