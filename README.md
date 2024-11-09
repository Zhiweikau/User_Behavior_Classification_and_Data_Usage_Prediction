# User Behavior Classification and Data Usage Prediction
---
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Main%20Image.png" alt="Main Image" width="350" height="350">

### Table of Contents



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

### Tools and Technolohy Used
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
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Data%20Display.png" alt="Main Image" width="1000" height="350">

#### 3. Data Cleaning:
- Access with the [**Result**](https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Analysis%20PDF/Data%20Preparation%20-%20Data%20Checking.pdf)
- The dataset contains 700 entries, each representing a unique user with 11 attributes. Key attributes include the User ID (unique identifier), Device Model (5 unique types), Operating System (3 types), App Usage Time (min/day), Screen On Time (hours/day), Battery Drain (mAh/day), Number of Apps Installed, Data Usage (MB/day), Age (42 unique values), Gender (likely binary), and User Behavior Class (5 unique classes). There are no missing values or duplicates, ensuring data completeness and quality. Additionally, the dataset of each attribute has a mix of categorical and numerical data, providing a broad range of information on user behavior and device usage patterns.

---

### EDA
#### 1. Descriptive Statistics:
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/EDA%20-%20Descriptive%20Statistic.png" width="1000" height="350">
- This descriptive statistic for all of the numerical numbers in this dataset exclude the User ID. Its summary for 700 users across several usage and demographic metrics. On average, users spend 271.13 minutes on apps and 5.27 hours with their screen on daily, resulting in an average battery drain of 1525.16 mAh. The typical user has around 50.68 apps installed, with data usage averaging 929.74 MB per day. Users range in age from 18 to 59, with an average of 38 years old. The "User Behavior Class" is ranging from 1 to 5, indicating varying levels of engagement. Notably, there is a wide spread in app usage time, battery drain, and data usage, as reflected in the standard deviations, showing diverse usage patterns within the user base.

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
<img src="https://github.com/Zhiweikau/User_Behavior_Classification_and_Data_Usage_Prediction/blob/main/Images/Classification%2CRegression%20-%20ColumnTransformer.png" width="500" height="200">
