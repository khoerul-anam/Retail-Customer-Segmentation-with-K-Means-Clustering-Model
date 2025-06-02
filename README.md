# 🛍️ **Retail Customer Segmentation with K-Means Clustering Model** 📊

## 📚 **Project Description**

This project aims to segment customers in the retail industry based on their characteristics and purchasing behavior. By using a **machine learning** model (*unsupervised learning*), specifically **K-Means Clustering**, this project identifies different customer groups. The dataset used includes various features such as transaction count, customer age, account balance, and transaction duration. The main goal is to transform raw data into actionable insights for the management team to improve business strategies.

## 🎯 **Project Objectives**

The main objectives of this project are:

-   Build and train a **clustering** model using the **K-Means** algorithm to group customers into meaningful segments.
-   Determine the optimal number of **clusters** using the **Elbow Method** with `KElbowVisualizer`.
-   Evaluate the quality of the formed **clusters** using the **Silhouette Score**.
-   Visualize the **clustering** results using **Principal Component Analysis (PCA)** for dimensionality reduction to facilitate interpretation.
-   Interpret the characteristics of each identified customer **cluster** and provide corresponding business strategy recommendations for each segment.

## ⚙️ **Methodology and Project Stages**

This project was carried out through several main stages:

1.  **Library Importation:**
    * **Step:** Importing all necessary Python libraries such as `pandas` for data manipulation, `numpy` for numerical computation, `matplotlib` and `seaborn` for visualization, and `scikit-learn` for data preprocessing (`MinMaxScaler`, `LabelEncoder`), *clustering* (`KMeans`), evaluation (`silhouette_score`), and dimensionality reduction (`PCA`). The `yellowbrick` library was used for `KElbowVisualizer` and `joblib` for model saving.
    * **Objective:** To prepare all the tools needed for the entire project workflow, from data analysis to model storage.

2.  **Data Loading and Initial EDA (Exploratory Data Analysis):**
    * **Step:** The dataset was loaded into a `pandas` DataFrame. Initial checks were performed using `df.head()`, `df.info()`, and `df.describe()` to understand the structure, data types, descriptive statistics, and to detect missing values and duplicates.
    * **Visualization:** Data visualization using correlation heatmaps and histograms of numerical features to gain a deeper understanding.
    * **Objective:** To understand the basic characteristics of the dataset as a foundation for the preprocessing stage.

3.  **Data Cleaning and Preprocessing:**
    * **Step:**
        * **Checking for Missing Values and Duplicates:** Using `isnull().sum()` and `duplicated().sum()`.
        * **Handling Missing Values:** Imputing missing values in numerical features using the mean.
        * **Handling Duplicates:** Removing duplicated data rows.
        * **Removing Irrelevant Features:** Identifier columns (`TransactionID`, `AccountID`, etc.) were removed.
        * **Feature Scaling:** Numerical features like `TransactionAmount`, `CustomerAge`, etc., were scaled using `MinMaxScaler`.
        * **Feature Encoding:** Categorical features like `TransactionDate`, `TransactionType`, etc., were converted to numerical representations using `LabelEncoder`.
    * **Objective:** To clean the data and transform it into a format suitable for the K-Means algorithm, ensuring each feature contributes proportionally and the model can process the data correctly.

4.  **Clustering Model Development:**
    * **Step:**
        * **Determining the Optimal Number of Clusters (k):** Using the *Elbow Method* with `KElbowVisualizer` from `yellowbrick`. The optimal number of *clusters* chosen was **3**.
        * **Training K-Means Model:** A `KMeans` model from `sklearn.cluster` was trained with `n_clusters=3`.
        * **Adding Cluster Labels:** The resulting *cluster* labels were added to the main DataFrame.
        * **Evaluating Cluster Quality:** Using *Silhouette Score* (result: 0.5405).
        * **Visualizing Clustering Results:** Using PCA to reduce dimensions and visualize *clusters* in a 2D *scatter plot*.
    * **Objective:** To group customers into distinct segments based on their feature similarities and to evaluate how well the grouping was performed.

5.  **Cluster Interpretation:**
    * **Step:** Analyzing the characteristics of each *cluster* based on descriptive statistics (mean, min, max) of the features for each *cluster*.
    * **Objective:** To translate the numerical results of *clustering* into understandable business *insights* and to assign meaningful personas or labels to each customer segment, along with strategic recommendations.

6.  **Model Saving and Data Export:**
    * **Step:** The trained K-Means model was saved using `joblib`. The final DataFrame containing *cluster* labels was exported to a CSV file.
    * **Objective:** To store the project for the use of classification models. (*Note: This objective might need adjustment if the primary focus is clustering, not classification. For clustering, the goal is more towards storing segmentation results for further analysis or strategy implementation.*)

## ⚙️ **Algorithms Used**

The main algorithms implemented in this project are:

-   **K-Means Clustering**: A partition-based *clustering* algorithm that aims to group data into *k* *clusters* where each data point belongs to the *cluster* with the nearest *mean* (centroid).
-   **Principal Component Analysis (PCA)**: Used for dimensionality reduction to visualize *clusters* in 2D space.

## 🧠 **Key Insights/Findings**

Key insights derived from this project include:

-   Based on the **Elbow Method**, the optimal number of *clusters* identified is **3 clusters**.
-   The **K-Means** model built yielded a **Silhouette Score** of approximately **0.5405**, indicating that the formed *clusters* are reasonably well-defined.
-   Three distinct customer segments were successfully identified:
    -   **Cluster 0 (Mature, Most Active & Easy Access Customers)**: Relatively older, fewest login attempts, moderate transaction and balance amounts, most recent transactions. ***Recommendation***: Maintain engagement with relevant offers, simplify complex transactions, offer products for the mature age segment.
    -   **Cluster 1 (Young, Less Active & Small Transaction Customers)**: Relatively younger, lowest average transaction value, shortest transaction duration, highest login attempts, longest since last transaction. ***Recommendation***: Re-engagement campaigns, improve login/UX process, offer entry-level products.
    -   **Cluster 2 (Established Customers with Highest Transactions & Balances)**: Highest average transaction and account balance values, longest transaction duration, with medium age and recency. ***Recommendation***: Focus on premium services, investment products, exclusive loyalty programs, cross-selling high-value products.
-   This segmentation provides a solid basis for the company to design more personalized and effective marketing strategies.

## 🛠️ **Dependencies**

To run this project locally, you need to install the following Python dependencies:

-   📚 `pandas`: Library for data manipulation and analysis.
-   🔢 `numpy`: Package for scientific computing, especially for arrays and matrices.
-   📊 `matplotlib`: Library for creating static, animated, and interactive visualizations.
-   📈 `seaborn`: Data visualization library based on `matplotlib` that provides a high-level interface for drawing attractive and informative statistical graphics.
-   🤖 `scikit-learn`: Machine learning library that provides various tools for data preprocessing, *clustering*, classification, regression, and model evaluation.
-   🟡 `yellowbrick`: Machine learning visualization library that extends the `scikit-learn` API to assist in model selection and hyperparameter tuning.
-   💾 `joblib`: Library for running Python functions as parallel jobs and for saving/loading Python objects (like models).

