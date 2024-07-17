<div align="center">
  <h1>Indian Election 2024 Prediction</h1>
</div>


<div align="center">
This project uses machine learning, SQL, and Power BI to predict the Indian Election 2024 outcomes, providing data-driven insights and visualizations to enhance understanding and engagement in the electoral process.
</div>



<div align="center">
  <img src="https://github.com/user-attachments/assets/c3d0a4c3-773a-43fa-bbd9-508ac2aa3f1a">
</div>


### Tools Used:

- **Database Management System**: SQL Server
- **SQL Editor**: SQL Server Management Studio (SSMS)
- **Programming Language**: Python 🐍
- **IDE**: Jupyter Notebook 📓
- **Data Manipulation and Analysis**:
  - NumPy 📊
  - pandas 🐼
  - datetime ⏰
- **Data Visualization**:
  - Matplotlib 📊
  - Seaborn 📈
- **Data Preprocessing and Metrics**:
  - StandardScaler ⚖️
  - confusion_matrix 🔄
  - accuracy_score 🎯
  - classification_report 📝
- **Machine Learning Models**:
  - Logistic Regression 🔄
  - Decision Tree Classifier 🌳
  - Random Forest Classifier 🌲
  - K-Neighbors Classifier 👥
  - Gaussian Naive Bayes 🌐
- **Data Visualization**: Power BI 📊


## Dataset Description: 

### Table- Party_Data

#### Variables-

| Column Name           | Data Type | Information Details                        |
|-----------------------|-----------|--------------------------------------------|
| ID                    | Integer   | Unique identifier for each record          |
| State                 | String    | Name of the state                          |
| Constituency          | String    | Name of the constituency                   |
| Candidate             | String    | Name of the candidate                      |
| Party                 | String    | Political party of the candidate           |
| Result                | String    | Election result (e.g., Won, Lost)          |

### Table- Votes_Info

#### Variables-

| Column Name           | Data Type | Information Details                        |
|-----------------------|-----------|--------------------------------------------|
| ID                    | Integer   | Unique identifier for each record          |
| EVM_Votes             | Integer   | Number of votes cast through EVM           |
| Postal_Votes          | Integer   | Number of votes cast through postal ballots|
| Total_Votes           | Integer   | Total number of votes                      |
| Percentage_of_Votes   | Float     | Percentage of total votes received         |


# ------------------------------------------------------------------------------

 <div align="center">
  <h1>Descriptive Analysis with SQL</h1>
</div>


#### 1. Create a view to join both tables to display all results(columns).

**Query:**
```
CREATE VIEW FullElectionResults AS
SELECT p.ID,p.State,p.Constituency,p.Candidate,p.Party,p.Result,v.EVM_Votes,v.Postal_Votes,v.Total_Votes,v.Percentage_of_Votes
FROM Party_Data p
JOIN Votes_Info v ON p.ID = v.ID;

Select * from FullElectionResults;
```

**Result:**
![image](https://github.com/user-attachments/assets/f9db1626-e285-4c9a-8d0d-6d0864dbcc7e)




#### 2. 

**Query:**
```

```

**Result:**





















































# ------------------------------------------------------------------------------

<div align="center">
  <h1>Exploratory Data Analysis with Python</h1>
</div>

This repository contains code and resources for Exploratory Data Analysis (EDA) and machine learning prediction on 2024 Indian Election data. The project demonstrates fundamental data manipulation techniques using SQL for data retrieval and Python programming language. It covers essential operations such as data loading, cleaning, handling missing values, outliers detection, transformation, analysis, and data visualization using Pandas, NumPy, Matplotlib, Seaborn, and Power BI. Machine learning modules are employed for accuracy assessment and predictive modeling.


## Insights from the Dataset:

- After importing the dataset, our first step is to check if the data is imported properly, we can use `election_data.shape` to check the number of observations (rows) and features (columns) in the dataset
- Output will be : ![image](https://github.com/user-attachments/assets/21cbcae8-8da8-465b-9378-c597a0fbdf39)
- which means that the dataset contains 8902 records and 10 variables.
- We will now use `election_data.head(2)` to display the top 2 observations of the dataset
- ![image](https://github.com/user-attachments/assets/747eeaa2-17a0-4dec-b28b-da44c5ad81a4)
- To understand more about the data, including the number of non-null records in each columns, their data types, the memory usage of the dataset, we use `election_data.info()`
- ![image](https://github.com/user-attachments/assets/8729dee2-c227-41bf-979b-4d3f444358e2)
- Count of distinct values in each column with `election_data.nunique()`
- 


## Data Preparation:

Data can have different sorts of quality issues. It is essential that you clean and preperate your data to be analyzed.  Therefore, these issues must be addressed before data can be analyzed.

**Checking the variable names for appropriate naming convention-**
- ![image](https://github.com/user-attachments/assets/5cddd83a-17ec-473f-9ad1-d69d7df5c519)
- A lot of these variables have spaces in inbetween the names which is not an appropriate naming convention
- Using list comprehension for replacing spaces with underscores in column names of the data.
- Renaming column with special character % in the variable name
````
election_data.columns = [col.replace(' ','_') for col in election_data.columns]
election_data = election_data.rename(columns = {'%_of_Votes':'Percentage_of_Votes'})
````
- ![image](https://github.com/user-attachments/assets/1b84af20-02aa-417f-9603-4559c1f32555)

**Datatype Conversion-**
- Converting datatype of ID from int to object (as it is a categorical variable) and votes columns from object to float (as they are numeric).

**Checking Data Duplicacy-**
- Checking if there are any duplicate records in the dataset with `election_data.duplicated().value_counts()`
- There is no dupicacy in our data.

**Replacing - (hyphen) with NULL in integer data values:**
- Treating the values with - (hyphen) as nulls, pass the values as NULL instead of - (hyphen) with replace
````
election_data['EVM_Votes'] = election_data['EVM_Votes'].replace('-',None)
````


## Handling Missing Values:

Next step is to check for missing values in the dataset. It is very common for a dataset to have missing values.
- `election_data.isna().sum()` isna() is used for detecting missing values in the dataframe, paired with sum() will return the number of missing values in each column.
- ![image](https://github.com/user-attachments/assets/1b903cf7-f968-4ab6-85c1-6b9372813691)
- There is around 6.5% of missing data in our dataset. Treating it by filling the missing values with median of respective variable.
````
election_data['EVM_Votes'] = election_data['EVM_Votes'].fillna(election_data['EVM_Votes'].median())
````

## Data PreProcessing:

Separating the Categorical variables and Numerical variables into two different datasets for Data Preparations for easier analysis.

**Categorical Election data**
````
cat = [var for var in election_data.columns if election_data[var].dtype == 'O']
categorical_election_data = election_data[cat]
````
**Numerical Election data**
````
num = [var for var in election_data.columns if election_data[var].dtype != 'O']
numerical_election_data = election_data[num]
````
For merging the datasets back later, there should be a common column between the 2 seperated datasets `numerical_election_data['ID'] = election_data['ID']`


## Outlier Detection:

- To detect outliers in your dataset, you can use statistical methods or visualizations.
- Visualize the distribution of each numerical feature using box plots.
````
for col in numerical_election_data:
    plt.figure(figsize=(4, 3))
    sns.boxplot(numerical_election_data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
````
![image](https://github.com/user-attachments/assets/39d5834f-f350-47ac-ab31-e20efc2fef99) ![image](https://github.com/user-attachments/assets/e15d720f-ecca-4428-8424-710a6cc0c8cc)
- In a box plot, potential outliers are typically represented as individual points that fall outside the whiskers of the plot.
- The whiskers of the box plot extend to the smallest and largest data points within a certain range from the lower and upper quartiles.

**Potential Ouliers**
````
Q1 = numerical_election_data.quantile(0.25)
Q3 = numerical_election_data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

potential_outliers = ((numerical_election_data < lower_bound) | (numerical_election_data > upper_bound)).sum()
````
- ![image](https://github.com/user-attachments/assets/229d7f70-b6be-4041-8684-0fcdad8b15c5)
- In election data, outliers are less common because predicting which party or candidate will recieve the most or least votes can be unpredictable due to factors like voter sentiment and local issues.
- Outliers are not treated because they are numerous and appear to be accurate as per manual observation.
- Removing the outliers will negatively impact the overall data analysis.



## One Hot Encoding:

- Transforming categorical variables into numerical format - Result variable
````
election_data_merged_copy = pd.get_dummies(election_data_merged_copy,columns = ['Result'], dtype=int)
````
- ![image](https://github.com/user-attachments/assets/de8590b3-e7b7-4438-a77c-05d7c984b5b9)



## Visualizing Data:

### Univariate analysis:
Univariate analysis helps in understanding the distribution and characteristics of a single variable, which helps in pattern recognition. The chosen visualization method depends on the nature of the data — bar charts for discrete (numerical) data, histograms for continuous (numerical) data, and pie charts for categorical data.

**Distribution of number of Candidates per State**
````
plt.figure(figsize=(10, 6))
election_data_merged_copy['State'].value_counts(ascending=True).plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Number of Candidates per State')
plt.xlabel('Number of Candidates')
plt.ylabel('State')
plt.show()
````
![image](https://github.com/user-attachments/assets/a722d484-69af-4346-bce3-aa7a813fa2c6)

**Top 5 Parties by total votes**
````
plt.figure(figsize=(12, 3))
election_data_merged_copy.groupby('Party')['Total_Votes'].sum().nlargest(5).sort_values().plot(kind='barh', color='lightcoral', edgecolor='black')
plt.title('Top 5 Parties by Total Votes Across India')
plt.xlabel('Total Votes')
plt.ylabel('Party')
plt.grid(axis='x')
plt.show()
````
![image](https://github.com/user-attachments/assets/d6ad1b21-c764-479c-bf0f-a942b778c975)

**Distribution of Percentage of Votes**
````
sns.histplot(election_data_merged_copy['Percentage_of_Votes'], bins=15, kde=True, color='lightsteelblue')
````
![image](https://github.com/user-attachments/assets/e86b65bd-6f0d-44d0-9c69-e2c2f5059a7e)


### Bivariate analysis:

Bivariate analysis examines relationships between pairs of variables using scatter plots, line charts for trends, box plots for distribution, and heatmaps for correlations. These visualizations are crucial for uncovering connections and dependencies in the dataset. 

**Distribution of Number of Wins and Losses by State**
````
plt.figure(figsize=(12, 8))
election_data_merged.groupby(['State', 'Result']).size().unstack().plot(kind='barh', stacked=True, figsize=(12, 8))
plt.title('Number of Wins and Losses by State')
plt.xlabel('Count')
plt.ylabel('State')
plt.xticks(rotation=0)
plt.legend(title='Result')
plt.show()
````
![image](https://github.com/user-attachments/assets/d30fe4b8-2ed0-41c8-b899-e0dc42eefc65)

**Scatter Plot of Percentage votes against Total votes with Result Correlation**
````
sns.scatterplot(x=election_data_merged.Percentage_of_Votes,y=election_data_merged.Total_Votes, hue=election_data_merged['Result'])
plt.xlabel('Percentage of Votes')
plt.ylabel('Total Votes')
plt.title('Percentage votes against Total votes')
plt.show()
````
![image](https://github.com/user-attachments/assets/275b9739-c9f3-482d-9db7-997e30c60620)


### Multivariate analysis:

A heatmap visually represents data through color intensity, illustrating the magnitude of values across a matrix. This technique highlights patterns, correlations, and clusters within the data, making it easy to identify trends and outliers.

**Correlation Heatmap of Voting Data**
````
heatmap_data = election_data_merged_copy[['EVM_Votes', 'Postal_Votes', 'Total_Votes', 'Percentage_of_Votes', 'Result_Won', 'Result_Lost']]
correlation_matrix = heatmap_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
````
![image](https://github.com/user-attachments/assets/6378d57b-42c2-4a2d-955f-f3640a30253e)



# ------------------------------------------------------------------------------

<div align="center">
  <h1>Predictive Analysis with Machine Learning Modules</h1>
</div>

### Feature Engineering: Correlation Analysis:
- Including votes variables in features
`features = election_data_ML[['EVM_Votes', 'Postal_Votes', 'Total_Votes', 'Percentage_of_Votes']]`
- And result in target to compare
`target = election_data_ML[['Result']]`

### Split train-test data
`X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)`

### Standardization
````
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
````



## Logistic Regression:

A method that predicts a yes/no outcome based on input features.
````
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
performance['Logistic Regression'] = accuracy_score(y_test, y_pred_log)
log_matrix = confusion_matrix(y_test, y_pred_log)
````
![image](https://github.com/user-attachments/assets/7b13e4f1-8b5e-4a55-9829-f99ab4919bac) ![image](https://github.com/user-attachments/assets/d1c92e29-8caf-4e3f-a457-e5b676c46c84)


## Decision Tree:

A model that splits the data into branches based on feature values to make predictions.
````
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
performance['Decision Tree'] = accuracy_score(y_test, y_pred_tree)
tree_matrix = confusion_matrix(y_test, y_pred_tree)
````
![image](https://github.com/user-attachments/assets/774fa4b0-f781-4b28-b040-af075c15c294) ![image](https://github.com/user-attachments/assets/efcfcb3a-0cfb-406d-a961-8c50ce4fc3d4)


## Random Forest:

An ensemble method that combines multiple decision trees to improve prediction accuracy and control overfitting.

````
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
performance['Random Forest'] = accuracy_score(y_test, y_pred_rf)
rf_matrix = confusion_matrix(y_test, y_pred_rf)
````
![image](https://github.com/user-attachments/assets/31567ccb-2d01-4bbb-bb90-0ba6507620bb) ![image](https://github.com/user-attachments/assets/e475836c-d333-45a4-9805-610ab1dfa030)


## K-Nearest Neighbour:

Classifies an item based on the most common type among its closest neighbors.
````
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
performance['K-Nearest Neighbors'] = accuracy_score(y_test, y_pred_knn)
knn_matrix = confusion_matrix(y_test, y_pred_knn)
````
![image](https://github.com/user-attachments/assets/b4a24c19-b643-486f-a9eb-5bd5ea032247) ![image](https://github.com/user-attachments/assets/65d6931d-1749-48fb-bd79-68ad6a516643)


## Naive Bayes:

A simple way to classify things using the probabilities of features, assuming they are independent.
````
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
performance['Naive Bayes'] = accuracy_score(y_test, y_pred_nb)
nb_matrix = confusion_matrix(y_test, y_pred_nb)
````
![image](https://github.com/user-attachments/assets/63f3c2cb-ff01-4751-a5bb-d091637e7728) ![image](https://github.com/user-attachments/assets/8e8c149f-6934-4e0a-a014-373d8ed4db18)


## Comparison of all Machine Learning Models

- DataFrame with Model and its accuracy - `performance_df = pd.DataFrame(list(performance.items()), columns=['Model', 'Accuracy'])`
- Line chart to compare all the model accuracies
- ![image](https://github.com/user-attachments/assets/5d3c5967-1252-4625-b4c1-50550cefbf28)
- ![image](https://github.com/user-attachments/assets/06a42ac0-c610-48bd-a301-b932ed98723a)

**Logistic Regression, Decision Tree, and Random Forest:**
- Logistic Regression, Random Forest, and Decision Tree models perform very similarly with high accuracy scores around 97.7% to 97.9%.
- These models are well-suited for classification task, indicating robust performance in predicting election outcomes based on the given features and data.

## Conclusion

- Based on our analysis of the election data, we noticed clear voting trends across different areas, particularly regarding EVM and postal votes.
- This analysis highlighted that total votes play a crucial role in determining election outcomes.
- Among the models evaluated, the Logistic Regression and Random Forest Model demonstrated superior predictive accuracy in forecasting election results based on available data.



# ------------------------------------------------------------------------------

<div align="center">
  <h1>Data Visualization with Power BI</h1>
</div>


upload power bi dashboard and its description



































