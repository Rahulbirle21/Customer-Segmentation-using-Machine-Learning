# CREDIT CARD USER SEGMENTATION USING DIFFERENT CLUSTERING TECHNIQUES.

---

## __Installation Guide__
1. Clone or Fork the project
2. Create a Virtual Environment and write the given command.
```python
pip install -r .\requirements.txt
```
---
## __Tools and Libraries used__
* Python
* Jupyter Notebook
* Pandas
* Numpy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit for Web application

## __Problem Statement__ 
Businesses all over the world are growing every day with a wider access to market and ever increasing customer base. To improve the services and  business outcomes, it becomes necessary for the companies to classify their customers based on different characterists and requirements. By using different Machine learning clustering techniques, we can identify the factors which differentiates one group of customers from the other.


## __Introduction Of The Project__
The project aims to develop a Machine learning clustering model to create different segments of credit card users based on different parameters such as their credit limit, purchases made by them, spending pattern and other financial parameters.

## __Data Collection__
For this project we used the given dataset : 

The dataset has total 8950 rows and 14 columns.

#### Attributes of the dataset :

 1. CUSTOMER_ID : ID to uniquely identify each customer.
 2. Card_type : Type of credit card (Gold,Silver,Titanium,Platinum)
 3. City : City where the users lives.
 4. Gender : Gender of the users.
 5. Credit_limit : Credit_limit assigned to the user.
 6. Balance : Amount of credit used by the user/
 7. Payments : Payment made by the user.
 8. Purchases : Purchase amount spent by the user
 9. Oneoff_purchases : Purchases made on one time payment.
 10. Installment_purchases : Purchases made on installments.
 11. Cash_advance : Cash advance by the user.
 12. Purchase_frequency : How frequently the user makes purchases.
 13. Oneoff_purchase_frequency : How frequently user makes purchases on one time payment.
 14. Purchase_installment_frequency : How frequently user makes purchases on installments.

---

## __Project Summary__
### 1. __Loading the dataset__
```python
df = pd.read_csv('User Data.csv')
```
---

### 2. __Data Processing And Exploratory Data Analysis__
* Identified the outliers using a boxplot
 
  ![Image Link](https://github.com/Rahulbirle21/Images-for-readme/blob/main/boxplot1.png)
  
---

* Outlier treatment using the Capping method -- In Capping method, the outlier values are replaced either with the upper limit or with the lower limit of the Interquartile range (IQR)
```python
# Treating the outliers using the Capping method.

for i in df.columns:
    if df[i].dtype!='object':
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        IQR=q3-q1
    
        lower = q1-1.5*IQR
        upper = q3+1.5*IQR
        df[i]=np.where(df[i]>upper,upper,np.where(df[i]<lower,lower,df[i]))
```
---
* Checked the null values but no null values were found in the dataset.
```python
df.isnull().sum()
```
---
* Countplot of Card type and insights generated from it.

  ![Image Link](https://github.com/Rahulbirle21/Images-for-readme/blob/main/crd%20count.png)
---
* Relationship between Card type and Credit limit using a barplot and insights generated from it.
```python
# Relationship between Card_type and Credit_limit

plt.figure(figsize=(4,4))
sns.barplot(x='Card_type', y ='Credit_limit', data = df)
print('The average Credit limit is', df['Credit_limit'].mean())
print('Number of Users having Credit limit higher than the average limit is ',
      df[df['Credit_limit'] > 4419.30].value_counts().sum())
plt.show()
```

![Image Link](https://github.com/Rahulbirle21/Images-for-readme/blob/main/card%20credit.png)

---
* Relationship between total Purchase amount based on each city using pie chart.
```python
# Relationship between total Purchase amount based on each city using pie chart

explode = (0.05,0.05,0.05,0.05,0.05,0.05)
df.groupby('City').sum().plot(kind='pie',y='Purchases',autopct='%1.0f%%',explode=explode,legend=False)
plt.axis('off')
plt.show()
```

![image](https://github.com/Rahulbirle21/Images-for-readme/blob/main/city%20purchase.png)
---
---

* Relationship between City, card type and purchases made using stacked bar chart
```python
df.groupby(by=['City','Card_type'])['Purchases'].sum().unstack().plot(kind='bar',stacked=True)
```

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/city%20card%20type%20purchases.png)
* __Observations__ :

   1. All the users from Pune holds Silver card (Lower middle income city)
   2. All the users from Kolkata hold Gold Credit card (upper middle income city).
   3. Majority of users in Delhi holds Platinum card, followed by titanium card.
   4. Majority of users from Bengaluru holds Gold credit card, followed by Platinum card.

---
---
* Relationship between Card type, Gender and Installment purchases made using stacked bar chart.

  ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/crdgeninst.png)
  * __Observations__:
    1. Majority of males users makes purchases in intallments.
    2. Silver card holders make highest purchases in installments, followed by Gold card holders.
    3. Female users holding Silver card make more purchases in installments as compared to female users who hold other type of cards.
    4. Titanium card holders make lowest purchases in installments, followed by Platinum card holders (Higher income group).
---
---
* Distribution of Payment amount using Histogram.

 ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/pay%20hist.png)
---
---
* Distribution of Credit limit

  ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/cr%20hist.png)
---
---
* Scatterplot of Installment Purchase and Total purchases

  ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/scatter%20install.png)
---
* Top 10 users based on Purchase frequency
```python
# Top 10 users based on Purchase frequency

plt.figure(figsize=(4,4))
top_purchase = df.nlargest(10,['Purchase_frequency'])
sns.barplot(x = 'Customer_ID',y = 'Purchase_frequency', data = top_purchase)
plt.title('Top 10 User Based On Purchase_frequency')
plt.ylabel('Purchase_frequency')
plt.xticks(rotation = 90)
plt.show()
```

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/top%2010.png)
---
---

### 3. __Machine Learning Model  Building__
* __Ordinal Encoding of Categorical Variables__ : Here, we've used a unique method of ordinal encoding of categorical features based on the 'Credi_limit' variable. For example, a category having higher value of credit limit is given higher order or integer value in the encoding, whereas a category having lower credit limit is given a lower order.
```python
# Ordinal Encoding of categorical features based on the 'Cred_limit' variable

for i in df1.columns:
    if df1[i].dtype == 'object':
        categories = df1.groupby(i)['Credit_limit'].sum().sort_values(ascending=False).index.to_list()
        order = [x for x in range(len(categories),0,-1)]
        df1.replace(categories,order,inplace=True) 
```
---
* __Feature scaling__
```python
# Scaling the features using Standardscaler
sc = StandardScaler()
df1 = pd.DataFrame(sc.fit_transform(df1),columns=df1.columns)
```
---
* __Principal Component Analysis__ : To identify the two most important variables explaining the highest variance in the dataset.
```python
pca = PCA(n_components=2)
pca = pd.DataFrame(pca.fit_transform(df1),columns = ['PC1','PC2'])
pca_variance = PCA(n_components=2).fit(df1)
print('Total variance explained by the two Principal components is :',round(pca_variance.explained_variance_.sum()*10,2), '%')
```
---

### __K-Means Clustering Model__  

   1. Identifying Optimum number of clusters to be formed using Elbow Method and Silhouette score.

      ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/curve.png)

__Observation__ : Based on the elbow method,the optimum number of clusters came out to be in a range of 4 to 8. So we validated each cluster value using Silhouette score and the Silhouette Score for n_clusters = 5 was the maximum with least negative values.

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/kmeans.png)

---

### __Density Based Spatial Clustering Of Application With Noise (DBSCAN Clustering)__

   1. Finding the optimum value of 'epsilon' parameter based on K-distance graph

      ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/download.png)

   __Observation__: The optimum epsilon (eps) value is at the point of maximum curvature in the K distance graph. Here the optimum epsilon value came out to be 0.4. But, as we can observe from the below given scatterplot, the DBSCAN model failed to build effieicnet clusters.

  ![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/dbscan.png)
---
---
### __Agglomerative Hierarchial Clustering__

   1. Identified the optimum number of clusters using a Dendrogram. A dendrogram is a tree like diagram thatrecords the sequence of merge and splits.More the distance of the vertical lines, more will be the distance between the two clusters.Here, we've set the threshold distance = 100, so that the horizontal red line cuts the tallest vertical line. At this threshold, the optimum number of clusters came out to be 4.
![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/dendro.png)

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/hierarchial.png)

---
---
## __Identifying the best performing Model__

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/best.png)

###  __Observation__ : The K-Means clustering model came out to be the best performing model creating the most efficient clusters with highest Silhouette score.

---
## 4. __Training A Random Forest Classification Model to Predict the clusters__

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/random.png)


###  __Observation__ : The accuracy of the classification model came out to be 95% (approx)

---

## 5. __Deployed the Classification Model as a Web application to predict the clusters__

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/st.png)

---

## 6. __Power BI Dashboard

![im](https://github.com/Rahulbirle21/Images-for-readme/blob/main/2024-05-19%20(3).png)

---

## 7. __Project Highlights__

   1. Easy to use and understand
   2. Open Source
   3. High Accuracy
   4. Will help the company to classify the customers for targeted marketing and improve the business outcomes.
