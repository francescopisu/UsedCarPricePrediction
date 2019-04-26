# Regression Analysis for Used Car Price Prediction
[Francesco Pisu](http://github.com/francescopisu), Hicham Lafhouli

- [Regression Analysis for Used Car Price Prediction](#regression-analysis-for-used-car-price-prediction)
  * [1.1 Introduction](#11-introduction)
  * [1.2 Tools](#12-tools)
  * [1.3 Used car price prediction problem](#13-used-car-price-prediction-problem)
- [2. Notebook structure](#2-notebook-structure)
- [3. Methodology](#3-methodology)
  * [3.1 ELK Stack Analysis](#31-elk-stack-analysis)
    - [Logstash](#logstash)
    - [Elasticsearch](#elasticsearch)
    - [Kibana](#kibana)
  * [3.2 Regression Analysis](#32-regression-analysis)
    - [Data Analysis](#data-analysis)
    - [3.2.1 Data preprocessing](#321-data-preprocessing)
      - [Removing Outliers](#removing-outliers)
      - [Managing categorical attributes](#managing-categorical-attributes)
- [4. Comparing regression models](#4-comparing-regression-models)
  * [4.1 Parallelizing Hyperparameter Tuning with SparkSklearn](#41-parallelizing-hyperparameter-tuning-with-spark-sklearn)
  * [4.2 Linear Regression](#42-linear-regression)
  * [4.3 Decision Tree Regression](#43-decision-tree-regression)
  * [4.4 Random Forest Regression](#44-random-forest-regression)
- [5. Experimental Results](#5-experimental-results)
- [6. Conclusions](#6-conclusion)
- [References](#references)

<!-- FIRST CHAPTER -->
## 1.1 Introduction
This project aims to solve the problem of predicting the price of a used car, using Sklearn's supervised machine learning techniques integrated with Spark-Sklearn library. It is clearly a regression problem and predictions are carried out on dataset of used car sales in the american car market. Several regression tecniques have been studied, including Linear Regression, Decision Trees and Random forests of decision trees. Their performances were compared in order to determine which one works best with out dataset.

This project has been developed for the [Big Data](http://people.unica.it/diegoreforgiato/didattica/insegnamenti/?mu=Guide/PaginaADErogata.do;jsessionid=3BAC62552963C7EB2AD77FCA1D703ACF.jvm1?ad_er_id=2018*N0*N0*S1*29056*20383&ANNO_ACCADEMICO=2018&mostra_percorsi=S&step=1&jsid=3BAC62552963C7EB2AD77FCA1D703ACF.jvm1&nsc=ffffffff0909189545525d5f4f58455e445a4a42378b) course at [University of Cagliari](http://corsi.unica.it/informatica/), supervised by professor [Diego Reforgiato](http://people.unica.it/diegoreforgiato/).

## 1.2 Tools
Most of the project has been developed using Python as the programming language of choice and the following libraries:
- [Scikit-Learn](https://scikit-learn.org/stable/), regression models and cross validation techniques.
- [Spark-Sklearn](https://github.com/databricks/spark-sklearn), parallelization of the hyperparameter tuning process.
- [Pandas](https://pandas.pydata.org), data analysis purposes.
- [ELK Stack](https://www.elastic.co/elk-stack), data analysis too.
- [Rfpimp](https://github.com/parrt/random-forest-importances), feature importances in random forests.

Last but not least, [Google Colaboratory](https://colab.research.google.com/) was the computing platform of choice for training and testing our models.

## 1.3 Used car price prediction problem
Used car price prediction problem has a certain value because different studies show that the market of used cars is destined to a continuous growth in the short term. In fact, leasing cars is now a common practice through which it is possible to get get hold of a car by paying a fixed sum for an agreed number of months rather than buying it in its entirety. Once leasing period is over, it is possible to buy the car by paying the residual value, i.e. at the **expected** resale price. It is therefore in the interest of vendors to be able to predict this value with a certain degree of accuracy, since if this value is initially underestimated, the installment will be higher for the customer which will most likely opt for another dealership. It is therefore clear that the price prediction of used cars has a high commercial value, especially in developed countries where the economy of *leasing* has a certain volume.  
This problem, however, is not easy to solve as the car's value depends on many factor including year of registration, manufacturer, model, mileage, horsepower, origin and several other specific informations such as type of fuel and braking sysrem, condition of bodywork and interiors, interior materials (plastics of leather), safery index, type of change (manual, assisted, automatic, semi-automatic), number of doors, number of previous owners, if it was previously owned by a private individual or by a company and the prestige of the manufacturer.  
Unfortunately, only a small part of this information is available and therefore it is very important to relate the results obtained in terms of accuracy to the features available for the analysis. Moreover, not all the previously listed features have the same importance, some are more so than others and therefore is essential to identify the most important ones, on which to perform the analysis.  
Since some attributes of the dataset aren't relevant to our analysis, they have been discarded; so, as mentioned above, this fact must be taken into account when conclusions on the accuracy are drawn.

<!-- FIRST CHAPTER -->

<!-- SECOND CHAPTER -->
# 2 Notebook structure
The python notebook is structured as follows:
<details><summary>Notebook structure</summary>
<p>
 
```
Used_car_price_prediction:
│   installing libraries 
│   imports
│   read csv file 
│
└─── Price attribute analysis
│     relationship with numerical features
│     relationship with categorical features
│     feature importance related to price
│     correlation matrices (price and general)
│     scatterplots
|
└─── Preprocessing
│     outliers management
│         │     bivariate analysis
│         │     removing outliers by model
│         │     read new csv file (cleaned_cars)
│         │     final outliers removal
│         --------------------------------------
│     towards normal distribution of prices
│     label encoding
│     train/test split
|     
└─── Training and evaluating models
│     function definitions
│     linear regressor
│     decision tree regressor
│         |     complexity curve
│         ----------------------
│     random forest regressor
│         |     complexity curve
│         ----------------------
|
└─── Cross validation
│     linear regression
│     decision tree regression
│     random forest regression
|
└─── Predictions on final model and conclusions
|     feature importance
```

</p>
</details>
<!-- SECOND CHAPTER -->

<!-- THIRD CHAPTER -->
# 3 Methodology
This chapter provides an in-depth description of the followed methodology for solving the problem discussed above, with particular emphasis on the first phase concerning the dataset analysis carried out with ELK stack and the consequent preprocessing of data, followed by the definition, training and evaluation of the chosen regression models, highlighting the importance of integrating these techniques with Spark to parallelize the process of hyperparameter tuning of decision models.  
The [dataset](https://www.kaggle.com/jpayne/852k-used-car-listings) on which the regression analysis was performed consists of approximately 1.2 milion records of used car sales in the american market from 1997 to 2018, acquired via scraping on [TrueCar.com](http://truecar.com) car sales portal.  
Each record has the following features: Id, price, year, mileage, city, state, license plate, manufacturer and model.

![Dataset sample](https://raw.githubusercontent.com/francescopisu/Used-car-price-prediction/master/images/data_sample.png)
 
## 3.1 ELK Stack Analysis
ELK Stack is a set of Open Source products designed by Elastic for analizing and visualizing data of any format through intuitive charts. The stack is composed by three tools: **Elasticsearch**, **Logstash** e **Kibana**.  

A more detailed discussion about this analysis is available at the following [link](...).

### Logstash
The main purpose of Logstash is to manipulate and adapt data of various format coming from different sources, so that the output data is compatible with the chosen destination.  
One of the first operations we needed to do was to adapt our dataset in order to make it compatible with Elasticsearch. To make this possible, we created a [*logstash.conf*](link_to_logstash.conf] configuration file which told Logstash about general structure of the datset, typology of data and filters that had to be applied to each row. Lastly, Elasticsearch as destination has been specified.

### Elasticsearch
This is the engine that allowed us to extract the relevant information from the dataset and to understand how the various features were related to each other.  
Below is an example of a query which groups similar models and for each of them extracts the average, mininum and maximum price.

<details><summary>Query example</summary>
 <p>
  
 ```
     aggs: {
           Model: {
             aggs: {
               info: {
                 stats: {field: "Price"}
               }
             }
             terms: {field: "Model.keyword", size: 10000}
           }
         }
       }
 ```

 </p>
</details> 

### Kibana
Kibana has been used to create a dashboard in order to visualize the data output of Elasticsearch queries.

![Gaussian distribution of Price by Car Manufacturers](https://raw.githubusercontent.com/francescopisu/Used-car-price-prediction/master/images/gaussian.JPG)

## 3.2 Regression Analysis
Formally, a regression analysis consists of a series of statistical processes aimed at estimating the relationships existing between a set of variables; in particular we try to estimate the relationship between a special variable called dependent (in our case the price) and the remaining independent variables (the other features). This analysis makes it possible to understand how the value of the dependent variable changes as the value of any of the independent variables changes, keeping the others fixed.  
To carry out the prediction, various techniques have been studied including linear regression, decision trees and decision tree forests.

### Data Analysis
Before preprocessing the data we must take a look to how the dataset shows up. In particular we carry out an analysis on the price attribute: describing it allows us to appreciate some informations such as min and max values and standard deviation.  We then proceed to compute skewness and kurtosis of the distribution. Next, we observe its relationship with numerical and categorical features by plotting some graphs.  
Feature importance computation showed us that Year and Mileage are both important for Price attribute and through correlation matrices we can learn a bit more about that.

### 3.2.1 Data Preprocessing
The dataset used to carry out the analysis is one of the best available in terms of cleanliness. Despite this, it was necessary to perform preprocessing in order to minimize the probability of incorrect learning by the models.  
First it was ascertained that none of the attributes of the dataset presented null values; surprisingly, no feature presented null values, so no action was required to do so. Subsequently, the plausibility of the values for each of the numerical attributes (Price, Imm. Year, Mileage) present in the dataset was verified. We observed some cars with an extremely high mileage and we applied a filter on mileage, taking all cars with a mileage between 5000 and 250000. Moreover, we decided to take only the cars registered between 2008 and 2017 (extremes included) because the majority of the dataset is refers to that particular year range.  
Lastly, the distribution of Prices has been normalized by applying a log transformation.

#### Removing Outliers
As the Car Manufacturer/Price box plot shows, there is a high presence of outliers in the dataset and the only way to tackle this problem is to apply an outliers removal procedure based on the single car model.  To do so, we take only the values between the 25th and 80th percentile of the gaussian distribution; this procedure is then applied to each of 3k models.

#### Managing categorical attributes
For managing categorical attributes two different approaches has been taken, depending on the particular regression model.  
For linear regression we had to apply a *One Hot Encoding*(OHE) procedure, through which a noolean attribute is added to the dataset for each unique value of the categorical attributes. Clearly, this procedure must be done with caution because the dimensionality of the set ramps up very quickly. 
In our case the only two categorical attributes were Make (car manufacturer) and Model; Model is a categorical variable with more than 3k values and, as mentioned, OHencoding it will generate a dataset with more or less 3000 attributes. That is the technique of choice for linear regressors.  
For decision tree and random forests we simply applied a label encoding procedure, through which an increasing number is associated to each value of a categorical attribute. Label Encoding doesn't fit linear regression well because this type of model will try to find a correlation between those values: for example, if Ford is mapped to 501 and BMW is mapped to 650, linear regressors will automatically assume that BMW in a certain way "is better" than Ford.  
An advantage of Label Encoding is obviously the fact that the dimensionality remains untouched.


<!-- THIRD CHAPTER -->
































