# ML Analysis on Airbnb Prices in NYC
![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/14248e89-cd03-4d65-9738-4da538f9b558)


**Authors: Maggie Davis, Harper Harrell, Chris Bruce, and Sweta Balaji** 

# Reproduce 
Before you read about our machine-learning analysis, here are some directions on how to use our GitHub Repository to reproduce and further our analysis! To access the Airbnb data, you can download the data on Kaggle at `https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data` OR you can simply download the `Airbnb NYC Dataset.csv` CSV file in our GitHub Repository. Next, you can import the data and perform data cleaning by accessing the `Import Data and Data Cleaning .ipynb` notebook. Lastly, we ran 8 different machine-learning models when performing our analysis on Airbnb Pricing in NYC. The code for each of the models can be accessed in the `Machine Learning Models` folder. The models found in the folder include `Decision Tree Regression.ipynb`, `Forward Selection .ipynb`, `KNN.ipynb`, `Lasso Regression.ipynb`, `PCR.ipynb`, `PLS.ipynb`, `Random Forest.ipynb`, and `Ridge Regression.ipynb`. As a bonus, if you want to explore basic statistics about the Airbnb data, please access the `Pictures of Basic Data Statistics` folder! If you are interested in how much you should charge per night for your Airbnb property, feel free to access our `Predicting the Price of Your Property` notebook! 

## Abstract 
The Airbnb marketplace in New York City has become one of the most popular marketplaces in the world. In order to better understand this dynamic and prestigious playground, we looked at an Airbnb dataset that included information such as id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, and availability_365. Using 8 different machine-learning models, we wanted to see how many and to what extent each variable impacted the **price** of one night in the Big Apple. Using various methodologies, we found that a **random forest regression model** was the most predictive. Using this model, we saw that the size/privacy of the listing was most predictive, followed by the exact location of the listing (i.e. longitude and latitude). Other important variables in predicting the price of an Airbnb included the host's ID (host_id), the number of reviews the Airbnb received each month (reviews_per_month), and how many days the Airbnb was available throughout the year (availability_365). Using our model, you can enter this information about your property to estimate the market price.

## Introduction
Travel into the United States has been on the rise for the past several decades. Following the setback in travel during the pandemic, the number of trips and amount of money spent on domestic and international travel in the country is now projected to surpass pre-pandemic levels in the coming years. Domestic business and leisure trips are projected to increase to over 2.5 billion, while international travel into the US is expected to garner an estimated $185 billion within the year 2026 alone (US Travel). As the amount of domestic and international travel into the United States increases, so has the demand for improved access to accommodation for travelers. 

Airbnb is a service that allows anyone looking to temporarily rent out their property to connect with travelers seeking accommodations. Owners are given the opportunity to earn money passively, while simultaneously offering lodging options for tourists. We believe that the growing travel industry has heightened the need for diverse and improved options for accommodation, which serves as the motivation for this study. We seek to create a model that predicts the market price of property for those interested in renting out their homes via Airbnb. To achieve this, we examine data on Airbnb listings in New York City to investigate which characteristics of Airbnb properties are most predictive of their market price and provide an approximation for how well our model can estimate the potential cost of a prospective listing. New York City’s Airbnb market serves as an interesting point of study because it possesses one of the most diverse housing markets, containing anything from the country’s most expensive luxury homes to private studio apartments. 

Our quest to identify the most predictive variables of Airbnb price using data from New York City allows us to take on the bigger task of helping owners determine which characteristics of their property are most important to consider when pricing their Airbnb rentals. We hope to provide owners with the opportunity to input data on the characteristics of their properties and give them an estimate of their rental price. We seek to satisfy owners with an appropriate price estimate for their property that accounts for its amenities and characteristics while simultaneously giving renters cost-effective options for their accommodation. 

We aim to use various feature selection models to help us determine the most important predictors of Airbnb prices. These models include Ridge and Lasso Regressions, Forward Selection, Decision Tree Regression, and Random Forest Regression. We also used models that are effective at reducing the dimensionality of our complex data such as Partial Least Squares Regression and Principal Component Regression and models generally useful for making predictions, like the KNN Regression. We predict that the feature selection models will be useful in handling the complexity and potential noisiness of our dataset and reduce the number of variables that are least predictive of Airbnb prices. More specifically, we anticipate that the Random Forest model will be useful in deciphering the most predictive variables, since it can handle the mix of numeric and categorical data present in our dataset, potential missing values and noise, and is not as prone to overfitting or error due to bias, since it aggregates the average outputs of several decisions trees. Additionally, the Random Forest model does not assume linearity between variables, whereas other algorithms, such as Lasso Regression, Ridge Regression, and Forward Selection, choose the most important predictors by modifying or assuming a linear relationship between variables. Furthermore, though Random Forest does not explicitly conduct dimensionality reduction, it is capable of selecting a subset of relevant predictors from our dataset, making it suitable to use for datasets that are high in dimensionality. Though the Partial Least Squares and Principal Component Regression algorithms are useful in dimensionality reduction, they are often more suitable in cases where there is primarily a linear relationship between the predictor and target variables, while the Random Forest can address highly complex and nonlinear relationships between variables. Moreover, KNN regression, which is useful in conducting predictions, struggles to perform efficiently in cases where there is a large number of features to parse through, whereas the Random Forest model can decipher the most important variables even in datasets of high dimension.

Though the Random Forest method appears to be one of the best approaches to building our model, there are limitations in executing the algorithm. Firstly, in order to increase the accuracy of our model, we must increase the number of decision trees. However, increasing the number of decision trees within the model may require more computing power and decrease the computing speed. Therefore, for the scope of our project and resources, we may need to be selective about the number of trees included in our model in order to prioritize computing power and speed. Another limitation is that Random Forest combines decision trees independently of each other. Therefore, though building decision trees in a particular order or combination may result in a more accurate prediction model, Random Forest does not account for the order of decision trees combined. Upon further investigation following our project, we found that research studies that have previously solved our research question have identified models more accurate than the Random Forest. However, for the purposes of our project, we have included the models within the scope of our class. Further details about the limitations of our modeling algorithms and more accurate models from previous research are outlined in the Discussions.


## Setup
In the Airbnb dataset, we accessed through Kaggle, the data contained 16 variables that included: id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, and availability_365. The data contained 48,895 observations with the most popular borough being Manhattan, the most popular neighborhood being Williamsburg, and the most popular room type being an entire home or apartment. To access pictures showing the basic statistics of the Airbnb data, please explore the `Pictures of Basic Data Statistics` folder! 

#### Before we ran any models on the data, we performed the steps below in the data-cleaning process (this process is available to replicate in the `Import Data and Data Cleaning .ipynb` notebook): 
1. We first created a [nearby_subway_line_counts] variable. We found a dataset with latitude and longitude for each NYC Subway Station. Then, for each Airbnb listing, we calculated its distance to stations in the same borough. Lastly, for each station within 800m, we tallied each unique line that went to those stations
2. We dropped the columns, ['name', 'host_name', 'last_review', 'Unnamed: 0'], as they served no importance to our study.
3. We created binary columns for the categorical variables that included ['neighbourhood_group', 'neighbourhood', 'room_type']
4. We used multiple imputation to account for the 10,051 missing values in the [reviews_per_month] variable. 
5. We dropped all zero values for the [price] variable.
6. Lastly, we created a [log_price] column by taking the natural log of the [price] column. We did this to predict the [log_price] variable instead of [price] to account for the variability in the prices of Airbnb.

**The final dataset, after the data-cleaning process, contained 242 variables with 48,884 observations.** 

#### In our analysis, we ran 8 different machine-learning models. These included: 
  - Decision Tree Regression
  - Random Forest Regression
  - Lasso Regression
  - Ridge Regression
  - Partial Least Squares (PLS)
  - Principal Component Regression (PCR)
  - KNN Regression
  - Forward Selection Model

**All models we ran using Python 3 on the Jupyter Notebook environment.** 

## Parameters 
#### Below are the parameters used for each model: 
  - #### Decision Tree Regression:
      - Used Kfold cross-validation (n_splits = 5) to select the optimal values for the parameters ['max_depth', 'min_samples_split', 'min_samples_leaf']
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/9fbc06e6-5f59-49a1-b2aa-a3c09dbea7bb)

      - The optimal values were: Best max_depth: 10, Best min_samples_split: 10, and Best min_samples_leaf: 30
  - #### Random Forest Regression:
      - Unable to perform cross-validation due to a lack of memory and compute power
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/8767317f-635d-48a7-bc63-5b70cfa8e3df)

      - The parameters were set to: max_features = 4 and n_estimators = 500
  - #### Lasso Regression:
      - Used Kfold cross-validation (K=5) to select the optimal value for the alpha parameter
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/0dd1b90c-b1c4-4d60-b0b3-161456759433)

      - Optimal Alpha value selected: 0.0005678305304899897
  - #### Ridge Regression:
      - Used Kfold cross-validation (K=5) to select the optimal value for the alpha parameter
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/dbb638b1-8e83-47d9-bb49-65a93b0e24a8)

      - Optimal Alpha value selected: 0.006210413169280639
  - #### Partial Least Squares (PLS):
      - Ranged the number of components from 1 - 25 and selected the number of components that resulted in the lowest test MSE
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/61d03fa2-f771-482f-b609-c8af5ed3c637)

      - Number of components selected: 11 
  - #### Principal Component Regression (PCR):
      - Ranged the number of principal components from 20 - 200 and selected the number of components that resulted in the lowest validation MSE
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/32382aab-686d-4cb6-b8fc-4736757bd23e)

      - Number of principal components selected: 200 
  - #### KNN Regression:
      - Used cross-validation to find the number of nearest neighbors that resulted in the lowest test MSE. The range of nearest neighbors was 4 - 50.
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/29a07343-81e4-4716-933c-8a180e1de91a)

      - Number of nearest neighbors selected: 13 
  - #### Forward Selection Model:
      - Used the highest R-squared as the parameter to select the "best" forward selection model.
      - ![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/b3643b6d-193d-4d81-850c-c100256da9e2)

      - R-squared obtained: 0.521

## Results
### Ranking the Models Based on Lowest to Highest Test MSE: 
1. **Random Forest**: 0.19035830548538796
2. **Decision Tree Regressor**: 0.21295979454894307
3. **Lasso**: 0.22753221424556327
4. **PLS**: 0.227535
5. **Ridge**: 0.22759993189970487
6. **KNN Regression**: 0.2282538983042809
7. **Forward Selection Model**: 0.23675961752170155
8. **PCR**: 0.24525682291028444

### Most Useful Predictors for Random Forest: 
1. **Entire house/apt**: Having an entire house is very important, and can allow homeowners to list for much higher
2. **Private room**: Getting your own room is still very helpful 
3. **Longitude**: Location of listing dictates price heavily
4. **Latitude**: Location of listing dictates price heavily
5. **Host ID**: May indicate how much an individual host tends to charge per location
6. **Listing ID**: Identifies each unique listing in order of post date, so a sort of time-series marker that we don’t get elsewhere
7. **Reviews Per Month**: Could capture the demand for a certain location
8. **Availability**: # of days available throughout the year
9. **Nearby Subway Lines**: Locations within 0.5 miles of subways may be in higher demand

### Least Important Predictors for Random Forest: 
- **Most Neighborhoods and Boroughs**: No neighborhoods except for Midtown and no borough except for Manhattan accounted for more than 1% of importance
- **Host Listings Count**: The amount of listings (which was thought to be an indicator of lister experience and “sharkiness”) turned out to not be too helpful
- **Number of reviews**: Total count does not indicate price, likely because it could mean the place is really nice or really bad

The Random Forest Regressor appeared to be the best model when estimating the most important predictors in our dataset, giving us a mean squared error (MSE) of approximately 0.19. As determined by our Random Forest model, among the most important predictors of Airbnb price were room type (whether the property was an entire house or an apartment), private room (whether or not the property contained a private room for the renter), longitude, and latitude. Our findings indicated that renting a house versus renting an apartment was the most important predictor of Airbnb price. Those who rent a house via Airbnb are more likely to pay more money than for an apartment. This makes intuitive sense as well, given that the size and amenities of a house outweigh those of an apartment-style property. Similarly, the presence of a private room in an apartment may indicate a property’s larger size and access to amenities, which may explain a higher rental price compared to properties without a private room. Additionally, the longitude and latitude of the property may indicate the trends seen in the New York City housing or real estate market. As one travels throughout New York City, there is fluctuation in the rates of crime, infrastructure quality, job markets, etc. that may influence property pricing in certain regions.

![image](https://github.com/maggieedavisss/ML-Analysis-on-Airbnb-Prices-in-NYC/assets/151679687/4d0d08eb-993d-4948-8adb-6d20fedc4507)

Among the least important predictors of Airbnb price, as determined by the Random Forest Regressor, were most neighborhoods and boroughs, the number of host listings, and the number of reviews. Most neighborhoods, besides Midtown, and most boroughs, besides Manhattan, accounted for less than 1% importance in their ability to predict Airbnb prices. Manhattan, being the country’s most expensive housing market, is more likely to contain more expensive Airbnb properties, compared to listings in other New York City boroughs, explaining its importance in the prediction model (Porter). Furthermore, we initially believed that the number of listings of an individual host may indicate trends in how multiple properties of one host are typically priced. However, the variable containing the number of host listings proved to be of lesser importance in predicting Airbnb prices than other variables. Finally, the number of reviews, which indicates how good or bad the rental property is, was not important in predicting Airbnb prices either. 

Decision Tree Regressor and Lasso Regression were among the most useful models, according to their MSE values of approximately 0.213 and 0.227, respectively. Since both models are skilled in conducting feature selection, we understand why they resulted in relatively lower MSE values for the purposes of our study. However, since the Random Forest model proved to be most useful in predicting Airbnb prices, it may indicate certain characteristics of our dataset. Firstly, the success of the Random Forest model may indicate nonlinearity between variables. Since the Random Forest model does not assume any particular linear relationship between variables, as the Lasso regression does, it is capable of handling complex relationships between variables. Additionally, the mix of variable types, including categorical and numeric variables, was easily handled by the Random Forest model. Next, the Random Forest model is less sensitive to outliers and noise and is therefore less prone to overfitting noisy data. Therefore, the difference in the MSE values between the Random Forest model and the Decision Tree model may indicate the prevalence of noise in our dataset, which the Decision Tree may have been more prone to overfitting. 

Among the least predictive models were the Forward Selection model and Principal Component Regression, which contained MSE scores of 0.237 and 0.245, respectively. Since the Forward Selection model is a modification of the linear regression and is influenced by which predictors are linearly correlated with the target variable, the potential nonlinear relationships between our variables may have affected its accuracy. Additionally, the presence of noisy data may have made the algorithm prone to overfitting by continuously adding predictors to the model, even if they were not very relevant in predicting the target variable. Furthermore, these same characteristics may have influenced the accuracy of the Principal Component Regression, which assumes a linear relationship between the predictor and target variables and is often susceptible to overfitting noisy and complex data.

## Discussion 
Our results show that the Random Forest algorithm is superior at creating a model to predict Airbnb prices. Our MSE value of 0.19 is quite low, indicating a decent performance by the Random Forest. However, upon exploration of existing approaches in recent literature, we found that the Random Forest, though effective, may not be the most predictive algorithm. As outlined in the Introduction, computing power and the combination of decision trees served as limitations in our study within the scope of our resources and class objectives, which may explain why other algorithms were better at creating accurate models.

Existing literature that uses machine learning approaches to predict Airbnb prices has also found Random Forest, among other models, to demonstrate strong performance in predicting Airbnb prices. One paper published in the *Institute of Electrical and Electronics Engineers (IEEE)* identified the best machine-learning models to estimate New York City Airbnb prices by comparing R-squared and GCV values across various algorithms. However, this paper found that, compared to the Random Forest model, the XGBoost algorithm performed better at predicting Airbnb price, containing an R-squared value of 61.8%, while the Random Forest model followed closely behind with an R-squared of 61.2% (Zhu, et. al). The XGBoost algorithm operates similarly to Random Forest, whereby it combines the results of various decision trees. XGBoost more specifically combines decision trees in a sequential manner, in which each following decision tree added to the model corrects the errors of the preceding tree. Since Random Forest combines decision trees independently of each other and not sequentially, the XGBoost algorithm may have led to a more accurately predictive model that addressed the nuanced patterns of decision tree combinations. Additionally, the XGBoost model, similar to Lasso and Ridge regressions, contains regularization terms that penalize large coefficients to counter overfitting. Since the XGBoost model essentially combines the method of aggregating decision trees, as seen in Random Forest, and regularization, as seen in Lasso and Ridge regressions, it served as an effective algorithm to build a predictive model of Airbnb prices. Furthermore, its ability to handle complex, noisy data and a mix of variable types made it suitable for this particular problem. 

Since we did not attempt to look into the XGBoost algorithm, our most predictive model appeared to be the Random Forest. However, upon further investigation of the existing approaches to analyze this research question, we found that the XGBoost algorithm may have created a model with even more precise predictions of our data. The success of the XGBoost algorithm in the IEEE publication indicated that predictions of our data could have benefited from further exploration of the nuance in particular combinations of decision trees. By adding decision trees sequentially, the model is able to correct errors of each preceding tree in its predictions, which the Random Forest lacks. Last, a limitation of our study was the absence of cross-validation in the Random Forest model. The computational time of running the algorithm did not align with our deadlines. This is a significant limitation of our study because we observed a large difference between training and test MSE for this model, suggesting that our model may be overfitting. That being said, performing cross-validation to select the optimal values for the parameters 'n_estimators' and 'max_features' could have improved both the training and test MSE for the Random Forest model, potentially identifying different importance levels for certain features. If one has the computing ability to perform cross-validation using our Random Forest model, we would love for you to share your results! 

## Conclusion 
To summarize, this study delves into the dynamic landscape of the travel industry, which has experienced significant change and resurgence since the pandemic. As travel in the United States grows, the demand for diverse accommodations will increase as well. Recognizing this significant trend, our study focuses on Airbnb, a platform that connects travelers with property owners, with the goal of providing the customer their ideal accommodation during travel. 

This study is motivated by the need to understand which factors influence the market price of Airbnb properties, particularly in the diverse housing market of New York City. By developing predictive models, we hope to provide a tool that allows property owners to assess the market price of their property via its characteristics. 

To achieve this we employed several feature selection models such as Lasso Regression, Forward Selection, Decision Tree, Random Forest Regression, Partial Least Squares, and Principal Component Regression. Additionally, we used Ridge and KNN regression. While Random Forest emerged as the most promising approach, we acknowledge its limitations, such as the heavy amount of computing resources it requires and the lack of consideration for the order of the decision trees. Additionally, other researchers have identified more accurate models, XGBoosting, for similar purposes. 

In conclusion, our research aims to contribute to the understanding of Airbnb price dynamics in a complex market, while also aiming to provide a practical tool for property owners to make informed decisions about their own listings. As the travel industry continues to grow and change, our study contributes to the field of optimizing the homestay experience for both travelers and homeowners alike.


## References
https://www.ustravel.org/sites/default/files/2023-06/us_travel-forecast_summer2023.pdf 

https://www.airbnb.com/help/article/2503 

https://www.forbes.com/advisor/mortgages/real-estate/new-york-housing-market/   

https://ieeexplore.ieee.org/abstract/document/9253078 

https://www.nvidia.com/en-us/glossary/data-science/xgboost/

https://albertum.medium.com/l1-l2-regularization-in-xgboost-regression-7b2db08a59e0 

https://builtin.com/data-science/random-forest-algorithm 

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

https://data.ny.gov/widgets/i9wp-a4ja

https://nypost.com/2022/07/14/average-manhattan-rent-breaks-5000-for-the-first-time/
