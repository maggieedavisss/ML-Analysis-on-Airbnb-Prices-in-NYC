# ML Analysis on Airbnb Prices in NYC
**Authors: Maggie Davis, Harper Harrell, Chris Bruce, and Sweta Balaji** 

# Reproduce 
Before you read about our machine-learning analysis, here are some directions on how to use our GitHub Repository to reproduce and further our analysis! To access the Airbnb data, you can download the data on Kaggle at `https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data` OR you can simply download the `Airbnb NYC Dataset.csv` CSV file in our GitHub Repository. Next, you can import the data and perform data cleaning by accessing the `Import Data and Data Cleaning .ipynb` notebook. Lastly, we ran 8 different machine-learning models when performing our analysis on Airbnb Pricing in NYC. The code for each of the models can be accessed in the `Machine Learning Models` folder. The models found in the folder include `Decision Tree Regression.ipynb`, `Forward Selection .ipynb`, `KNN.ipynb`, `Lasso Regression.ipynb`, `PCR.ipynb`, `PLS.ipynb`, `Random Forest.ipynb`, and `Ridge Regression.ipynb`. 

## Abstract 
The Airbnb marketplace in New York City has become one of the most popular marketplaces in the world. In order to better understand this dynamic and prestigious playground, we looked at an Airbnb dataset that included information such as id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, and availability_365. Using 8 different machine-learning models, we wanted to see how many and to what extent each variable impacted the **price** of one night in the Big Apple. Using various methodologies, we found that a **random forest regression model** was the most predictive. Using this model, we saw that the size/privacy of the listing was most predictive, followed by the exact location of the listing (i.e. longitude and latitude). Other important variables in predicting the price of an Airbnb included the host's ID (host_id), the number of reviews the Airbnb received each month (reviews_per_month), and how many days the Airbnb was available throughout the year (availability_365). Using our model, you can enter some information about your home address to estimate the market price.

## **Introduction: Summarize your project report in several paragraphs.**
What is the problem? For example, what are you trying to solve? Describe the motivation.
Why is this problem interesting? Is this problem helping us solve a bigger task in some way? Where would we find use cases for this problem?
What is the approach you propose to tackle the problem? What approaches make sense for this problem? Would they work well or not? Feel free to speculate here based on what we taught in class.
Why is the approach a good approach compared with other competing methods? For example, did you find any reference for solving this problem previously? If there are, how does your approach differ from theirs?
What are the key components of my approach and results? Also, include any specific limitations.

## **Setup: Set up the stage for your experimental results.**
Describe the dataset, including its basic statistics.
Describe the experimental setup, including what models you are going to run, what parameters you plan to use, and what computing environment you will execute on.
Describe the problem setup (e.g., for neural networks, describe the network structure that you are going to use in the experiments).

## **Results: Describe the results from your experiments.**
Main results: Describe the main experimental results you have; this is where you highlight the most interesting findings.
Supplementary results: Describe the parameter choices you have made while running the experiments. This part goes into justifying those choices.

## **Discussion: Discuss the results obtained above.** 
If your results are very good, see if you could compare them with some existing approaches that you could find online. If your results are not as good as you had hoped for, make a good-faith diagnosis about what the problem is.

## **Conclusion: In several sentences, summarize what you have done in this project.**

## **References: Put any links, papers, blog posts, or GitHub repositories that you have borrowed from/found useful here.**
