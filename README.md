# doordash-delivery-prediction
Predicting the delivery duration for a DoorDash order by using LGBM

The project was from StrataScratch at https://platform.stratascratch.com/data-projects/delivery-duration-prediction.

## Project background
The project aims to use different features from a historic dataset of DoorDash deliveries to predict the delivery duration. Features includes Time features, Store features, Order features, Market features and Predictions from other models detailed below:

## Data Description
### 1. Time features
- `market_id`: A city/region in which DoorDash operates, e.g., Los Angeles, given in the data as an id
- `created_at`: Timestamp in UTC when the order was submitted by the consumer to DoorDash. (Note this timestamp is in UTC, but in case you need it, the actual timezone of the region was US/Pacific)
- `actual_delivery_time`: Timestamp in UTC when the order was delivered to the consumer

### 2. Store features
- `store_id`: an id representing the restaurant the order was submitted for
- `store_primary_category`: cuisine category of the restaurant, e.g., italian, asian
- `order_protocol`: a store can receive orders from DoorDash through many modes. This field represents an id denoting the protocol

### 3. Order features
- `total_items`: total number of items in the order
- `subtotal`: total value of the order submitted (in cents)
- `num_distinct_items`: number of distinct items included in the order
- `min_item_price`: price of the item with the least cost in the order (in cents)
- `max_item_price`: price of the item with the highest cost in the order (in cents)

### 4. Market features
- `total_onshift_dashers`: Number of available dashers who are within 10 miles of the store at the time of order creation
- `total_busy_dashers`: Subset of above total_onshift_dashers who are currently working on an order
- `total_outstanding_orders`: Number of orders within 10 miles of this order that are currently being processed.

### 5. Predictions from other models
- `estimated_order_place_duration`: Estimated time for the restaurant to receive the order from DoorDash (in seconds)
- `estimated_store_to_consumer_driving_duration`: Estimated travel time between store and consumer (in seconds)

## Explanatory Data Analysis
There were a total of 197428 entries, where they contains some missing values.
```
RangeIndex: 197428 entries, 0 to 197427
Data columns (total 16 columns):
 #   Column                                        Non-Null Count   Dtype  
---  ------                                        --------------   -----  
 0   market_id                                     196441 non-null  float64
 1   created_at                                    197428 non-null  object 
 2   actual_delivery_time                          197421 non-null  object 
 3   store_id                                      197428 non-null  int64  
 4   store_primary_category                        192668 non-null  object 
 5   order_protocol                                196433 non-null  float64
 6   total_items                                   197428 non-null  int64  
 7   subtotal                                      197428 non-null  int64  
 8   num_distinct_items                            197428 non-null  int64  
 9   min_item_price                                197428 non-null  int64  
 10  max_item_price                                197428 non-null  int64  
 11  total_onshift_dashers                         181166 non-null  float64
 12  total_busy_dashers                            181166 non-null  float64
 13  total_outstanding_orders                      181166 non-null  float64
 14  estimated_order_place_duration                197428 non-null  int64  
 15  estimated_store_to_consumer_driving_duration  196902 non-null  float64
```

A new feature `created_hours` were created to replace the `created_at` feature to just extract the hour in the day, while the target value `delivery_time` was extracted from the time difference between `created_at` and `actual_delivery_time`.

### Features distribution
![image](https://github.com/ReadingHui/doordash-delivery-prediction/assets/146915098/00a238a4-5318-4709-a2a9-ae39ef18b923)

### Target distribution
![image](https://github.com/ReadingHui/doordash-delivery-prediction/assets/146915098/14595904-5c98-4a7e-a850-0f09d3f2ef37)

## Data Cleaning (Missing Values)

### Filling missing values
Various techniques were used to fill the `NaN` values as much as possible.
1. `market_id`  
   984 missing values were filled in by using the corresponding `market_id` of the same `store_id`. There are still 3 data missing.
2. `store_primary_category`  
   154 missing values were filled in by using the corresponding `market_id` of the same `store_id`. There are still 4606 data missing.
3. Market features - number of dasher  
  All the missing values were common across all 3 features. The data were first filled by the median from the closest hour with data of the same `store_id`, then the median from the closest hour with data of the same `market_id`. There were 120 and 12 data being filled respectivly, leaving 16130 data missing afterwards.
4. `estimated_store_to_consumer_driving_duration`  
   Unfortunately most of the data when grouped by `store_id` and `created_hours` only has 1 entry, filling with such data will be too biased.

### Removing data with no `delivery_time`
There are still 7 data with no `actual_delivery_time`, hence no target values can be obtained, the data are removed instead.

## Model exploration
As there are quite a number of missing values and being a mostly numeric tabular dataset, LightGBM were chosen to predict the target.

### 1. Base Model
An LGBRegressor model was being chosen as the base model, the training and validation score were 960 and 1029 respectively, while the test score was 1018. Upon plotting the feature importance, `store_id` was the most important feature, followed by `created_hours`, `estimated_store_to_consumer_driving_duration` and `subtotal`, which is as expected as `store_id` gives the geographic information, `created_hours` relating to traffic condition, `estimated_store_to_consumer_driving_duration` provides time duration of just the delivery part and `subtotal` affects the tips of the Dashers' tips (income).  
![image](https://github.com/ReadingHui/doordash-delivery-prediction/assets/146915098/e5138f6a-c8e0-48ac-acd5-15e0a9370b43)

The error was concentrated, but I found Root Mean Squared Error (RMSE) may not be the most suitable candidate to evaluate the model performance as 900s error in a 10 mins delivery and 60 mins delivery has a very different meaning. Hence a relative error was then chosen to evaluate the models in a more accurate way.  
![image](https://github.com/ReadingHui/doordash-delivery-prediction/assets/146915098/a4679ccf-f724-450e-9df5-cc9db9167a1d)

### 2. Base Model (removing outliers)
From analysing the relative error distribution in the first model, we see the model does not preforms well in short deliveries, which is kind of expected as the fluctuation will be larger for shorter delivery time. I suspect the outliers (unusually long delivery time) may have affected the performance of the model, so I removed the outliers (z-score > 2) and re-trained the model, the scores for training set and validation set improved but the scores for test sets are the same, so it is not very useful.  
![image](https://github.com/ReadingHui/doordash-delivery-prediction/assets/146915098/f856e7f7-3fdd-4733-bed8-179a88edf1df)


### 3. Custom loss function (Relative error loss)
To penalize the larger relative error in the short deliveries, I decided to use the relative rmse loss function to train the model instead. I included the division of `y_true` in the loss function, first one squared, second one didn't to try different degree of panalties. It appears the second penalty was much stronger as the relative error of the training and validation sets drops significantly to ~0.16, which led me to try out more different loss function, by changing the exponent of `y_true` in the denominator. I basically ran a grid search on the parameter, and when the exponent was -0.5 the performance seems to be the best.  
![image](https://github.com/ReadingHui/doordash-delivery-prediction/assets/146915098/f7c2e17e-d6df-47ca-8bf3-e20f977f2e4c)

### 4. Further tackling error at short deliveries
The relative errors at short deliveries still are not satisfactory, as well as the overall performance. Hence I used the method of over-sampling on the short deliveries (`delivery_time` < 2000) to boost up the weight on the short deliveries. A pipeline were built to process the data before feeding to train the model. Surprisingly, the performance of using standard RMSE to train the model after resampling is better than using the relative error score at k = -0.5, both plain train-val-test split and k-fold cv.

## Final Model
The final model chosen was the standard LGBRegressor trained on the resampled data. Average relative error on the test set was 0.09.







