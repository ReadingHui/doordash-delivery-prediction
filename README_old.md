# doordash-delivery-prediction
Predicting the delivery duration for a DoorDash order by using LGBM

The project was from StrataScratch at https://platform.stratascratch.com/data-projects/delivery-duration-prediction, dataset can be downloaded in the above link.

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
![image](https://github.com/user-attachments/assets/561e38cf-ee75-4d47-8eb9-4d19f4f99748)

### Target distribution
![image](https://github.com/user-attachments/assets/2852c4fe-aa39-4a4f-b24c-6b89d4d11dd4)


## Data Cleaning (Missing Values)

### Filling missing values
Various techniques were used to fill the `NaN` values as much as possible.
1. `market_id`  
   984 missing values were filled in by using the latest corresponding `market_id` of the same `store_id`. There are still 3 data missing.
2. `store_primary_category`  
   154 missing values were filled in by using the latest corresponding `store_primary_category` of the same `store_id`. There are still 4606 data missing.
3. Market features - number of dasher  
  All the missing values were common across all 3 features. The data were first filled by the median from the closest hour with data of the same `store_id`, then the median from the closest hour with data of the same `market_id`. There were 120 and 12 data being filled respectivly, leaving 16130 data missing afterwards.
4. `estimated_store_to_consumer_driving_duration`  
   Unfortunately most of the data when grouped by `store_id` and `created_hours` only has 1 entry, filling with such data will be too biased.

### Removing data with no `delivery_time`
There are still 7 data with no `actual_delivery_time`, hence no target values can be obtained, the data are removed instead.

## Model exploration
As there are quite a number of missing values and being a mostly numeric tabular dataset, LightGBM were chosen to predict the target.

### 1. Base Model
An LGBMRegressor model was being chosen as the base model, the training and validation score were 903 and 973 respectively, while the test score was 963. Upon plotting the feature importance, `store_id` was the most important feature, followed by `total_outstanding_orders`, `total_onshift_dashers` and `estimated_store_to_consumer_driving_duration`, which is as expected as `store_id` gives the geographic information, `total_outstanding_orders` gives approximated waiting time, `total_onshift_dashers` provides time needed for a dasher to pick up the order and `estimated_store_to_consumer_driving_duration` affects the travelling time from store to customer.  
![image](https://github.com/user-attachments/assets/3c7039b9-5bc3-4a60-8a25-7c09e1fb32fe)

In error analysis, we can see that there are some outliers in the delivery time that contributes to some very high error as shown below, so it is natural to try and remove the outliers before training to see if it would improve the model.
![image](https://github.com/user-attachments/assets/4fead070-8d10-4297-bac5-795d808aa787)


### 2. Base Model (removing outliers)
I removed the outliers (z-score > 2) and re-trained the model, the scores for training set and validation set massively improved to 644 and 694, but the scores for test sets is still 975, which is even worse than keeping the outliers, so it is not very useful.  
![image](https://github.com/user-attachments/assets/b13176d2-9d9d-4417-9762-79e7c49f0571)

### 3. k-fold Bayesian CV for hyperparameter tuning
With the analysis above, we perform 10-fold Bayesian CV on the model to search for the best hyperparameter. The final hyperparameters can be found in the "doordash_delivery_prediction_lgbm" notebook, the RMSE for the training set iss 915 while that of the test set is 956, which is a bit of improvement comparing to the base model.

## Final Model
The final model chosen was the LGBMRegressor with tuned hyperparameter. Average RMSE is 956.







