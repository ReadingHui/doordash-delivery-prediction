TEST_FILE = 'datasets/test.csv'
MODEL_FILE = 'model_predictions/lgb_with_bayesCV.txt'
TARGET = 'delivery_time'
FEATURES = ['market_id',
 'store_id',
 'store_primary_category',
 'order_protocol',
 'total_items',
 'subtotal',
 'num_distinct_items',
 'min_item_price',
 'max_item_price',
 'total_onshift_dashers',
 'total_busy_dashers',
 'total_outstanding_orders',
 'estimated_order_place_duration',
 'estimated_store_to_consumer_driving_duration',
 'created_hours']
STORE_TO_MARKET = 'EDA/store_to_market.json'
STORE_TO_PRIMARY = 'EDA/store_to_primary.json'
STORE_PRIMARY_ENCODE = 'model_building/store_primary_category_encode.json'