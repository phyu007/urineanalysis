# %% [markdown]
# # Workshop: **Urine analysis**
#
#
# Contact: phyuphyuthein
#
# Email: e0457806@u.nus.edu.sg
#
#
# [Dataset] The dataset and the reference code are from https://www.kaggle.com/code/kerneler/starter-measurements-of-urine-ph-6e5c2fed-5
#

# %%
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pymysql
import pandas as pd
import json

with open('config.json') as f:
    config = json.load(f)

# RDS configuration details
rds_host = config['host']
name = config['user']
password = config['password']
db_name = config['database']

# Connect to the database
cnx = pymysql.connect(host=rds_host, user=name, password=password, database=db_name)

cursor = cnx.cursor()

# Example query
query = ("SELECT id,PH,temperature,food_name ,date,person_id ,SUM(amount) as amount FROM UrineAnalysis.PH_Temp_Diet t INNER JOIN (SELECT MAX(date) AS latest_date FROM UrineAnalysis.PH_Temp_Diet) latest ON t.date = latest.latest_date group by food_name,PH,temperature ,date,person_id")
cursor.execute(query)

# Fetch the rows
rows = cursor.fetchall()

# Create a pandas DataFrame
data = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])


cursor.close()
cnx.close()
data


# %%

# Load data from your database or CSV file
df3 = data

df3 = df3.drop(['id'], axis=1)

# Pivot the data to create columns for each unique food_name
pivoted_data = df3.pivot_table(
    index=['person_id', 'date', 'PH', 'temperature'], columns='food_name', values='amount')
pivoted_data
# Rename columns to remove the 'food_name' label prefix
pivoted_data.columns.name = None
pivoted_data.columns = pivoted_data.columns.str.replace('food_name_', '')

# # Join the pivoted data with the original data on the 'person_id' and 'date' columns
joined_data = pd.merge(data.drop(columns=['food_name', 'amount', 'id']), pivoted_data, on=[
                       'person_id', 'date', 'temperature', 'PH'])

# # Save the joined data to a new CSV file
#df3 = pivoted_data
# Group by 'person_id' and 'date', and aggregate the other columns
grouped_df = joined_data.groupby(
    ['person_id', 'date', 'temperature', 'PH'], as_index=False)
joined_data.drop_duplicates()
joined_data = joined_data.fillna(0)

# %%

y = joined_data['PH']
X = joined_data.iloc[:, 5:]

X

# %%

y = joined_data['PH']
X = joined_data.iloc[:, 5:]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42)

# %%
# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(mse)

# Check if the predicted urine pH level is within the recommended range
# if urine_ph >= 6.4 and urine_ph <= 6.8:
#     print("The predicted urine pH level is within the recommended range.")
# else:
#     print("The predicted urine pH level is not within the recommended range.")

# %%
#joblib.dump(lr, 'predict_urine_test.joblib')
