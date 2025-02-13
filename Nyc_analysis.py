# Databricks notebook source
# Define the path to your CSV file
file_path = "/FileStore/tables/Uber/nyc.csv"

# Read the CSV file into a DataFrame with inferred schema
nyc_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Register the DataFrame as a temporary view
nyc_df.createOrReplaceTempView("uber_data")

#SQL to query the data in the 'uber_data' view
spark.sql("SELECT * FROM uber_data").show(5)  # Display the first 5 rows of the table


# COMMAND ----------

df_filled = nyc_df.fillna(0)

# COMMAND ----------

df_filled = nyc_df.fillna({'temp': 0, 'precip': 0.0, 'snowdepth': 0.0})


# COMMAND ----------

from pyspark.sql import functions as F

# Convert 'datetime' column to DateType
nyc_df = nyc_df.withColumn("datetime", F.to_date("datetime", "yyyy-MM-dd"))

# COMMAND ----------

nyc_df = nyc_df.dropDuplicates()


# COMMAND ----------

# Data Cleaning
# Replace null values in a specific column
from pyspark.sql import functions as F

df_cleaned = nyc_df.fillna({'windgust': 0.0})

# Convert datetime column to DateType
df_cleaned = df_cleaned.withColumn("datetime", F.to_date("datetime", "yyyy-MM-dd"))

# Removing Duplicates
df_cleaned = df_cleaned.dropDuplicates()

# Data Visualization
# Example: Plotting a histogram of temperature
import matplotlib.pyplot as plt

temperature_data = df_cleaned.select('temp').toPandas()
plt.hist(temperature_data['temp'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Histogram of Temperature')
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert datetime column to pandas datetime for plotting
temperature_data = df_cleaned.select('datetime', 'temp').toPandas()
temperature_data['datetime'] = pd.to_datetime(temperature_data['datetime'])

# Plotting temperature over time
plt.figure(figsize=(20, 6))
plt.plot(temperature_data['datetime'], temperature_data['temp'], color='green', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Variation Over Time')
plt.grid(True)
plt.show()


# COMMAND ----------

import seaborn as sns

# Filter data for specific columns
humidity_precip_data = df_cleaned.select('humidity', 'preciptype').toPandas()

# Plotting box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='preciptype', y='humidity', data=humidity_precip_data)
plt.xlabel('Precipitation Type')
plt.ylabel('Humidity')
plt.title('Humidity Distribution by Precipitation Type')
plt.show()


# COMMAND ----------

import seaborn as sns

# Calculate correlation matrix
correlation_matrix = df_cleaned.select('temp', 'feelslike', 'dew', 'humidity', 'precip', 'snowdepth').toPandas().corr()

# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Weather Variables')
plt.show()


# COMMAND ----------

