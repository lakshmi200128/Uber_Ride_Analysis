# Databricks notebook source
# Define the path to your CSV file
file_path = "/FileStore/tables/Uber/taxi_zone_lookup.csv"

# Read the CSV file into a DataFrame with inferred schema
taxi_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Register the DataFrame as a temporary view
taxi_df.createOrReplaceTempView("uber_data")

#SQL to query the data in the 'uber_data' view
spark.sql("SELECT * FROM uber_data").show(5)  # Display the first 5 rows of the table

# COMMAND ----------

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Data Cleaning
# Convert LocationID column to IntegerType (assuming it's already IntegerType)
taxi_df = taxi_df.withColumn("LocationID", F.col("LocationID").cast("int"))

# Remove duplicates
taxi_df_cleaned = taxi_df.dropDuplicates()

# Data Analysis
# Summary Statistics
summary_stats = taxi_df_cleaned.describe().toPandas()
print(summary_stats)

# Visualizations
# Distribution of Boroughs
borough_counts = taxi_df_cleaned.groupBy("Borough").count().toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(x="Borough", y="count", data=borough_counts)
plt.title("Distribution of Taxi Pick-up Locations by Borough")
plt.xlabel("Borough")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Service Zone Analysis
service_zone_counts = taxi_df_cleaned.groupBy("service_zone").count().toPandas()
plt.figure(figsize=(10, 10))
plt.pie(service_zone_counts['count'], labels=service_zone_counts['service_zone'], autopct='%2.1f%%', startangle=140)
plt.title("Service Zone Distribution")
plt.show()


# COMMAND ----------

# Replace null values in a specific column
from pyspark.sql.types import IntegerType
df_cleaned = taxi_df.fillna({'service_zone': 'Unknown'})


df_cleaned = df_cleaned.withColumn("LocationID", df_cleaned["LocationID"].cast(IntegerType()))
df_cleaned = df_cleaned.dropDuplicates()
# Filter rows based on Borough
filtered_df = df_cleaned.filter(df_cleaned['Borough'] == 'Manhattan')



# COMMAND ----------

service_zone_counts = df_cleaned.groupBy('service_zone').count().toPandas()
plt.pie(service_zone_counts['count'], labels=service_zone_counts['service_zone'], autopct='%1.1f%%')
plt.title('Proportion of Service Zone')
plt.show()


# COMMAND ----------

location_id_data = df_cleaned.select('LocationID').toPandas()
plt.hist(location_id_data['LocationID'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Location ID')
plt.ylabel('Frequency')
plt.title('Histogram of Location ID')
plt.show()


# COMMAND ----------

