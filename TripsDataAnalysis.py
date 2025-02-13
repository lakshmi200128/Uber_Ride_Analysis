# Databricks notebook source
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # Importing functions module as F
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize PySpark session
spark = SparkSession.builder.appName("ReadParquetFiles").getOrCreate()

# Path to the folder containing Parquet files
folder_path = "/FileStore/tables/parquet"

# Read Parquet files from the folder
parquet_df = spark.read.parquet(folder_path)

# Show schema and some sample rows
parquet_df.printSchema()
parquet_df.show(5, truncate=False)

# Perform further operations on the DataFrame
# Example: Handle missing values by filling with mean values
# cleaned_df = parquet_df.fillna(parquet_df.agg(*(F.mean(col).alias(col) for col in parquet_df.columns)).first())
cleaned_df = parquet_df.dropna()

# Example: Data analysis and visualization
# Plotting histogram of the 'trip_miles' column




# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Assuming 'spark' is your SparkSession
# Read your parquet file directly into a DataFrame
df = spark.read.parquet("/FileStore/tables/parquet")

# Filter out rows with null values in 'trip_miles' column
cleaned_df = df.filter(F.col("trip_miles").isNotNull())

# Use Spark SQL aggregation to compute the histogram
histogram_df = cleaned_df.groupBy(F.floor(F.col("trip_miles") / 5).alias("bin")).count().orderBy("bin")

# Convert the resulting DataFrame to Pandas for plotting
histogram_data = histogram_df.toPandas()

# Plotting the histogram using matplotlib
plt.bar(histogram_data["bin"] * 5, histogram_data["count"], width=5, align="edge")
plt.title("Histogram of Trip Miles")
plt.xlabel("Miles")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# Calculate average trip miles per hour of the day
avg_trip_miles_per_hour = df.groupBy(F.hour("pickup_datetime").alias("hour_of_day")).agg(F.avg("trip_miles").alias("avg_trip_miles"))

# Convert DataFrame to Pandas for plotting
avg_trip_miles_per_hour_pd = avg_trip_miles_per_hour.toPandas()

# Plotting the bar chart using matplotlib
plt.figure(figsize=(10, 6))
plt.bar(avg_trip_miles_per_hour_pd["hour_of_day"], avg_trip_miles_per_hour_pd["avg_trip_miles"])
plt.title("Average Trip Miles per Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Trip Miles")
plt.xticks(range(24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# COMMAND ----------

# Extract date from pickup_datetime and calculate daily total trip miles
daily_trip_miles = df.withColumn("pickup_date", F.to_date("pickup_datetime")).groupBy("pickup_date").agg(F.sum("trip_miles").alias("total_trip_miles")).orderBy("pickup_date")

# Convert DataFrame to Pandas for plotting
daily_trip_miles_pd = daily_trip_miles.toPandas()

# Plotting the line chart using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(daily_trip_miles_pd["pickup_date"], daily_trip_miles_pd["total_trip_miles"], marker="o", linestyle="-")
plt.title("Daily Total Trip Miles Over Time")
plt.xlabel("Date")
plt.ylabel("Total Trip Miles")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# COMMAND ----------

# Filter out rows with null values in trip_miles and trip_time columns
filtered_df = df.filter((F.col("trip_miles").isNotNull()) & (F.col("trip_time").isNotNull()))

# Convert DataFrame to Pandas for plotting
filtered_df_pd = filtered_df.select("trip_miles", "trip_time").toPandas()

# Plotting the scatter plot using matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df_pd["trip_miles"], filtered_df_pd["trip_time"], alpha=0.5)
plt.title("Scatter Plot of Trip Miles vs. Trip Time")
plt.xlabel("Trip Miles")
plt.ylabel("Trip Time (minutes)")
plt.grid()
plt.show()

# COMMAND ----------

# Extract the hour of the day from the pickup_datetime column
df = df.withColumn("hour_of_day", F.hour("pickup_datetime"))

# Calculate the number of rides requested per hour
rides_per_hour = df.groupBy("hour_of_day").count().orderBy("hour_of_day")

# Convert DataFrame to Pandas for plotting
rides_per_hour_pd = rides_per_hour.toPandas()

# Plotting the bar chart using matplotlib
plt.figure(figsize=(12, 6))
plt.bar(rides_per_hour_pd["hour_of_day"], rides_per_hour_pd["count"], color='skyblue')
plt.title("Number of Rides Requested per Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Rides")
plt.xticks(range(24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# COMMAND ----------

# Extract the month and year from the pickup_datetime column
df = df.withColumn("year_month", F.date_format("pickup_datetime", "yyyy-MM"))

# Calculate the average trip time (in minutes) per month
avg_trip_time_per_month = df.groupBy("year_month").agg(F.avg("trip_time").alias("avg_trip_time")).orderBy("year_month")

# Convert DataFrame to Pandas for plotting
avg_trip_time_per_month_pd = avg_trip_time_per_month.toPandas()

# Plotting the line graph using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(avg_trip_time_per_month_pd["year_month"], avg_trip_time_per_month_pd["avg_trip_time"], marker='o', color='green')
plt.title("Average Trip Time Over Months")
plt.xlabel("Month")
plt.ylabel("Average Trip Time (minutes)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# COMMAND ----------

pickup_counts = df.groupBy("PULocationID").count().alias("pickup_count")
top_pickup_locations = pickup_counts.orderBy(F.desc("count"))
# Convert to Pandas DataFrame
top_pickup_locations_pd = top_pickup_locations.toPandas()

# Plotting the top pickup locations
plt.figure(figsize=(12, 8))
top_pickup_locations_pd.head(20).plot(kind='bar', x='PULocationID', y='count', legend=None)
plt.title("Top 20 High-Demand Pickup Locations")
plt.xlabel("Pickup Location ID")
plt.ylabel("Frequency of Pickups")
plt.xticks(rotation=45, ha='right')
plt.show()

# COMMAND ----------

# Assuming 'df' is your DataFrame containing ride data
revenue_df = df.select("pickup_datetime", "base_passenger_fare", "tolls", "sales_tax", "tips")
revenue_df = revenue_df.withColumn("total_revenue", F.col("base_passenger_fare") + F.col("tolls") + F.col("sales_tax") + F.col("tips"))
revenue_df = revenue_df.withColumn("pickup_day", F.to_date("pickup_datetime"))
daily_revenue = revenue_df.groupBy("pickup_day").agg(F.sum("total_revenue").alias("total_revenue"))
# Convert to Pandas DataFrame
daily_revenue_pd = daily_revenue.toPandas()

# Sort DataFrame by pickup_day
daily_revenue_pd = daily_revenue_pd.sort_values("pickup_day")

# Plotting the total revenue by day
plt.figure(figsize=(12, 8))
plt.plot(daily_revenue_pd["pickup_day"], daily_revenue_pd["total_revenue"], marker='o', linestyle='-', color='b')
plt.title("Total Revenue by Day")
plt.xlabel("Date")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# COMMAND ----------

# Assuming 'df' is your DataFrame containing ride data
revenue_df = df.select("pickup_datetime", "base_passenger_fare", "tolls", "sales_tax", "tips")
revenue_df = revenue_df.withColumn("total_revenue", 
                                   F.col("base_passenger_fare") + F.col("tolls") + F.col("sales_tax") + F.col("tips"))

# Calculate net revenue by subtracting tolls and sales tax from total revenue
revenue_df = revenue_df.withColumn("net_revenue", 
                                    F.col("base_passenger_fare") + F.col("tips") - F.col("tolls") - F.col("sales_tax"))
revenue_df = revenue_df.withColumn("pickup_day", F.to_date("pickup_datetime"))
daily_net_revenue = revenue_df.groupBy("pickup_day").agg(F.sum("net_revenue").alias("total_net_revenue"))
# Convert to Pandas DataFrame
daily_net_revenue_pd = daily_net_revenue.toPandas()

# Sort DataFrame by pickup_day
daily_net_revenue_pd = daily_net_revenue_pd.sort_values("pickup_day")

# Plotting the net revenue by day
plt.figure(figsize=(12, 8))
plt.plot(daily_net_revenue_pd["pickup_day"], daily_net_revenue_pd["total_net_revenue"], marker='o', linestyle='-', color='g')
plt.title("Net Revenue by Day")
plt.xlabel("Date")
plt.ylabel("Net Revenue")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# COMMAND ----------

