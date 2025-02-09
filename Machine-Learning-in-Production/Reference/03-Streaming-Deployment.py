# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Streaming Deployment
# MAGIC 
# MAGIC After batch deployment, continuous model inference using a technology like Spark's Structured Streaming represents the second most common deployment option.  This lesson introduces how to perform inference on a stream of incoming data.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Make predictions on streaming data
# MAGIC  - Predict using an **`sklearn`** model on a stream of data
# MAGIC  - Stream predictions into an always up-to-date delta file

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Knowledge of Structured Streams and how to work with Structured Streams is a prerequisite for this lesson.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference on Streaming Data
# MAGIC 
# MAGIC Spark Streaming enables...<br><br>
# MAGIC 
# MAGIC * Scalable and fault-tolerant operations that continuously perform inference on incoming data
# MAGIC * Streaming applications can also incorporate ETL and other Spark features to trigger actions in real time
# MAGIC 
# MAGIC This lesson is meant as an introduction to streaming applications as they pertain to production machine learning jobs.  
# MAGIC 
# MAGIC Streaming poses a number of specific obstacles. These obstacles include:<br><br>
# MAGIC 
# MAGIC * *End-to-end reliability and correctness:* Applications must be resilient to failures of any element of the pipeline caused by network issues, traffic spikes, and/or hardware malfunctions.
# MAGIC * *Handle complex transformations:* Applications receive many data formats that often involve complex business logic.
# MAGIC * *Late and out-of-order data:* Network issues can result in data that arrives late and out of its intended order.
# MAGIC * *Integrate with other systems:* Applications must integrate with the rest of a data infrastructure.

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC Streaming data sources in Spark...<br><br>
# MAGIC 
# MAGIC * Offer the same DataFrames API for interacting with your data
# MAGIC * The crucial difference is that in structured streaming, the DataFrame is unbounded
# MAGIC * In other words, data arrives in an input stream and new records are appended to the input DataFrame
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ETL-Part-3/structured-streamining-model.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC Spark is a good solution for...<br><br>
# MAGIC 
# MAGIC * Batch inference
# MAGIC * Incoming streams of data
# MAGIC 
# MAGIC For low-latency inference, however, Spark may or may not be the best solution depending on the latency demands of your task

# COMMAND ----------

# MAGIC %md ### Connecting to the Stream
# MAGIC 
# MAGIC As data technology matures, the industry has been converging on a set of technologies.  Apache Kafka and cloud-specific managed alternatives like AWS Kinesis and Azure Event Hubs have become the ingestion engine at the heart of many pipelines.  
# MAGIC 
# MAGIC This technology brokers messages between producers, such as an IoT device writing data, and consumers, such as a Spark cluster reading data to perform real time analytics. There can be a many-to-many relationship between producers and consumers and the broker itself is scalable and fault tolerant.
# MAGIC 
# MAGIC We'll simulate a stream using the **`maxFilesPerTrigger`** option.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/>  There are a number of ways to stream data.  One other common design pattern is to stream from an an object store where any new files that appear will be read by the stream.

# COMMAND ----------

airbnb_df = spark.read.parquet("dbfs:/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
display(airbnb_df)

# COMMAND ----------

# MAGIC %md Create a schema for the data stream.  Data streams need a schema defined in advance.

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType, StructType

schema = (StructType()
    .add("host_total_listings_count", DoubleType())
    .add("neighbourhood_cleansed", IntegerType())
    .add("zipcode", IntegerType())
    .add("latitude", DoubleType())
    .add("longitude", DoubleType())
    .add("property_type", IntegerType())
    .add("room_type", IntegerType())
    .add("accommodates", DoubleType())
    .add("bathrooms", DoubleType())
    .add("bedrooms", DoubleType())
    .add("beds", DoubleType())
    .add("bed_type", IntegerType())
    .add("minimum_nights", DoubleType())
    .add("number_of_reviews", DoubleType())
    .add("review_scores_rating", DoubleType())
    .add("review_scores_accuracy", DoubleType())
    .add("review_scores_cleanliness", DoubleType())
    .add("review_scores_checkin", DoubleType())
    .add("review_scores_communication", DoubleType())
    .add("review_scores_location", DoubleType())
    .add("review_scores_value", DoubleType())
    .add("price", DoubleType())
)

# COMMAND ----------

# MAGIC %md Check to make sure the schemas match.

# COMMAND ----------

schema == airbnb_df.schema

# COMMAND ----------

# MAGIC %md Check the number of shuffle partitions.

# COMMAND ----------

spark.conf.get("spark.sql.shuffle.partitions")

# COMMAND ----------

# MAGIC %md Change this to 8.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

# MAGIC %md Create a data stream using **`readStream`** and **`maxFilesPerTrigger`**.

# COMMAND ----------

streaming_data = (spark
                 .readStream
                 .schema(schema)
                 .option("maxFilesPerTrigger", 1)
                 .parquet("dbfs:/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
                 .drop("price"))

# COMMAND ----------

# MAGIC %md ### Apply `sklearn` model to streaming data
# MAGIC 
# MAGIC Using the DataFrame API, Spark allows us to interact with a stream of incoming data in much the same way that we did with a batch of data.  

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run(run_name="Final RF Model") as run: 
    df = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
    X = df.drop(["price"], axis=1)
    y = df["price"]

    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    rf.fit(X, y)

    mlflow.sklearn.log_model(rf, "random-forest-model")

# COMMAND ----------

# MAGIC %md Create a UDF from the model you just trained in **`sklearn`** so that you can apply it in Spark.

# COMMAND ----------

import mlflow.pyfunc

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run.info.run_id}/random-forest-model")

# COMMAND ----------

# MAGIC %md Before working with our stream, we need to establish a stream name so that we can have better control over it.

# COMMAND ----------

my_stream_name = "lesson03_stream"

# COMMAND ----------

# MAGIC %md Next create a utility method that blocks until the stream is actually "ready" for processing.

# COMMAND ----------

import time

def until_stream_is_ready(name, progressions=3):  
    # Get the query identified by "name"
    queries = list(filter(lambda query: query.name == name, spark.streams.active))

    # We need the query to exist, and progress to be >= "progressions"
    while (len(queries) == 0 or len(queries[0].recentProgress) < progressions):
        time.sleep(5) # Give it a couple of seconds
        queries = list(filter(lambda query: query.name == name, spark.streams.active))

    print(f"The stream {name} is active and ready.")

# COMMAND ----------

# MAGIC %md Now we can transform the stream with a prediction and preview it with the **`display()`** command.

# COMMAND ----------

predictions_df = streaming_data.withColumn("prediction", pyfunc_udf(*streaming_data.columns))

display(predictions_df, streamName=my_stream_name)

# COMMAND ----------

until_stream_is_ready(my_stream_name)

# COMMAND ----------

# When you are done previewing the results, stop the stream.
for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md ### Write out Streaming Predictions to Delta
# MAGIC 
# MAGIC You can also write out a streaming dataframe to a Feature Store table as well (will need unique ID).

# COMMAND ----------

checkpoint_location = f"{working_dir}/stream.checkpoint"
write_path = f"{working_dir}/predictions"

(predictions_df
    .writeStream                                           # Write the stream
    .queryName(my_stream_name)                             # Name the query
    .format("delta")                                       # Use the delta format
    .partitionBy("zipcode")                                # Specify a feature to partition on
    .option("checkpointLocation", checkpoint_location)     # Specify where to log metadata
    .option("path", write_path)                            # Specify the output path
    .outputMode("append")                                  # Append new records to the output path
    .start()                                               # Start the operation
)

# COMMAND ----------

until_stream_is_ready(my_stream_name)

# COMMAND ----------

# MAGIC %md Take a look at the underlying file.  
# MAGIC 
# MAGIC Refresh this a few times to note the changes.

# COMMAND ----------

spark.read.format("delta").load(write_path).count()

# COMMAND ----------

# When you are done previewing the results, stop the stream.
for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Review
# MAGIC 
# MAGIC **Question:** What are commonly approached as data streams?  
# MAGIC **Answer:** Apache Kafka and cloud-managed solutions like AWS Kinesis and Azure Event Hubs are common data streams.  Additionally, it's common to monitor a directory for incoming files.  When a new file appears, it is brought into the stream for processing.
# MAGIC 
# MAGIC **Question:** How does Spark ensure exactly-once data delivery and maintain metadata on a stream?  
# MAGIC **Answer:** Checkpoints give Spark this fault tolerance through the ability to maintain state off of the cluster.
# MAGIC 
# MAGIC **Question:** How does the Spark approach to streaming integrate with other Spark features?  
# MAGIC **Answer:** Spark Streaming uses the same DataFrame API, allowing easy integration with other Spark functionality.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I get more information on integrating Streaming and Kafka?  
# MAGIC **A:** Check out the <a href="https://spark.apache.org/docs/latest/structured-streaming-kafka-integration.html" target="_blank">Structured Streaming + Kafka Integration Guide</a>
# MAGIC 
# MAGIC **Q:** What's new in Spark 3.1 with Structured Streaming?  
# MAGIC **A:** Check out the Databricks blog post <a href="https://databricks.com/blog/2021/04/27/whats-new-in-apache-spark-3-1-release-for-structured-streaming.html" target="_blank">What’s New in Apache Spark™ 3.1 Release for Structured Streaming</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
