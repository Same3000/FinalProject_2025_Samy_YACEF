import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, isnan, when, lit, expr, explode, row_number, sum, udf, min, coalesce
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def save_model(model, path):
    print(f"Saving model to {path}...")
    try:
        model.save(path)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_data(spark, train_path, test_path):
    print("Loading training dataset...")
    training_df = spark.read.csv(train_path, header=True, inferSchema=True)
    print("Loading test dataset...")
    test_df = spark.read.csv(test_path, header=True, inferSchema=True)
    return training_df, test_df

def clean_data(df):
    df = df.filter(col("watch_ratio").isNotNull())
    df = df.filter(col("watch_ratio") >= 0)
    df = df.filter(~isnan(col("watch_ratio")))
    df = df.withColumn("watch_ratio", when(col("watch_ratio") > 5, 5).otherwise(col("watch_ratio")))
    df = df.withColumn("liked", when(col("watch_ratio") >= 0.90, 1).otherwise(0))
    return df

def index_data(training_df, test_df):
    print("Using existing numerical user_id and video_id columns for ALS...")
    training = training_df.withColumnRenamed("user_id", "userIndex").withColumnRenamed("video_id", "videoIndex")
    test = test_df.withColumnRenamed("user_id", "userIndex").withColumnRenamed("video_id", "videoIndex")
    return training, test

def evaluate_regression(predictions):
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="watch_ratio", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="watch_ratio", predictionCol="prediction")
    mae = mae_evaluator.evaluate(predictions)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")


def evaluate_als_precision(test_df, predictions,  k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):
    # for the predictions and the test_df order by user_id and then watch_ratio then limit to k
    predictions = predictions.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col("prediction").desc())))
    predictions = predictions.filter(col("rank") <= k)
    predictions = predictions.select(user_col, item_col, "prediction")
    test_df = test_df.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col(rating_col).desc())))
    test_df = test_df.filter(col("rank") <= k)
    test_df = test_df.select(user_col, item_col, rating_col)
    # join the predictions and the test_df on user_id and video_id
    joined_df = predictions.join(test_df, on=[user_col, item_col], how="inner")
    # calculate the precision
    precision = joined_df.count() / predictions.count()
    print(f"Precision at {k}: {precision:.4f}")

def evaluate_als_recall(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):
    predictions = predictions.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col("prediction").desc())))
    predictions = predictions.filter(col("rank") <= k)
    predictions = predictions.select(user_col, item_col, "prediction")
    # filter test_df to only get items with rating_col value higher than 0.8
    test_df = test_df.filter(col(rating_col) >= 0.8)
    test_df = test_df.select(user_col, item_col, rating_col)
    joined_df = predictions.join(test_df, on=[user_col, item_col], how="inner")
    # calculate the recall
    recall = joined_df.count() / test_df.count()
    print(f"Recall at {k}: {recall:.4f}")

def evaluate_als_hit_rate(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):

    # Get top-K predictions for each user
    user_window = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    top_k_preds = predictions.withColumn("rank", row_number().over(user_window)) \
        .filter(col("rank") <= k) \
        .select(user_col, item_col)

    # Filter test_df for relevant items (e.g., items with watch_ratio >= 0.8)
    # This defines what a "relevant" item is for the ground truth
    relevant_test_items = test_df.filter(col(rating_col) >= 0.8) \
        .select(user_col, item_col)

    # Identify the set of users who have at least one relevant item in the test set
    # These are the users for whom a "hit" is possible
    users_with_relevant_items_in_test = relevant_test_items.select(user_col).distinct()
    num_users_with_relevant_items = users_with_relevant_items_in_test.count()

    if num_users_with_relevant_items == 0:
        print(f"Hit Rate at {k}: 0.0000 (No users with relevant items (e.g., {rating_col} >= 0.8) in the test set)")
        return

    # Find hits: (user, item) pairs that were recommended in top-K AND are relevant
    hits_df = top_k_preds.join(relevant_test_items, on=[user_col, item_col], how="inner")

    # Count the number of unique users who had at least one hit
    users_with_hits_count = hits_df.select(user_col).distinct().count()

    # Calculate Hit Rate
    hit_rate = users_with_hits_count / num_users_with_relevant_items
    print(f"Hit Rate at {k}: {hit_rate:.4f}")


def evaluate_als_mrr(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):
    # Rank predictions for each user
    user_window = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    ranked_predictions = predictions.withColumn("rank", row_number().over(user_window))

    # Filter for top-K predictions
    top_k_ranked_predictions = ranked_predictions.filter(col("rank") <= k) \
        .select(user_col, item_col, "rank")

    # Filter test_df for relevant items
    relevant_test_items = test_df.filter(col(rating_col) >= 0.8) \
        .select(user_col, item_col)

    # Identify users who have at least one relevant item in the test set
    # These are the users for whom MRR is calculated
    users_for_mrr_denominator_df = relevant_test_items.select(user_col).distinct()
    num_users_for_mrr_denominator = users_for_mrr_denominator_df.count()

    if num_users_for_mrr_denominator == 0:
        print(f"MRR at {k}: 0.0000 (No users with relevant items (e.g., {rating_col} >= 0.8) in the test set)")
        return

    # Join top-K predictions with relevant items to find hits and their ranks
    hits_with_rank_df = top_k_ranked_predictions.join(
        relevant_test_items,
        on=[user_col, item_col],
        how="inner"
    ) # Columns: user_col, item_col, rank
    # For each user, find the minimum rank of a relevant item (rank of the first hit)
    first_hit_rank_per_user_df = hits_with_rank_df.groupBy(user_col) \
        .agg(min("rank").alias("min_rank_of_hit"))
    # Columns: user_col, min_rank_of_hit

    # Calculate reciprocal rank for users who had a hit
    # Users not in this DataFrame had no relevant item in their top-K, their RR is effectively 0
    reciprocal_ranks_df = first_hit_rank_per_user_df \
        .withColumn("reciprocal_rank", 1.0 / col("min_rank_of_hit"))

    # Sum all reciprocal ranks.
    # Users who had relevant items in test_df but no hit in top_k_preds contribute 0 to this sum.
    # Users who had no relevant items in test_df are not in the denominator.
    sum_of_reciprocal_ranks_result = reciprocal_ranks_df.agg(
        coalesce(sum("reciprocal_rank"), lit(0.0)).alias("total_reciprocal_rank")
    ).first()

    total_sum_of_reciprocal_ranks = 0.0
    if sum_of_reciprocal_ranks_result:
        total_sum_of_reciprocal_ranks = sum_of_reciprocal_ranks_result["total_reciprocal_rank"]

    # Calculate MRR
    mrr = total_sum_of_reciprocal_ranks / num_users_for_mrr_denominator
    print(f"MRR at {k}: {mrr:.4f}")

def main():
    spark = SparkSession.builder \
    .appName("KuaiRec_ALS_Model") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

    train_path = "../data_final_project/KuaiRec 2.0/data/big_matrix.csv"
    test_path = "../data_final_project/KuaiRec 2.0/data/small_matrix.csv"
    training_df, test_df = load_data(spark, train_path, test_path)
    training_df = clean_data(training_df)
    test_df = clean_data(test_df)
    print(f"Training dataset shape after cleaning: {training_df.count()} rows")
    print(f"Test dataset shape after cleaning: {test_df.count()} rows")
    training, test = index_data(training_df, test_df)
    print("Configuring ALS model...")
    als = ALS(
        maxIter=15,
        regParam=0.1,
        rank=10,
        userCol="userIndex",
        itemCol="videoIndex",
        ratingCol="watch_ratio",
        coldStartStrategy="drop",
        nonnegative=True
    )
    try:
        model = als.fit(training)
        #model = ALS.load("../saved_models/als_model")
        print("Generating predictions...")
        predictions = model.transform(test)
        print(f"Predictions shape: {predictions.count()} rows")
        predictions.show(5)
        if predictions.count() == 0:
            print("Warning: No valid predictions generated. Check your data.")
            return
        evaluate_regression(predictions)
        evaluate_als_precision(test, predictions, k=20)
        #evaluate_als_recall(test, predictions, k=20)
        evaluate_als_hit_rate(test, predictions, k=20)
        evaluate_als_mrr(test, predictions, k=20)
        #save_model(model, "../saved_models/als_model")
    except Exception as e:
        print(f"Error during model training or evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
