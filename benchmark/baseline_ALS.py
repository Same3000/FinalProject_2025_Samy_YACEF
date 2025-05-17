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
    predictions = predictions.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col("prediction").desc())))
    predictions = predictions.filter(col("rank") <= k)
    predictions = predictions.select(user_col, item_col, "prediction")
    test_df = test_df.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col(rating_col).desc())))
    test_df = test_df.filter(col("rank") <= k)
    test_df = test_df.select(user_col, item_col, rating_col)
    joined_df = predictions.join(test_df, on=[user_col, item_col], how="inner")
    precision = joined_df.count() / predictions.count()
    print(f"Precision at {k}: {precision:.4f}")

def evaluate_als_recall(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):
    predictions = predictions.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col("prediction").desc())))
    predictions = predictions.filter(col("rank") <= k)
    predictions = predictions.select(user_col, item_col, "prediction")
    test_df = test_df.filter(col(rating_col) >= 0.8)
    test_df = test_df.select(user_col, item_col, rating_col)
    joined_df = predictions.join(test_df, on=[user_col, item_col], how="inner")
    recall = joined_df.count() / test_df.count()
    print(f"Recall at {k}: {recall:.4f}")

def evaluate_als_hit_rate(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):

    user_window = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    top_k_preds = predictions.withColumn("rank", row_number().over(user_window)) \
        .filter(col("rank") <= k) \
        .select(user_col, item_col)

    relevant_test_items = test_df.filter(col(rating_col) >= 0.8) \
        .select(user_col, item_col)

    users_with_relevant_items_in_test = relevant_test_items.select(user_col).distinct()
    num_users_with_relevant_items = users_with_relevant_items_in_test.count()

    if num_users_with_relevant_items == 0:
        print(f"Hit Rate at {k}: 0.0000 (No users with relevant items (e.g., {rating_col} >= 0.8) in the test set)")
        return

    hits_df = top_k_preds.join(relevant_test_items, on=[user_col, item_col], how="inner")

    users_with_hits_count = hits_df.select(user_col).distinct().count()

    hit_rate = users_with_hits_count / num_users_with_relevant_items
    print(f"Hit Rate at {k}: {hit_rate:.4f}")


def evaluate_als_mrr(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):
    user_window = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    ranked_predictions = predictions.withColumn("rank", row_number().over(user_window))

    top_k_ranked_predictions = ranked_predictions.filter(col("rank") <= k) \
        .select(user_col, item_col, "rank")

    relevant_test_items = test_df.filter(col(rating_col) >= 0.8) \
        .select(user_col, item_col)

    users_for_mrr_denominator_df = relevant_test_items.select(user_col).distinct()
    num_users_for_mrr_denominator = users_for_mrr_denominator_df.count()

    if num_users_for_mrr_denominator == 0:
        print(f"MRR at {k}: 0.0000 (No users with relevant items (e.g., {rating_col} >= 0.8) in the test set)")
        return

    hits_with_rank_df = top_k_ranked_predictions.join(
        relevant_test_items,
        on=[user_col, item_col],
        how="inner"
    )
    first_hit_rank_per_user_df = hits_with_rank_df.groupBy(user_col) \
        .agg(min("rank").alias("min_rank_of_hit"))
    reciprocal_ranks_df = first_hit_rank_per_user_df \
        .withColumn("reciprocal_rank", 1.0 / col("min_rank_of_hit"))

    sum_of_reciprocal_ranks_result = reciprocal_ranks_df.agg(
        coalesce(sum("reciprocal_rank"), lit(0.0)).alias("total_reciprocal_rank")
    ).first()

    total_sum_of_reciprocal_ranks = 0.0
    if sum_of_reciprocal_ranks_result:
        total_sum_of_reciprocal_ranks = sum_of_reciprocal_ranks_result["total_reciprocal_rank"]

    mrr = total_sum_of_reciprocal_ranks / num_users_for_mrr_denominator
    print(f"MRR at {k}: {mrr:.4f}")

def evaluate_als_ndcg(test_df, predictions, k=10, rating_col="watch_ratio", user_col="userIndex", item_col="videoIndex"):
    window_spec_pred = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    ranked_predictions = predictions.withColumn("rank", row_number().over(window_spec_pred))

    top_k_predictions = ranked_predictions.filter(col("rank") <= k)

    user_item_relevance = top_k_predictions.join(
        test_df.select(user_col, item_col, col(rating_col).alias("relevance")),
        on=[user_col, item_col],
        how="left"
    ).fillna(0, subset=["relevance"])

    user_item_relevance = user_item_relevance.withColumn(
        "dcg_term",
        col("relevance") / expr(f"log2(rank + 1)")
    )
    dcg_per_user = user_item_relevance.groupBy(user_col).agg(
        sum("dcg_term").alias("dcg")
    )

    window_spec_true = Window.partitionBy(user_col).orderBy(col(rating_col).desc())
    ranked_true_items = test_df.withColumn("true_rank", row_number().over(window_spec_true))

    top_k_true_items = ranked_true_items.filter(col("true_rank") <= k)

    top_k_true_items = top_k_true_items.withColumn(
        "idcg_term",
        col(rating_col) / expr(f"log2(true_rank + 1)")
    )
    idcg_per_user = top_k_true_items.groupBy(user_col).agg(
        sum("idcg_term").alias("idcg")
    )

    ndcg_data = dcg_per_user.join(idcg_per_user, on=user_col, how="left").fillna(0, subset=["idcg"])

    ndcg_data = ndcg_data.withColumn(
        "ndcg",
        when(col("idcg") == 0, 0.0).otherwise(col("dcg") / col("idcg"))
    )

    num_users_for_ndcg = ndcg_data.count()
    if num_users_for_ndcg == 0:
        print(f"NDCG at {k}: 0.0000 (No users with recommendations or relevant items to calculate NDCG)")
        return

    avg_ndcg_result = ndcg_data.agg(sum("ndcg")).first()
    if avg_ndcg_result is None or avg_ndcg_result[0] is None:
        avg_ndcg = 0.0
    else:
        avg_ndcg = avg_ndcg_result[0] / num_users_for_ndcg
    
    print(f"NDCG at {k}: {avg_ndcg:.4f}")

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
        print("Generating predictions...")
        predictions = model.transform(test)
        print(f"Predictions shape: {predictions.count()} rows")
        if predictions.count() == 0:
            print("Warning: No valid predictions generated. Check your data.")
            return
        k = 100
        evaluate_regression(predictions)
        evaluate_als_ndcg(test, predictions, k)
        #evaluate_als_precision(test, predictions, k)
        #evaluate_als_hit_rate(test, predictions, k)
        #evaluate_als_mrr(test, predictions, k)
    except Exception as e:
        print(f"Error during model training or evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
