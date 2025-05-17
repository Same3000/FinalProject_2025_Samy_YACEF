import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, isnan, when, lit, expr, explode, row_number, sum, udf, min, coalesce, array_contains
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import ArrayType, IntegerType, StringType
import ast

def load_data(spark, train_path, test_path, items_df, users_df):
    print("Loading training dataset...")
    training_df = spark.read.csv(train_path, header=True, inferSchema=True)
    print("Loading items dataset...")
    items_df = spark.read.csv(items_df, header=True, inferSchema=True)
    print("Loading users dataset...")
    users_df = spark.read.csv(users_df, header=True, inferSchema=True)
    # Join items and users with training data
    training_df = training_df.join(items_df, on="video_id", how="left")
    training_df = training_df.join(users_df, on="user_id", how="left")
    print("Loading test dataset...")
    test_df = spark.read.csv(test_path, header=True, inferSchema=True)
    return training_df, test_df

def parse_string_to_list(s):
    if s is None:
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None

parse_string_to_list_udf = udf(parse_string_to_list, ArrayType(IntegerType()))

def clean_data(df):
    print("Starting data cleaning...")

    # Original watch_ratio cleaning
    df = df.filter(col("watch_ratio").isNotNull())
    df = df.filter(col("watch_ratio") >= 0)
    df = df.filter(~isnan(col("watch_ratio")))
    df = df.withColumn("capped_watch_ratio", when(col("watch_ratio") > 5, 5).otherwise(col("watch_ratio")))

    interaction_cols_to_drop = []
    if "date" in df.columns:
        interaction_cols_to_drop.append("date")
    if "time" in df.columns:
        interaction_cols_to_drop.append("time")
    if interaction_cols_to_drop:
        df = df.drop(*interaction_cols_to_drop)
        print(f"Dropped interaction columns: {interaction_cols_to_drop}")
    if "feat" in df.columns:
        print("Processing item features...")
        df = df.withColumn("feat_parsed", parse_string_to_list_udf(col("feat")))

        all_categories = list(range(31))
        item_cols_to_drop_categories = [14, 23, 27, 21, 0, 30, 22, 24, 29]
        
        generated_category_cols = []
        for category in sorted(all_categories):
            cat_col_name = f"item_category_{category}"
            df = df.withColumn(cat_col_name, 
                               when(array_contains(col("feat_parsed"), category), 1).otherwise(0))
            generated_category_cols.append(cat_col_name)

        item_cat_cols_to_actually_drop = [f"item_category_{i}" for i in item_cols_to_drop_categories]
        item_cat_cols_to_actually_drop = [c for c in item_cat_cols_to_actually_drop if c in df.columns]

        df = df.drop(*item_cat_cols_to_actually_drop)
        print(f"Dropped item category columns: {item_cat_cols_to_actually_drop}")
        df = df.drop("feat", "feat_parsed")
        print("Dropped 'feat' and 'feat_parsed' columns.")
    else:
        print("Columns are:")
        print(df.columns)
        print("Warning: 'feat' column not found. Skipping item feature preprocessing.")

    if "user_active_degree" in df.columns:
        print("Processing user features...")
        user_cols_to_drop_initial = [
            'onehot_feat5', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17',
            'is_lowactive_period', 'is_live_streamer', 'follow_user_num_range',
            'fans_user_num_range', 'register_days_range', 'friend_user_num_range'
        ]
        actual_user_cols_to_drop_initial = [c for c in user_cols_to_drop_initial if c in df.columns]
        if actual_user_cols_to_drop_initial:
            df = df.drop(*actual_user_cols_to_drop_initial)
            print(f"Dropped user feature columns: {actual_user_cols_to_drop_initial}")

        potential_degree_values = []
        if "user_active_degree" in df.columns:

            active_degree_categories = ["high_active", "middle_active", "low_active", "UNKNOWN"] # Example list
            for cat_val in active_degree_categories:
                if cat_val in df.columns: # If it was already a column somehow
                    print(f"Warning: Column {cat_val} for active degree already exists. Skipping creation.")
                    continue
                df = df.withColumn(cat_val, when(col("user_active_degree") == cat_val, 1).otherwise(0))
            
            df = df.drop("user_active_degree")
            print("Dropped 'user_active_degree' column.")

            user_cols_to_drop_dummies = []
            if "UNKNOWN" in df.columns:
                 user_cols_to_drop_dummies.append("UNKNOWN")
            if "middle_active" in df.columns: 
                 user_cols_to_drop_dummies.append("middle_active")
            
            if user_cols_to_drop_dummies:
                df = df.drop(*user_cols_to_drop_dummies)
                print(f"Dropped user active degree dummy columns: {user_cols_to_drop_dummies}")
    else:
        print("Warning: 'user_active_degree' column not found. Skipping user feature preprocessing related to active degree.")

    if "capped_watch_ratio" not in df.columns:
        print("Error: 'capped_watch_ratio' is not available for engagement score calculation. Setting engagement_score to 0.")
        df = df.withColumn("engagement_score", lit(0.0))
        return df

    engagement_score_expr = col("capped_watch_ratio") * lit(1.0)

    interaction_signals = [
        ("is_like", 1.5),
        ("is_comment", 2.0),
        ("is_share", 2.5)
    ]

    for signal_col_name, weight in interaction_signals:
        if signal_col_name in df.columns:
            df = df.withColumn(signal_col_name, coalesce(col(signal_col_name).cast("integer"), lit(0)))
            engagement_score_expr = engagement_score_expr + (col(signal_col_name) * lit(weight))
        else:
            print(f"Warning: Column '{signal_col_name}' not found. It will not be included in the engagement score.")

    df = df.withColumn("engagement_score", engagement_score_expr)
    print("Calculated 'engagement_score'.")
    print("Data cleaning finished.")
    return df


def index_data(training_df, test_df):
    print("Using existing numerical user_id and video_id columns for ALS...")
    training = training_df.withColumnRenamed("user_id", "userIndex").withColumnRenamed("video_id", "videoIndex")
    test = test_df.withColumnRenamed("user_id", "userIndex").withColumnRenamed("video_id", "videoIndex")
    return training, test

def evaluate_als_precision(test_df, predictions,  k=10, user_col="userIndex", item_col="videoIndex"):
    if "prediction" not in predictions.columns:
        print(f"Error in evaluate_als_precision: 'prediction' col not found in predictions. Aborting metric calculation.")
        return 0.0

    print(f"Calculating Precision@{k}. Relevant items in test set are any (user,item) pairs present.")

    predictions_top_k = predictions.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col("prediction").desc())))
    predictions_top_k = predictions_top_k.filter(col("rank") <= k)
    predictions_top_k = predictions_top_k.select(user_col, item_col)

    relevant_test_items = test_df.select(user_col, item_col).distinct()

    if relevant_test_items.count() == 0:
        print(f"Precision@{k}: 0.0000 (No items in the test set for comparison)")
        return 0.0
        
    hits_df = predictions_top_k.join(relevant_test_items, on=[user_col, item_col], how="inner")
    
    num_hits = hits_df.count()
    num_recommended_in_top_k = predictions_top_k.count()

    precision = num_hits / num_recommended_in_top_k if num_recommended_in_top_k > 0 else 0.0
    print(f"Precision at {k}: {precision:.4f}")
    return precision

def evaluate_als_recall(test_df, predictions, k=10, user_col="userIndex", item_col="videoIndex", rating_col="engagement_score"):
    if "prediction" not in predictions.columns:
        print(f"Error in evaluate_als_recall: 'prediction' col not found in predictions. Aborting metric calculation.")
        return 0.0

    print(f"Calculating Recall@{k}. Relevant items in test set are any (user,item) pairs present.")

    predictions_top_k = predictions.withColumn("rank", row_number().over(Window.partitionBy(user_col).orderBy(col("prediction").desc())))
    predictions_top_k = predictions_top_k.filter(col("rank") <= k)
    predictions_top_k = predictions_top_k.select(user_col, item_col)

    relevant_test_items = test_df.select(user_col, item_col).distinct()

    num_total_relevant_items = relevant_test_items.count()
    if num_total_relevant_items == 0:
        print(f"Recall at {k}: 0.0000 (No items in the test set)")
        return 0.0

    hits_df = predictions_top_k.join(relevant_test_items, on=[user_col, item_col], how="inner")
    num_hits = hits_df.count()

    recall = num_hits / num_total_relevant_items if num_total_relevant_items > 0 else 0.0
    print(f"Recall at {k}: {recall:.4f}")
    return recall

def evaluate_als_hit_rate(test_df, predictions, k=10, user_col="userIndex", item_col="videoIndex", rating_col="engagement_score"):
    if "prediction" not in predictions.columns:
        print(f"Error in evaluate_als_hit_rate: 'prediction' col not found in predictions. Aborting metric calculation.")
        return 0.0

    print(f"Calculating HitRate@{k}. Relevant items in test set are any (user,item) pairs present.")

    user_window = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    top_k_preds = predictions.withColumn("rank", row_number().over(user_window)) \
        .filter(col("rank") <= k) \
        .select(user_col, item_col)

    relevant_test_items = test_df.select(user_col, item_col).distinct()

    users_with_items_in_test = relevant_test_items.select(user_col).distinct()
    num_users_with_items_in_test = users_with_items_in_test.count()

    if num_users_with_items_in_test == 0:
        print(f"Hit Rate at {k}: 0.0000 (No users with items in the test set)")
        return 0.0

    hits_df = top_k_preds.join(relevant_test_items, on=[user_col, item_col], how="inner")
    users_with_hits_count = hits_df.select(user_col).distinct().count()

    hit_rate = users_with_hits_count / num_users_with_items_in_test if num_users_with_items_in_test > 0 else 0.0
    print(f"Hit Rate at {k}: {hit_rate:.4f}")
    return hit_rate

def evaluate_als_mrr(test_df, predictions, k=10, user_col="userIndex", item_col="videoIndex", rating_col="engagement_score"):
    if "prediction" not in predictions.columns:
        print(f"Error in evaluate_als_mrr: 'prediction' col not found in predictions. Aborting metric calculation.")
        return 0.0

    print(f"Calculating MRR@{k}. Relevant items in test set are any (user,item) pairs present.")

    user_window = Window.partitionBy(user_col).orderBy(col("prediction").desc())
    ranked_predictions = predictions.withColumn("rank", row_number().over(user_window))

    top_k_ranked_predictions = ranked_predictions.filter(col("rank") <= k) \
        .select(user_col, item_col, "rank")

    relevant_test_items = test_df.select(user_col, item_col).distinct()

    users_for_mrr_denominator_df = relevant_test_items.select(user_col).distinct()
    num_users_for_mrr_denominator = users_for_mrr_denominator_df.count()

    if num_users_for_mrr_denominator == 0:
        print(f"MRR at {k}: 0.0000 (No users with items in the test set)")
        return 0.0

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

    mrr = total_sum_of_reciprocal_ranks / num_users_for_mrr_denominator if num_users_for_mrr_denominator > 0 else 0.0
    print(f"MRR at {k}: {mrr:.4f}")
    return mrr


def main():
    spark = SparkSession.builder \
    .appName("KuaiRec_ALS_Engagement_Score_Model") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
    base_path = "../data_final_project/KuaiRec 2.0/data/"
    train_path = base_path + "big_matrix.csv"
    test_path = base_path + "small_matrix.csv"
    items_path = base_path + "item_categories.csv"
    users_path = base_path + "user_features.csv"
    
    try:
        training_df, test_df = load_data(spark, train_path, test_path, items_path, users_path)
        
        training_df = clean_data(training_df)
        test_df = clean_data(test_df)

        if "engagement_score" not in training_df.columns:
            print("Error: 'engagement_score' column not found after clean_data. Exiting.")
            return
        if "engagement_score" not in test_df.columns:
            print("Error: 'engagement_score' column not found in test_df after clean_data. Exiting.")
            return

        print(f"Training dataset count after cleaning: {training_df.count()}")
        print(f"Test dataset count after cleaning: {test_df.count()}")
        
        print("Sample of training data with engagement_score (showing relevant columns):")

        final_cols_to_show = []
        potential_cols = ["user_id", "video_id", "watch_ratio", "capped_watch_ratio", "engagement_score"]

        for c in potential_cols:
            if c in training_df.columns:
                final_cols_to_show.append(c)
        
        if final_cols_to_show:
            training_df.select(final_cols_to_show).show(5, truncate=False)
        else:
            print("No columns selected for showing sample training data, or training_df is empty.")

        training, test = index_data(training_df, test_df)

        print("Configuring ALS model with 'engagement_score' as rating column...")
        als = ALS(
            maxIter=15,
            regParam=0.1,
            rank=10,
            userCol="userIndex",
            itemCol="videoIndex",
            ratingCol="engagement_score",
            coldStartStrategy="drop",
            nonnegative=True
        )
    
        model = als.fit(training)
        print("Generating predictions...")

        predictions = model.transform(test) 
        
        print(f"Predictions count: {predictions.count()}")
        if predictions.count() == 0:
            print("Warning: No valid predictions generated. Check your data and ALS parameters.")
            return

        if "prediction" not in predictions.columns:
            print("Error: 'prediction' column not found in model output. Schema:")
            predictions.printSchema()
            return
            
        print("Sample of predictions (showing relevant columns):")
        test_users_df = test.select(als.getUserCol()).distinct()

        num_recs_to_generate = 50

        user_recs_df = model.recommendForUserSubset(test_users_df, num_recs_to_generate)
        
        recommendations_for_eval_df = user_recs_df.select(
            col(als.getUserCol()),
            explode(col("recommendations")).alias("rec")
        ).select(
            col(als.getUserCol()).alias("userIndex"),
            col("rec." + als.getItemCol()).alias("videoIndex"),
            col("rec.rating").alias("prediction")
        )
        
        print(f"Generated recommendations count for evaluation: {recommendations_for_eval_df.count()}")
        if recommendations_for_eval_df.count() == 0:
            print("Warning: No recommendations generated by model.recommendForUserSubset. Metrics will be 0.")
        else:
            print("Sample of generated recommendations (showing relevant columns):")
            recommendations_for_eval_df.select("userIndex", "videoIndex", "prediction").show(5, truncate=False)

        k_eval = 20
        
        evaluate_als_precision(test, recommendations_for_eval_df, k_eval, user_col="userIndex", item_col="videoIndex")
        evaluate_als_recall(test, recommendations_for_eval_df, k_eval, user_col="userIndex", item_col="videoIndex")
        evaluate_als_hit_rate(test, recommendations_for_eval_df, k_eval, user_col="userIndex", item_col="videoIndex")
        evaluate_als_mrr(test, recommendations_for_eval_df, k_eval, user_col="userIndex", item_col="videoIndex")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping Spark session.")
        spark.stop()

   
if __name__ == "__main__":
    main()
