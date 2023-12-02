import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, udf, explode, split
from pyspark.sql.types import ArrayType, FloatType


def init_spark_session_with_ratings():
    """
    Reads the ratings file and returns the SparkSession and DataFrame created from the file.

    Returns:
    SparkSession, DataFrame: The Spark session and the DataFrame created from the ratings file.
    """

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("MovieRecommend") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.instances", "4") \
        .config("spark.driver.memory", "16G") \
        .getOrCreate()

    ratings_df = spark.read.csv("dataset/ratings.csv", header=True, inferSchema=True)
    print(ratings_df)
    ratings_df.show()

    return spark, ratings_df


def feature_engineering(spark, ratings_df):
    # relevance
    genome_scores_df = spark.read.csv("dataset/genome-scores.csv", header=True, inferSchema=True)
    take_top_n_udf = udf(lambda arr: sorted(arr, reverse=True)[:20], ArrayType(FloatType()))
    genome_scores_df = genome_scores_df.groupby('movieId').agg(
        take_top_n_udf(collect_list('relevance')).alias('relevance_list')
    )
    genome_scores_df = genome_scores_df.orderBy('movieId')
    genome_scores_df.show(truncate=False)

    # genre
    movies_df = spark.read.csv("dataset/movies.csv", header=True, inferSchema=True)
    movies_df = movies_df.withColumn("split_genres", split(col("genres"), "\|"))
    movies_exploded = movies_df.withColumn("genre", explode(col("split_genres")))

    genre_indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    indexed_genre = genre_indexer.fit(movies_exploded).transform(movies_exploded)
    genre_encoder = OneHotEncoder(inputCol="genreIndex", outputCol="genreVec")
    encoded_genre = genre_encoder.fit(indexed_genre).transform(indexed_genre)

    genre_aggregated = encoded_genre.groupBy("movieId").agg(collect_list("genreVec").alias("genreVecList"))

    sum_vectors_udf = udf(lambda vectors: Vectors.dense(np.sum([v.toArray() for v in vectors], axis=0)), VectorUDT())
    genre_aggregated = genre_aggregated.withColumn("genresVec", sum_vectors_udf("genreVecList"))
    genre_aggregated.show(truncate=False)

    # tag
    tags_df = spark.read.csv("dataset/tags.csv", header=True, inferSchema=True)
    genome_tags_df = spark.read.csv("dataset/genome-tags.csv", header=True, inferSchema=True)
    tags_df = tags_df.join(genome_tags_df, tags_df.tag == genome_tags_df.tag, "inner").select(tags_df["*"])

    tag_indexer = StringIndexer(inputCol="tag", outputCol="tagIndex")
    tag_model = tag_indexer.fit(genome_tags_df)
    indexed_tags_df = tag_model.transform(tags_df)
    tag_encoder = OneHotEncoder(inputCols=["tagIndex"], outputCols=["tagVec"])
    tag_encoder_model = tag_encoder.fit(indexed_tags_df)
    tags_encoded = tag_encoder_model.transform(indexed_tags_df)

    movie_tags_features = tags_encoded.groupBy('movieId').agg(collect_list('tagVec').alias('tagVectors'))
    sum_vectors_udf = udf(lambda vectors: Vectors.dense(np.sum([v.toArray() for v in vectors], axis=0)), VectorUDT())
    movie_tags_features = movie_tags_features.withColumn('tagFeatures', sum_vectors_udf('tagVectors'))
    movie_tags_features = movie_tags_features.drop("tagVectors")
    movie_tags_features.show(truncate=False)

    # join
    complete_data_df = ratings_df.join(genre_aggregated.select("movieId", "genresVec"), "movieId") \
        .join(movie_tags_features.select("movieId", "tagFeatures"), "movieId") \
        .join(genome_scores_df.select('movieId', 'relevance_list'), 'movieId', 'left')
    complete_data_df = complete_data_df.orderBy('userId')
    complete_data_df.show(truncate=False)

    return complete_data_df


def rf_train_evaluate(data_ready):
    """
    Train, evaluate, and visualize movie recommendations using a Random Forest model.

    Args:
    data_ready (DataFrame): DataFrame containing prepared features for training.
    """

    # Splitting the dataset
    training_features, test_features = data_ready.randomSplit([0.7, 0.3])

    # Training the Random Forest model with optimized parameters
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="rating",
        numTrees=5,  # 减少树的数量以加快训练
        maxDepth=5,  # 降低树的最大深度
        maxBins=32,
        minInstancesPerNode=1  # 每个节点的最小实例数
    )
    model = rf.fit(training_features)

    # Evaluate model on test dataset
    predictions_df = model.transform(test_features)
    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    # Calculate and output evaluation metrics
    rmse = rmse_evaluator.evaluate(predictions_df)
    mae = mae_evaluator.evaluate(predictions_df)
    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    return predictions_df


def main():
    """
    Main function to execute the movie recommendation process.
    """

    spark, ratings_df = init_spark_session_with_ratings()
    complete_data_df = feature_engineering(spark, ratings_df)
    predictions_df = rf_train_evaluate(complete_data_df)
    predictions_df.show()


if __name__ == "__main__":
    main()
