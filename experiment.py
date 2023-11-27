from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import time
import pandas as pd


def read_files(filename):
    """
    Reads the ratings file and returns the SparkSession and RDD created from the file.

    Args:
    filename (str): The path to the ratings file.

    Returns:
    SparkSession, RDD: The Spark session and the RDD created from the ratings file.
    """

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("MovieRecommend") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.instances", "4") \
        .config("spark.driver.memory", "8G") \
        .getOrCreate()

    # Retrieve SparkContext from SparkSession
    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    # Load the data into an RDD
    ratings_rdd = sc.textFile(filename)

    # Remove the header from the RDD and parse each line into a tuple
    header = ratings_rdd.first()
    ratings_rdd = (ratings_rdd.filter(lambda line: line != header)
                   .map(lambda line: line.split(','))
                   .map(lambda tokens: (tokens[0], tokens[1], float(tokens[2]), tokens[3])))

    num_ratings = ratings_rdd.count()
    print(f"Number of ratings: {num_ratings}")

    return spark, ratings_rdd


def basic_recommend(spark, ratings_rdd):
    """
    Performs basic movie recommendation based on the average rating.

    Args:
    spark (SparkSession): SparkSession object for DataFrame operations.
    ratings_rdd (RDD): The RDD containing movie ratings.
    """

    # Convert RDD to DataFrame for easier processing
    ratings_df = spark.createDataFrame(ratings_rdd, ["userId", "movieId", "rating", "timestamp"])

    # Compute the average rating for each movie
    avg_ratings_df = ratings_df.groupBy("movieId").avg("rating")

    # Retrieve top 5 movies based on average ratings
    top_movies = avg_ratings_df.orderBy("avg(rating)", ascending=False).limit(5)
    top_movies.show()

    # Plotting the top 5 movies and save the figure to a file
    top_movies_pd = top_movies.toPandas()   # Convert Spark DataFrame to Pandas DataFrame for plotting
    plt.figure(figsize=(10, 6))
    plt.bar(top_movies_pd['movieId'], top_movies_pd['avg(rating)'])
    plt.xlabel('Movie ID')
    plt.ylabel('Average Rating')
    plt.title('Top 5 Movies by Average Rating')
    plt.savefig('results/basic_recommend_top5_movies.png')
    plt.close()  # Close the plt object to free memory


def als_recommend(spark, ratings_rdd):
    """
    Performs movie recommendations using the ALS (Alternating Least Squares) model.

    Args:
    spark (SparkSession): SparkSession object for DataFrame operations.
    ratings_rdd (RDD): The RDD containing movie ratings.
    """

    # Convert RDD to DataFrame for easier processing
    ratings_rdd = ratings_rdd.map(lambda r: Row(userId=int(r[0]), movieId=int(r[1]), rating=float(r[2]), timestamp=r[3]))
    ratings_df = spark.createDataFrame(ratings_rdd, ["userId", "movieId", "rating", "timestamp"])

    # Split the dataset into training and test sets
    (training, test) = ratings_df.randomSplit([0.7, 0.3])

    # Create an ALS model
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)

    # Define a grid of parameters for tuning
    param_grid = {
        "rank": [10, 20],
        "maxIter": [5, 10],
        "regParam": [0.01, 0.1]
    }

    # Define multiple evaluators
    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    # Variables to keep track of the best model and its performance
    best_model = None
    best_error = float('inf')
    best_params = None
    results = []

    # Grid search through the parameter space
    for rank in param_grid["rank"]:
        for max_iter in param_grid["maxIter"]:
            for reg_param in param_grid["regParam"]:
                start_time = time.time()

                # Set model parameters
                als.setParams(rank=rank, maxIter=max_iter, regParam=reg_param)
                
                # Fit ALS model on training data
                model = als.fit(training)
                
                # Evaluate the model on test data
                predictions = model.transform(test)
                rmse = rmse_evaluator.evaluate(predictions)
                mae = mae_evaluator.evaluate(predictions)
                training_time = time.time() - start_time

                # Append results
                results.append({
                    "rank": rank, "maxIter": max_iter, "regParam": reg_param,
                    "RMSE": rmse, "MAE": mae, "Training Time": training_time
                })

                # Update best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = (rank, max_iter, reg_param)

    # Save the results DataFrame to a CSV file
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("results/als_model_performance.csv", index=False)

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Training Time'], results_df['RMSE'])
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Training Time')
    plt.savefig('RMSE_vs_Training_Time.png')
    plt.close()

    # Display the best model parameters and its RMSE
    print("Best Parameters:", best_params)
    print("Best RMSE:", best_error)

    # Generate top 3 movie recommendations for each user using the best model
    recommendations = best_model.recommendForAllUsers(3)
    recommendations.show(truncate=False)


def main():
    """
    Main function to execute the movie recommendation process.
    """

    filename = "dataset/ratings.csv"
    spark, ratings_rdd = read_files(filename)

    # basic_recommend(spark, ratings_rdd)

    als_recommend(spark, ratings_rdd)


if __name__ == "__main__":
    main()
