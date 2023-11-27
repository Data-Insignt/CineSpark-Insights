from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import explode, col
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
    # sc.setLogLevel("INFO")

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
    best_rmse = float('inf')
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

    # Display the best model parameters and its RMSE
    print("Best Parameters:", best_params)
    print("Best RMSE:", best_error)

    # Generate top 5 movie recommendations for each user using the best model
    recommendations = best_model.recommendForAllUsers(5)
    recommendations.show(truncate=False)

    # Explode the recommendations to create a row for each movie
    recs_exploded = recommendations.withColumn("rec_exp", explode("recommendations")).select("userId", col("rec_exp.movieId"), col("rec_exp.rating"))

    # Plotting the number of times each movie is recommended
    recs_pd = recs_exploded.toPandas()
    plt.figure(figsize=(10, 6))
    recs_pd['movieId'].value_counts().head(5).plot(kind='bar')
    plt.xlabel('Movie ID')
    plt.ylabel('Number of Recommendations')
    plt.title('Top 5 Recommended Movies')
    plt.savefig('results/als_recommended_top5_movies.png')
    plt.close()



def als_recommend_best(spark, ratings_rdd):
    """
    Performs movie recommendations using the ALS (Alternating Least Squares) model with optimal parameters.

    Args:
    spark (SparkSession): SparkSession object for DataFrame operations.
    ratings_rdd (RDD): The RDD containing movie ratings.
    """

    # Convert RDD to DataFrame for easier processing
    ratings_rdd = ratings_rdd.map(lambda r: Row(userId=int(r[0]), movieId=int(r[1]), rating=float(r[2]), timestamp=r[3]))
    ratings_df = spark.createDataFrame(ratings_rdd, ["userId", "movieId", "rating", "timestamp"])

    # Split the dataset into training and test sets
    (training, test) = ratings_df.randomSplit([0.7, 0.3])

    # Define multiple evaluators
    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    # Create an ALS model
    als = ALS(rank=10,maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
   
    # Fit ALS model on training data
    model = als.fit(training)

    # Evaluate the model on test data
    predictions = model.transform(test)
    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Generate top 5 movie recommendations for each user using the best model
    recommendations = model.recommendForAllUsers(5)
    recommendations.show(truncate=False)

    # Prepare data for visualization
    recs_exploded = recommendations.withColumn("rec_exp", explode("recommendations")).select("userId", col("rec_exp.movieId"), col("rec_exp.rating"))
    movies_df = spark.read.csv("dataset/movies.csv", header=True, inferSchema=True)    # Movies: movieId, title, genres
    recs_joined = recs_exploded.join(movies_df, "movieId").select("userId", "title", "rating")
    recs_pd = recs_joined.toPandas()

    # Count the number of recommendations for each movie and get the top 5
    top_movies = recs_pd['title'].value_counts().head(5)

    # Plotting the number of times each movie is recommended using the movie titles
    plt.figure(figsize=(12, 8))
    top_movies.plot(kind='bar')
    plt.xlabel('Movie Title')  # Changed from 'Movie ID' to 'Movie Title'
    plt.ylabel('Number of Recommendations')
    plt.title('Top 5 Recommended Movies by ALS Model')
    plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust the layout to fit the labels
    plt.savefig('results/als_recommended_top5_movies.png')
    plt.close()


def random_forest_recommend(spark, ratings_file, movies_file, tags_file):
    """
    Perform movie recommendations using a Random Forest model.

    Args:
    spark (SparkSession): SparkSession object for DataFrame operations.
    ratings_file (str): Path to the ratings.csv file.
    movies_file (str): Path to the movies.csv file.
    tags_file (str): Path to the tags.csv file.
    """

    # Load and preprocess datasets
    ratings_df = spark.read.csv(ratings_file, header=True, inferSchema=True)    # Ratings: userId, movieId, rating, timestamp
    movies_df = spark.read.csv(movies_file, header=True, inferSchema=True)    # Movies: movieId, title, genres
    tags_df = spark.read.csv(tags_file, header=True, inferSchema=True)    # Tags: userId, movieId, tag, timestamp

    # Process genres from movies dataset
    # Using StringIndexer to convert genre strings to genre indices
    stringIndexer = StringIndexer(inputCol="genres", outputCol="genresIndex")
    model = stringIndexer.fit(movies_df)
    indexed = model.transform(movies_df)
    # Using OneHotEncoder to convert genre indices to binary vector
    encoder = OneHotEncoder(inputCol="genresIndex", outputCol="genresVec")
    movies_encoded = encoder.transform(indexed)

    # Joining tags and ratings data
    movie_tags_df = tags_df.join(ratings_df, ["userId", "movieId"])
    # Combining movie information with tag features
    movie_features_df = movies_encoded.join(movie_tags_df, "movieId")
    # Merging user ratings with movie features
    complete_data_df = ratings_df.join(movie_features_df, "movieId")

    # Feature Vectorization
    assembler = VectorAssembler(inputCols=["genresVec", "tagFeatures"], outputCol="features")
    data_ready = assembler.transform(complete_data_df)

    # Splitting the dataset
    (training_features, test_features) = data_ready.randomSplit([0.7, 0.3])

    # Training the Random Forest model
    rf = RandomForestRegressor(featuresCol="features", labelCol="rating")
    rf_model = rf.fit(training_features)

    # Making predictions on the test dataset
    predictions_rf = rf_model.transform(test_features)

    # Evaluating the model
    evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions_rf)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


def main():
    """
    Main function to execute the movie recommendation process.
    """

    filename = "dataset/ratings.csv"
    spark, ratings_rdd = read_files(filename)

    # basic_recommend(spark, ratings_rdd)

    # als_recommend(spark, ratings_rdd)
    als_recommend_best(spark, ratings_rdd)


if __name__ == "__main__":
    main()
