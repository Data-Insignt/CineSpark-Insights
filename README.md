# CineSpark Insights

## Overview
CineSpark Insights is a data-driven project focused on providing in-depth insights into the cinematic world. Leveraging Apache Spark, this project processes the MovieLens 25M Dataset to reveal patterns and trends in movie ratings and preferences.

### Objectives
- To conduct thorough data analysis using Spark RDDs and DataFrames.
- To develop and compare machine learning models, specifically ALS and Random Forest, for predicting movie ratings.
- To derive actionable insights through advanced data visualizations and predictive modeling.

## Project Structure
```
CineSpark Insights/
│
│
├── experiment.ipynb        - Comprehensive overall code notebook detailing the project workflow
│
├── deployment/             - Spark and Hadoop deployment configuration
│   ├── images/             - Docker images for Hadoop and Spark
│   ├── config/             - Configuration files for Hadoop cluster
│   ├── docker-compose.yaml - Docker Compose configuration for the cluster
│
├── model/                  - Source code for the recommendation models
│   ├── als.py              - Alternating Least Squares model implementation
│   ├── rf.py               - Random Forest model implementation
│
├── dataset/                - MovieLens dataset files
├── visualization/          - Visualization scripts and web interface
└── results/                - Output results, figures, and analysis
```

## Detailed Analysis in `experiment.ipynb`
- **Environment Import**: Setting up the necessary environment for data processing.
- **Part 1: Init Spark Session with Ratings**: Starting Spark session and loading the ratings data.
- **Part 2: Basic Recommend with Visualization**: Implementing a basic recommendation system and visualizing the results.
- **Part 3: ALS Recommend with Visualization**: Advanced recommendations using the ALS model, accompanied by visualizations.
- **Part 4: RF Recommend with Visualization**: Implementing and visualizing recommendations using the Random Forest model.
- **Part 5: Model Comparison**: Comparing the effectiveness of the ALS and RF models.


### Dataset Overview
The project uses various components of the MovieLens 25M Dataset, including:
+ [ratings.csv](dataset/ratings.csv)
  + Purpose: Store user ratings for movies. 
  + Format: Including userId (user ID), movieId (movie ID), rating (rating), and timestamp (timestamp). 
  + Features: 
    + The rating is based on a five-star system, with half-star increments (from 0.5 stars to 5 stars). 
    + The timestamp is the number of seconds since midnight UTC on January 1, 1970.
+ [tags.csv](dataset/tags.csv)
  + Purpose: Record the tags users give to movies. 
  + Format: Including userId (user ID), movieId (movie ID), tag (tag), and timestamp (timestamp). 
  + Features: 
    + Tags are user-generated, metadata about a movie, usually a word or phrase. 
    + Timestamps also represent the number of seconds since midnight UTC on January 1, 1970.
+ [movies.csv](dataset/movies.csv)
  + Purpose: Provide basic information about the movie. 
  + Format: including movieId (movie ID), title (title), genres (type). 
  + Features: Movie titles are entered manually or imported from The Movie Database, including year of release. Genres are a pipe-separated list of common movie genres.
+ [links.csv](dataset/links.csv)
  + Purpose: Provide identifiers that link to other movie data sources. 
  + Format: Including movieId (movie ID of MovieLens website), imdbId (movie ID of IMDb), tmdbId (movie ID of The Movie Database).
+ **Tag Genome**:
  + Purpose: Tag Genome is a more complex and comprehensive system designed to quantify and structure the properties of films. It contains a comprehensive score of movie tag relevance. 
  + File structure:
    + [genome-scores.csv](dataset/genome-scores.csv): Contains correlation scores between movies and tags. Each movie has a relevance score for each tag, which are calculated via machine learning algorithms based on user-contributed content such as tags, ratings, and text comments. 
    + [genome-tags.csv](dataset/genome-tags.csv): Provides tag IDs used in tagged genomes and their descriptions. 
  + Features: Tag Genome encodes movie attributes (such as atmosphere, thought-provoking, realistic, etc.) through tags. It is a dense matrix that provides a score for each tag for each movie.

## Technologies Used
- **Apache Spark**: For large-scale data processing and machine learning.
- **Python**: As the primary programming language.
- **Docker**: For containerizing the Hadoop and Spark environment.
- **Jupyter Notebook**: For interactive data exploration and visualization.
- **Other Libraries**: Including Matplotlib, Pandas for data handling and visualization.


## Running the Project
1. **Prerequisites**: Ensure Docker, Apache Spark, and Python are installed.
2. **Environment Setup**: Follow the instructions in the `deployment/` folder to set up the Dockerized Spark cluster.
3. **Executing the Notebook**: Run `experiment.ipynb` for a step-by-step execution of the project, including data preprocessing, model training, and visualization.