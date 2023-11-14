# CineSpark Insights

## Overview
CineSpark Insights is a data-driven project aimed at providing deep insights into the world of movies. Utilizing the power of Apache Spark, this project analyzes the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/) to uncover patterns and predictions in movie ratings and preferences.

### Objectives
- To perform extensive data analysis and manipulation using Spark RDDs and DataFrames.
- To build and compare machine learning models using Spark MLlib for predicting movie ratings.
- To gain actionable insights through data visualizations and model predictions.

###  Dataset
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

## Project Structure
```
CineSpark Insights/
│
├── src/                  - Source code for the project
│   ├── data_processing/  - Scripts for data manipulation and processing
│   ├── model_building/   - Machine learning models and predictions
│   └── visualization/    - Code for data visualizations
│
├── data/                 - Directory for storing MovieLens dataset files
│
├── docs/                 - Documentation and additional resources
│
├── notebooks/            - Jupyter notebooks for exploratory data analysis
│
└── results/              - Output results and figures
```

## Technologies Used
- **Apache Spark**: For distributed data processing and machine learning.
- **Python**: Primary programming language.
- **Jupyter Notebooks**: For interactive data exploration and analysis.
- **Other Libraries**: Matplotlib, Pandas, etc., for data manipulation and visualization.

## Setup and Installation
*Instructions on setting up the project environment and installation.*

```bash
# Example steps
$ pip install pyspark
$ pip install jupyter
# other necessary installations
```

## Usage
*Instructions on how to run the project, scripts, and notebooks.*

```bash
# Example command
$ spark-submit src/data_processing/data_processor.py
```

## Contributors
*List of team members and contributors to the project.*

## License
*State the project license here.*
