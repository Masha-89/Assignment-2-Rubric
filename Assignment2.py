import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub  

# Loading the dataset
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
print("Path to dataset files:", path)  

# Loading the dataset into a DataFrame
df = pd.read_csv(
    "imdb_top_1000.csv",
    delimiter=";",
    on_bad_lines="skip",  # skip bad lines instead of erroring
    engine="python"       # use the Python engine for more flexible parsing
)

print(df.columns.tolist())

# Displaying the first few rows of the DataFrame
print(df.head())

# Defining the critical columns to retain
critical_columns = ['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']

# Dropping rows with missing values in these critical fields
df_cleaned = df.dropna(subset=critical_columns)

# Save or inspect cleaned data
print(df_cleaned.info())
print(df_cleaned.head())

# Checking how many complete duplicate rows exist
duplicate_rows = df_cleaned[df_cleaned.duplicated()]
print(f"Number of duplicate rows: {len(duplicate_rows)}")
# Removing duplicate rows
df_cleaned = df_cleaned.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df_cleaned)}")

# Remove "min" and convert to integer
duplicate_rows["Runtime"] = duplicate_rows["Runtime"].str.replace("min", "").str.strip()
duplicate_rows["Runtime"] = pd.to_numeric(duplicate_rows["Runtime"], errors="coerce")

# Convert to numeric, drop rows where year isn't valid
duplicate_rows["Released_Year"] = pd.to_numeric(duplicate_rows["Released_Year"], errors="coerce")

# Create a new Decade column
duplicate_rows["Decade"] = (duplicate_rows["Released_Year"] // 10 * 10).astype("Int64").astype(str) + "s"

actor_columns = ["Star1", "Star2", "Star3", "Star4"]
duplicate_rows["Lead_Actors"] = duplicate_rows[actor_columns].fillna("").astype(str).agg(", ".join, axis=1)
duplicate_rows["Lead_Actors"] = duplicate_rows["Lead_Actors"].str.replace(r"(, )+", ", ", regex=True).str.strip(", ")
