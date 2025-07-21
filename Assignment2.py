import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
import kagglehub  
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
print("Path to dataset files:", path)  

# Load the dataset into a DataFrame
import pandas as pd

df = pd.read_csv(
    "imdb_top_1000.csv",
    delimiter=";",
    on_bad_lines="skip",  # skip bad lines instead of erroring
    engine="python"       # use the Python engine for more flexible parsing
)

print(df.columns.tolist())

# Display the first few rows of the DataFrame
print(df.head())

# Define the critical columns to retain
critical_columns = ['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']

# Drop rows with missing values in these critical fields
df_cleaned = df.dropna(subset=critical_columns)

# Save or inspect cleaned data
print(df_cleaned.info())
print(df_cleaned.head())

# Check how many complete duplicate rows exist
duplicate_rows = df_cleaned[df_cleaned.duplicated()]
print(f"Number of duplicate rows: {len(duplicate_rows)}")
# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df_cleaned)}")

