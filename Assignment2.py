import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub  
from scipy.stats import pearsonr

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
duplicate_rows["Lead_Actors"] = duplicate_rows[actor_columns].fillna("").astype(str).agg(lambda row: ", ".join(row.astype(str)), axis=1)
duplicate_rows["Lead_Actors"] = duplicate_rows["Lead_Actors"].astype(str).replace(r"(, )+", ", ", regex=True).str.strip(", ")
# Display the cleaned DataFrame
print(duplicate_rows.head())

# Data Visualization

df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")

# Plotting the distribution of IMDB Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df["IMDB_Rating"], color="skyblue", label="IMDB Rating", bins=20, kde=True)
sns.histplot(df["Meta_score"], color="orange", label="Meta Score", bins=20, kde=True)

plt.title("Distribution of IMDB Ratings vs Meta Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Drop rows with missing Genre
df = df.dropna(subset=["Genre"])

# Explode genres into separate rows
genre_series = df["Genre"].str.split(', ').explode()

# Count top 10 genres
top_genres = genre_series.value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis")

plt.title("Top 10 Genre Frequencies")
plt.xlabel("Frequency")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

df = df.dropna(subset=["Gross", "No_of_Votes"])
# Convert to numeric (remove commas and cast to int/float)
df["Gross"] = df["Gross"].replace('[\$,]', '', regex=True).astype(float)
df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

plt.title("Scatter Plot of Gross Revenue vs. Number of Votes")
plt.xlabel("Number of Votes")
plt.ylabel("Gross Revenue (in $)")
plt.xscale("log")  # Optional: log scale for better distribution
plt.yscale("log")  # Optional
plt.tight_layout()
plt.show()

# Drop missing values in Certificate and IMDB_Rating
df = df.dropna(subset=["Certificate", "IMDB_Rating"])

df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
plt.figure(figsize=(12, 6))
sns.boxplot(x="Certificate", y="IMDB_Rating", data=df, palette="Set2")

plt.title("IMDB Rating by Certificate")
plt.xlabel("Certificate")
plt.ylabel("IMDB Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Applied Statistical Analysis
# Make sure columns are numeric
df["Gross"] = pd.to_numeric(df["Gross"], errors='coerce')
df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"], errors='coerce')
df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors='coerce')

# Drop rows with missing values in these columns to avoid skewing statistics
df_stats = df.dropna(subset=["Gross", "No_of_Votes", "IMDB_Rating"])

# Calculate descriptive statistics
mean_values = df_stats[["Gross", "No_of_Votes", "IMDB_Rating"]].mean()
median_values = df_stats[["Gross", "No_of_Votes", "IMDB_Rating"]].median()
std_values = df_stats[["Gross", "No_of_Votes", "IMDB_Rating"]].std()

# Present results
print("Mean values:\n", mean_values)
print("\nMedian values:\n", median_values)
print("\nStandard Deviation values:\n", std_values)

# Correlation Analysis
corr_coef, p_value = pearsonr(df['Gross'], df['No_of_Votes'])
print(f"Correlation coefficient between Gross and Number of Votes: {corr_coef:.2f}")
print(f"P-value: {p_value:.4f}")

# Ensure Gross is numeric and drop missing values
df_gross = df[["Gross"]].dropna()
df_gross["Gross"] = pd.to_numeric(df_gross["Gross"], errors="coerce")

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_gross["Gross"].quantile(0.25)
Q3 = df_gross["Gross"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df_gross[(df_gross["Gross"] < lower_bound) | (df_gross["Gross"] > upper_bound)]

print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print(f"Number of outliers detected: {len(outliers)}")

# Directors with the highest average gross revenue

# Convert Gross to numeric (remove commas or $ if needed)
df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

avg_gross_by_director = df.groupby("Director")["Gross"].mean().sort_values(ascending=False)
top_directors = avg_gross_by_director.head(10)
print("Top 10 Directors by Average Gross Revenue:\n", top_directors)

# plotting top 5 directors by gross 

top_5_directors = avg_gross_by_director.head(5)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_5_directors.index, y=top_5_directors.index, palette="viridis")
plt.title("Top 5 Directors by Average Gross Revenue")
plt.xlabel("Average Gross Revenue (USD)")
plt.ylabel("Director")
plt.tight_layout()
plt.show()

top_rated = df[df["IMDB_Rating"] > 8.5]
top_star_counts = top_rated["Star1"].value_counts()
top_star_counts.head(5)

df["Actor_Pair"] = df["Star1"] + " & " + df["Star2"]
actor_pair_gross = df.groupby("Actor_Pair")["Gross"].mean().sort_values(ascending=False)
actor_pair_gross.head(5)

top_5_pairs = actor_pair_gross.head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_pairs.values, y=top_5_pairs.index, palette="magma")
plt.title("Top 5 Actor Pairs by Average Gross Revenue")
plt.xlabel("Average Gross (USD)")
plt.ylabel("Actor Pair")
plt.tight_layout()
plt.show()

df_genre = df.copy()
df_genre["Genre"] = df_genre["Genre"].str.split(", ")
df_genre = df_genre.explode("Genre")

genre_rating = df_genre.groupby("Genre")["IMDB_Rating"].mean().sort_values(ascending=False)

genre_rating.head(10)

pivot_table = df_genre.pivot_table(
    index="Genre", 
    values="IMDB_Rating", 
    aggfunc="mean"
).sort_values("IMDB_Rating", ascending=False)

# Plot heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Average IMDB Rating by Genre", fontsize=14)
plt.xlabel("IMDB Rating")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()