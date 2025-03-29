import pandas as pd
import numpy as np

def generate_rating_map(fico_scores, num_buckets):
    boundaries = np.percentile(fico_scores, np.linspace(0, 100, num_buckets+1))
    return boundaries

def map_fico_to_rating(fico, boundaries):
    num_buckets = len(boundaries) - 1
    bucket_index = np.searchsorted(boundaries, fico, side='right') - 1
    bucket_index = min(bucket_index, num_buckets - 1)
    rating = num_buckets - bucket_index
    return rating

if __name__ == "__main__":
    df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
    num_buckets = 5
    fico_scores = df['fico_score'].values
    boundaries = generate_rating_map(fico_scores, num_buckets)
    print("Bucket Boundaries:", boundaries)
    df['rating'] = df['fico_score'].apply(lambda x: map_fico_to_rating(x, boundaries))
    print(df[['fico_score', 'rating']].head())
    df.to_csv("Mortgage_Loan_Rating_Mapped.csv", index=False)