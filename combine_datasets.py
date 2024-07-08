# combine datasets to see differences

import pandas as pd

# Load the CSV files
results_df = pd.read_csv('results_10M_val.csv')
rewrite_df = pd.read_csv('rewrite_10M_val.csv')

# Rename the 'name' columns
results_df = results_df.rename(columns={"name": "original_caption"})
rewrite_df = rewrite_df.rename(columns={"name": "new_caption"})

# drop videoid column
results_df = results_df.drop(columns=['videoid'])
rewrite_df = rewrite_df.drop(columns=['videoid'])

# drop duration column
results_df = results_df.drop(columns=['duration'])
rewrite_df = rewrite_df.drop(columns=['duration'])

# drop page_dir column
results_df = results_df.drop(columns=['page_dir'])
rewrite_df = rewrite_df.drop(columns=['page_dir'])

# Join the dataframes on 'contentUrl'
joined_df = pd.merge(results_df, rewrite_df, on="contentUrl", how="inner")

# reorder columns as: original_caption,new_caption,contentUrl
joined_df = joined_df[['original_caption', 'new_caption', 'contentUrl']]

# Save the joined dataframe to a new CSV file if needed
joined_df.to_csv('joined_file.csv', index=False)
