import pandas as pd
from transformers import pipeline

# Load the existing dataset
dataset_path = 'synthetic_amazon_reviews.csv'

try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at {dataset_path} was not found.")
    exit()
except pd.errors.ParserError:
    print(f"Error: Could not parse the CSV file at {dataset_path}.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Specify the model explicitly to suppress the warning
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)

# Define prompts for generating positive reviews
prompts = [
    "Generate a positive review for a vitamin supplement that boosts energy.",
    "Create a glowing review for a natural protein powder.",
    "Write an enthusiastic review for a multivitamin.",
]

# Function to determine if the review is positive
def is_positive_review(review_text):
    result = sentiment_analyzer(review_text)[0]
    return result['label'] == 'POSITIVE' and result['score'] > 0.90  # Use a threshold

# Filter for positive reviews
positive_reviews = df[df['text'].apply(is_positive_review)]

positive_reviews_df = positive_reviews[['rating', 'title', 'text']]

# Check the columns of the DataFrame
print("Columns in positive_reviews_df:", positive_reviews_df.columns)


# Generate new positive reviews based on prompts
def generate_positive_review(review_text):
    if is_positive_review(review_text):
        return prompts[0]  # You can change the prompt based on your requirements
    return review_text

# Create a new DataFrame for positive reviews
#positive_reviews_df = positive_reviews[['rating', 'title']]
#positive_reviews_df['text'] = positive_reviews_df['text'].apply(generate_positive_review)
positive_reviews_df['text'] = positive_reviews_df['text'].apply(generate_positive_review)

# Save the positive reviews to a new CSV file
output_path = 'positive_amazon_reviews.csv'
positive_reviews_df.to_csv(output_path, index=False)

print(f"Filtered dataset containing generated positive reviews saved to '{output_path}'")