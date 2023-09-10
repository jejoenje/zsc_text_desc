import pandas as pd
from transformers import pipeline

# Define your class titles, descriptions, and examples
class_data = [
    {
        "title": "Sports",
        "description": "News and updates about various sports.",
        "examples": [
            "A thrilling football match took place yesterday.",
            "The Olympics are coming up this year.",
            "Tennis championships concluded with an exciting final match."
        ],
    },
    {
        "title": "Science",
        "description": "Discover the latest scientific advancements.",
        "examples": [
            "Scientists have discovered a new species of butterfly.",
            "The Mars rover sent back fascinating data about the planet's surface.",
            "Researchers made a breakthrough in cancer treatment."
        ],
    },
    {
        "title": "Technology",
        "description": "Stay informed about tech trends and innovations.",
        "examples": [
            "Apple just released a new iPhone model with exciting features.",
            "Artificial intelligence is revolutionizing various industries.",
            "The tech industry is abuzz with rumors of a new product launch."
        ],
    },
]

# Prepare the input data
input_data = [f"Classify the text as {item['title']}: {item['description']}" for item in class_data]

# Load the zero-shot text classification pipeline
classifier = pipeline("zero-shot-classification")

# List of input text strings to classify
texts_to_classify = [
    "This article discusses the latest developments in artificial intelligence.",
    "Yesterday's tennis match was incredibly intense.",
    "A new scientific paper on climate change was published.",
]

# Perform zero-shot classification for each input text
results = classifier(texts_to_classify, input_data)

# Create a DataFrame to format the results
result_df = pd.DataFrame(results)

# Pivot the DataFrame to have labels as columns and scores as values
result_df = result_df.pivot_table(columns='labels', values='scores', aggfunc='first').reset_index()

# Display the result DataFrame
print(result_df)