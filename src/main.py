import pandas as pd
from transformers import pipeline

# Define your class titles, descriptions, and examples
class_data = [
    {
        "title": "Sports",
        "description": "News and updates about various sports."
    },
    {
        "title": "Science",
        "description": "Discover the latest scientific advancements."
    },
    {
        "title": "Technology",
        "description": "Stay informed about tech trends and innovations."
    },
]

# Prepare the input data
input_titles = [item["title"] for item in class_data]
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

# Extract labels and scores for each text classification
label_lists = [result['labels'] for result in results]
score_lists = [result['scores'] for result in results]

# Create a list of dictionaries for DataFrame construction using list comprehension
data = []

for i, text in enumerate(texts_to_classify):
    row_data = {'Text': text}
    for label, score in zip(label_lists[i], score_lists[i]):
        row_data[label] = score
    data.append(row_data)

# Create a DataFrame
df = pd.DataFrame(data)

# Fill NaN values with 0
df.fillna(0, inplace=True)

df.to_csv("output/temp.csv")

# Display the DataFrame
print(df)