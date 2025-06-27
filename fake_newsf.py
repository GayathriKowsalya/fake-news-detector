import os
import pandas as pd

# Define the paths
base_path = os.getcwd()
fake_path = os.path.join(base_path, 'data', 'Fake.csv')
real_path = os.path.join(base_path, 'data', 'True.csv')

# Debug print
print(f"Fake file path: {fake_path}")
print(f"Real file path: {real_path}")

# Check file existence and size
if not os.path.exists(fake_path) or not os.path.exists(real_path):
    print("❌ ERROR: One or both data files do not exist.")
    exit()

if os.path.getsize(fake_path) == 0 or os.path.getsize(real_path) == 0:
    print("❌ ERROR: One or both data files are empty.")
    exit()

# Load both datasets
fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

print("✅ Successfully loaded both datasets!")

# Add labels
fake_df['label'] = 0  # Fake news
real_df['label'] = 1  # Real news
data = pd.concat([fake_df, real_df])

# Combine datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Shuffle the combined dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Show dataset info
print(f"Combined dataset shape: {df.shape}")
print(f"Sample rows:\n{df[['title', 'label']].head()}")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Clean the text (optional but useful)
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Lowercase
    return text

# Apply cleaning function
data['text'] = data['text'].astype(str).apply(clean_text)

# Drop unwanted columns
data = data.drop(['date', 'subject'], axis=1, errors='ignore')

# Check for null values
data = data.dropna()

# Split features and labels
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("✅ Preprocessing and vectorization completed.")
