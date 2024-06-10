import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the data
bbc_data = pd.read_csv("bbc_data.csv")
print(bbc_data.info())
print(bbc_data.head())

# Tokenize and transform the data using TF-IDF
vector = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.75)
matrix = vector.fit_transform(bbc_data['data'])
print(f'rows: {matrix.shape[0]}, columns: {matrix.shape[1]}')

# Perform LDA
num_topics = 5
lda_topic_model = LatentDirichletAllocation(n_components=num_topics, random_state=12345)
matrix = lda_topic_model.fit_transform(matrix)

# Create a DataFrame to display each document's topic assignments
column_names = [f'Topic {x}' for x in range(1, num_topics + 1)]
topic_data_frame = pd.DataFrame(matrix, columns=column_names)
print(topic_data_frame.head(n=50))  # Display the first 50 rows

# Find topics
# Display the top X words for each topic
number_of_words = 6
feature_names = vector.get_feature_names_out()  # Get feature names from the vectorizer
for topic, words in enumerate(lda_topic_model.components_):
    word_total = words.sum()  # Total word weight for that topic
    sorted_words = words.argsort()[::-1]  # Sort in descending order
    print(f'\nTopic {topic + 1:02d}')  # Print the topic
    for i in range(0, number_of_words):
        word = feature_names[sorted_words[i]]
        word_weight = words[sorted_words[i]]
        print(f' {word} ({word_weight})')



