import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

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

# Prepare data for visualization
top_words = []
for topic, words in enumerate(lda_topic_model.components_):
    sorted_words = words.argsort()[::-1]  # Sort in descending order
    top_words.append({
        'topic': topic + 1,
        'words': [feature_names[i] for i in sorted_words[:number_of_words]],
        'weights': [words[i] for i in sorted_words[:number_of_words]],
        'total_weight': words.sum()  # Sum of all weights for this topic
    })


# Sort topics by total weight
top_words = sorted(top_words, key=lambda x: x['total_weight'], reverse=True)

# Create subplots
fig, axes = plt.subplots(nrows=num_topics, ncols=1, figsize=(10, 6 * num_topics), constrained_layout=True)

# Plot bar chart for each topic
for i, data in enumerate(top_words):
    sns.barplot(ax=axes[i], x=data['weights'], y=data['words'], palette='viridis')
    axes[i].set_title(f'Top {number_of_words} Words for Topic {data["topic"]} (Total Weight: {data["total_weight"]:.2f})')
    axes[i].set_xlabel('Weight')
    axes[i].set_ylabel('Word')

# Adjust layout
plt.show()

'''
#create CSV

# Assign each document to the topic with the highest weight
topic_data_frame['Assigned_Topic'] = topic_data_frame.idxmax(axis=1)

# Create a column for the highest weight (probability) for the assigned topic
topic_data_frame['Max_Weight'] = topic_data_frame.max(axis=1)

# Merge the original data with the topic assignments
bbc_data_with_topics = bbc_data.copy()
bbc_data_with_topics['Assigned_Topic'] = topic_data_frame['Assigned_Topic']
bbc_data_with_topics['Max_Weight'] = topic_data_frame['Max_Weight']

# Order the rows by the assigned topic and then by the relevance (Max_Weight)
bbc_data_with_topics = bbc_data_with_topics.sort_values(by=['Assigned_Topic', 'Max_Weight'], ascending=[True, False], ignore_index=True)


# Save to CSV
bbc_data_with_topics.to_csv('bbc_data_with_topics.csv', index=False)

print("CSV file 'bbc_data_with_topics.csv' created successfully!")

'''

# Identify the documents with the highest weights for Topic 2
top_comments_topic_2 = topic_data_frame[topic_data_frame['Topic 2'] == topic_data_frame['Topic 2'].max()]

# Retrieve the indices of the top comments for Topic 2
top_comment_indices_topic_2 = top_comments_topic_2.index

# Extract and display the top comments for Topic 2
print("Top Comments for Topic 2 (Blogs and Blogging):")
for index in top_comment_indices_topic_2:
    print(bbc_data.iloc[index]['data'])

