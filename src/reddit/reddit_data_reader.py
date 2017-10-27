def read_reddit_data_and_timestamps(reddit_data_path, timestamps_path, stopwords_path):
    """
    Parameters
    reddit_data_path: input file containing reddit data documents
    timestamps_path: input file containing timestamps for the documents
    stopwords_path: input file containing stopwords to be filtered

    Returns
    documents: a list of text combining a reddit post and its comments filtered for stopwords
    timestamps: a list of timestamps for the documents index aligned with the documents list
    unique_words: a list of all of the unique words in the reddit documents that are not stopwords
    """
    documents = []
    filtered_documents = []
    timestamps = []
    unique_words = set()
    stopwords = set()

    with open(stopwords_path, 'r') as stopwords_file:
        lines = stopwords_file.readlines()
        for line in lines:
            stopwords.update(set(line.lower().strip().split()))

        stopwords_file.close()

    with open(reddit_data_path, 'rb') as documents_file:
        documents = documents_file.readlines()
        documents_file.close()

    with open(timestamps_path, 'r') as timestamps_file:
        timestamps = timestamps_file.readlines()
        timestamps_file.close()

    for i in range(0, len(documents)):
        document = documents[i]
        timestamp = timestamps[i]

        new_words = [word for word in document.lower().strip().split() if word not in stopwords]
        unique_words.update(set(new_words))

        filtered_documents.append(new_words)

    timestamps = [int(timestamp[0:timestamp.index('.')]) for timestamp in timestamps]

    return filtered_documents, timestamps, list(unique_words)

if __name__ == "__main__":
    documents, timestamps, unique_words = read_reddit_data_and_timestamps('data/reddit_documents.txt', 'data/timestamps.txt', 'data/stopwords.txt')
  
    print(len(documents))
    print(len(timestamps))
    print(len(unique_words))
