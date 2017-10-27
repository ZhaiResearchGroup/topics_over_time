import reddit_parser
from datetime import datetime

def get_reddit_data(subreddits, limit):
    """
    Parameters
    subreddits: the subreddits from which to retrieve data
    limit: the number of posts to retrieve from each subreddit

    Returns a list of documents and a list of timestamps.
    Documents: a blob of text containing the body of the post and comments on the post.
    Timestamps: a list of timestamps where the indices are aligned with the corresponding post.
    """
    posts = reddit_parser.format_posts(reddit_parser.get_posts(subreddits, limit))
    post_bodies = get_bodies(posts)

    timestamps = [post["created"] for post in posts]
    documents = build_documents(zip(posts, post_bodies))
    
    return documents, timestamps

def dump_reddit_data(documents, timestamps, documents_path, timestamps_path):
    """
    Parameters
    documents: the list of reddit posts and comments to be dumped
    timestamps: the list of timestamps with indices aligned to the documents to be dumped
    documents_path: the output file path for the documents
    timestamps_path: the output file path for the timestamps
    """
    with open(documents_path, 'wb') as outfile:
        for document in documents:
            outfile.write((document.replace('\n', '') + "\n").encode("utf-8"))

        outfile.close()

    with open(timestamps_path, 'w') as outfile:
        for timestamp in timestamps:
            outfile.write(str(timestamp) + "\n")

        outfile.close()

def get_bodies(items):
    """Gets all of the bodies associated with inputted posts or comments"""
    return [item['body'] for item in items]

def concatenate_comments_to_post(post_body, comments):
    """Returns a combined document of a post body and comment bodies"""
    comment_bodies = get_bodies(comments)
    document = post_body

    for comment in comment_bodies:
        document += "\n" + comment

    return document

def build_documents(post_pairs):
    """Combines the body and comments of a post into a single document
    Returns a list of documents
    """
    documents = []

    for post, post_body in post_pairs:
        comments = reddit_parser.get_post_comments(post)
        documents.append(concatenate_comments_to_post(post_body, comments))

    return documents

if __name__ == "__main__":
    subreddits = ['news']
    limit = 100
    reddit_data_file = 'data/reddit_documents.txt'
    timestamps_file = 'data/timestamps.txt'

    documents, timestamps = get_reddit_data(subreddits, limit)
    dump_reddit_data(documents, timestamps, reddit_data_file, timestamps_file)

    