import requests
import json
import uuid
from reddit_constants import default_user_agent

# example JSON is available in post.json, subreddit.json, and output.json

def get_posts(subreddits, limit, user_agent=default_user_agent):
    """Returns a specified number of posts from a specified group of subreddits."""
    all_posts = []

    for subreddit in subreddits:
        print subreddit
        data_url = 'https://www.reddit.com/r/{}.json?limit={}'.format(subreddit, limit)
        response_data = requests.get(data_url, headers = {'User-agent': user_agent})

        posts = response_data.json()['data']['children']

        all_posts.extend(posts)

    return all_posts

def format_posts(posts):
    """Returns a list of posts with the desired extracted fields"""
    formatted_posts = []

    for post in posts:
        post_data = post['data']
        formatted_post = {
            "title": post_data['title'],
            "post_id": post_data['id'],
            "subreddit": post_data['subreddit'],
            "score": post_data['score'],
            "url": post_data['url'],
            "author": post_data['author'],
            "permalink": format_post_permalink(post_data['permalink']),
            "num_comments": post_data['num_comments'],
            "created": post_data['created'],
            "body": post_data['selftext']
        }

        formatted_posts.append(formatted_post)

    return formatted_posts

def format_post_permalink(post_permalink):
    """Returns a formatted url for the post json"""
    return 'https://www.reddit.com' + post_permalink + '.json' 

def get_post_comments(post, user_agent=default_user_agent):
    """Returns the comment data for a specified post."""
    post_permalink = post['permalink']

    response_data = requests.get(post_permalink, headers = {'User-agent': user_agent})
    post_data = response_data.json()[1]

    # right now this gets the title, eventually convert to unique id for each title
    post_id = post['post_id']

    return get_post_comments_recur(post_data, [], -1, post_id)

def get_post_comments_recur(comment, comments, parent_comment_id, parent_post_id):
    """Recursive helper function to gather all comments in a Reddit thread.
    
    Right now this function is only getting 200 comments because that is all is shown on the
    JSON page for the post. There are properties called 'more' with lists of id's. Perhaps 
    those can be a link to a complete set of comments.
    """
    if 'data' in comment and type(comment) is dict:
        comment_data = comment['data']

        new_comment = None

        # a new comment exists at this layer, add it to the total list of comments
        if 'body' in comment_data:
            new_comment = {
                "score": comment_data['score'],
                "body": comment_data['body'],
                "subreddit": comment_data['subreddit'],
                "author": comment_data['author'],
                "parent_comment_id": parent_comment_id,
                "parent_post_id": parent_post_id,
                "created": comment_data['created'],
                "comment_id": comment_data['id']
            }
            comments.append(new_comment)

        next_parent_comment_id = parent_comment_id if new_comment is None else new_comment['comment_id']

        # recurse on children
        if 'children' in comment_data:
            for child in comment_data['children']:
                comments = get_post_comments_recur(child, comments, next_parent_comment_id, parent_post_id)

        # recurse on replies
        if 'replies' in comment_data:
            comments = get_post_comments_recur(comment_data['replies'], comments, next_parent_comment_id, parent_post_id)

    return comments

if __name__ == "__main__":

    subreddits = []
    with open('subreddits.txt', 'r') as f:
        subreddits = f.read().split('\n')

    print subreddits
    limit = 500
    user_agent = 'ResearchBot'

    posts = get_posts(subreddits, limit)

    formatted_posts = format_posts(posts)

    print(json.dumps(formatted_posts))

    post_comments = get_post_comments(formatted_posts[0])
    print(json.dumps(post_comments))