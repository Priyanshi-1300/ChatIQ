from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
extractor = URLExtract()  # Create a URL extractor instance


# Function to fetch statistics
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # Total number of messages
    num_messages = df.shape[0]

    # Total number of words
    words = []
    for message in df['Message']:
        words.extend(message.split())

    # Number of media messages
    num_media_messages = df[df['Message'] == '<Media omitted>\n'].shape[0]

    # Number of links shared
    links = []
    for message in df['Message']:
        links.extend(extractor.find_urls(message))

    # --- New Metrics ---
    total_chars = df['Message'].str.len().sum()  # total characters
    avg_message_length = df['Message'].str.len().mean()  # average chars per message

    return num_messages, len(words), num_media_messages, len(links), total_chars, avg_message_length

# Function to find the most busy users
def most_busy_user(df):
    user_counts = df['User'].value_counts().head()
    percentage_df = (
        (df['User'].value_counts() / df.shape[0]) * 100
    ).reset_index().rename(columns={'index': 'percent', 'User': 'Name'})
    return user_counts, percentage_df


from wordcloud import WordCloud
import re

def remove_emojis(text):
    # Remove emojis and non-ASCII characters
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002700-\U000027BF"  # dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def create_wordcloud(selected_User, df):
    if selected_User != 'Overall':
        df = df[df['user'] == selected_User]

    # Combine all messages
    text = " ".join(df['Message'])

    # Clean text to remove emojis and unsupported symbols
    clean_text = remove_emojis(text)

    # Generate word cloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(clean_text)
    return df_wc



# Function to find most common words
def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    temp = df[df['User'] != 'group_notification']
    temp = temp[temp['Message'] != '<Media omitted>\n']

    words = []
    for message in temp['Message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Count'])
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    emojis=[]
    for Message in df['Message']:
        emojis.extend([c for c in Message if c in emoji.EMOJI_DATA])
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    timeline=df.groupby(['year','month_num','month']).count()['Message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+ "-"+str(timeline['year'][i]))
    timeline['time']=time
    return timeline

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    daily_timeline=df.groupby('only_date').count()['Message'].reset_index()

    return daily_timeline

def weak_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='Message', aggfunc='count').fillna(0)
    return user_heatmap


def avg_response_time(df, selected_user='Overall'):
    """
    Calculate average response time (in minutes) per user,
    ignoring group notifications.
    """
    # Remove system notifications
    df = df[df['User'] != 'group_notification']

    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    df = df.sort_values('date').reset_index(drop=True)

    # Only consider messages from a different user than previous
    mask = df['User'] != df['User'].shift(1)
    df['delta_min'] = (df['date'] - df['date'].shift(1)).dt.total_seconds() / 60.0

    responses = df[mask]

    # Group by user and take average response time
    avg_response = responses.groupby('User')['delta_min'].mean().sort_values()

    return avg_response


def add_sentiment(df, selected_user='Overall'):
    """
    Adds sentiment scores to the dataframe and returns:
    - df with sentiment
    - average sentiment per user (if Overall)
    """
    # Filter for selected user
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]

    # Remove system messages
    df = df[df['User'] != 'group_notification']

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Message'].apply(lambda m: analyzer.polarity_scores(m)['compound'])

    # Average sentiment per user if overall
    if selected_user == 'Overall':
        avg_sentiment = df.groupby('User')['sentiment'].mean().sort_values()
    else:
        avg_sentiment = None

    return df, avg_sentiment

























