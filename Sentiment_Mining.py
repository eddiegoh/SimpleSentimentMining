import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Run once will do if you do not have the library
nltk.download('vader_lexicon')

# read the data from csv file and store as review
review = pd.read_csv('Trip_Advisor_Reviews_MBS_Casino_Couple.csv')
print(review.info())


# Define a function to process each review and return the score of the sentiment
# compound is the aggregate result based on my interpretation range from -1(neg sentiment) to 1(positive sentiment)
# The rest are positive, negative and neutral range from 0 to 1. These 3 summed up together will give a perfect 1
def sentiment(text):
    sentiment_object = SentimentIntensityAnalyzer()
    score = sentiment_object.polarity_scores(text)
    return score


# each review will be process by the sentiment function and the score will store in sentiments_results
# it will return a list of dictionaries
sentiment_results = [sentiment(text) for text in review['review']]
print(sentiment_results)

# Convert the sentiment result into a data frame
results_df = pd.DataFrame(sentiment_results)
# merge the review and sentiment result data frame together
reviewWithSentiment = review.join(results_df)
# convert the merged data frame to csv
reviewWithSentiment.to_csv('Trip_Advisor_Reviews_MBS_Casino_Couple_with_sentiments.csv')

