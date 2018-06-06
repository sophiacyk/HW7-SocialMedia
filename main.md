
**Observaions: **
    1. All the 5 News channels are emotinally neutral. Only BBS is slightly higher. 
    2. Compared to other News channels, NY Times tweets less frequently. 
    3. All the 5 News Channels tend to tweet positive and negative tweets alternatively. There were no group of positive or negative tweets at the short period of time.  
    


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

```


```python
# collect 100 tweets from each channel 

target_media = ("@BBC", "@CNN", "@CBSNews",
                "@FoxNews", "@nytimes")

results_list = []

username = []
date = []
com_list = []
pos_list = []
neu_list = []
neg_list = []
count = []

# user_tweets = {
#     "Username": username,
#     "Date": date, 
#     "Compound Score": com_list,
#     "Postive Score": pos_list,
#     "Neutral Score": neu_list,
#     "Negative Score": neg_list,
#     "Tweets Ago": count
# }


for media in target_media:
    
    public_tweets = api.user_timeline(media, count=100)
    
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    counter_list = []
    date_list = []
    
    counter = 1

    
    for tweet in public_tweets:
        
        results = analyzer.polarity_scores(tweet["text"])
        
        compound = results["compound"]
        pos = results["pos"]
        neu = results["neu"]
        neg = results["neg"]

        # Add each value to the appropriate list
        compound_list.append(compound)
        positive_list.append(pos)
        negative_list.append(neg)
        neutral_list.append(neu)
        counter_list.append(counter)
        date_list.append(tweet['created_at'])
      
        counter += 1
        
    username.extend([media]*100),
    date.extend(date_list), 
    com_list.extend(compound_list),
    pos_list.extend(positive_list),
    neu_list.extend(neutral_list),
    neg_list.extend(negative_list),
    count.extend(counter_list)    
    
  
 
    
    user_results = {
        "Username": media,
        "Compound Score": np.mean(compound_list),
        "Postive Score": np.mean(positive_list),
        "Neutral Score": np.mean(neutral_list),
        "Negative Score": np.mean(negative_list),
    }
    
    

    results_list.append(user_results)



```


```python
user_tweets = {
    "Username": username,
    "Date": date, 
    "Compound Score": com_list,
    "Postive Score": pos_list,
    "Neutral Score": neu_list,
    "Negative Score": neg_list,
    "Tweets Ago": count
}


```


```python
results_df = pd.DataFrame(results_list).set_index("Username").round(3)
results_df.head()

user_tweets = pd.DataFrame(user_tweets).set_index("Username")


```


```python
user_tweets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound Score</th>
      <th>Date</th>
      <th>Negative Score</th>
      <th>Neutral Score</th>
      <th>Postive Score</th>
      <th>Tweets Ago</th>
    </tr>
    <tr>
      <th>Username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>0.1779</td>
      <td>Wed Jun 06 18:03:04 +0000 2018</td>
      <td>0.122</td>
      <td>0.718</td>
      <td>0.160</td>
      <td>1</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>0.0000</td>
      <td>Wed Jun 06 17:02:09 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>0.0000</td>
      <td>Wed Jun 06 16:00:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>0.3182</td>
      <td>Wed Jun 06 13:02:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.892</td>
      <td>0.108</td>
      <td>4</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>-0.5574</td>
      <td>Wed Jun 06 12:21:20 +0000 2018</td>
      <td>0.153</td>
      <td>0.847</td>
      <td>0.000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot of sentiments
fig, ax = plt.subplots(1, 5, figsize=(35,6), sharex='col', sharey='row') 
count = 1

for media in target_media:
    tweet_df = user_tweets.loc[media,:]
    # Create plot
    x_vals = tweet_df["Tweets Ago"]
    y_vals = tweet_df["Compound Score"]
    
    plt.subplot(1, 5, count)
    plt.plot(x_vals,
         y_vals, marker="o", linewidth=0.5,
         alpha=0.8)
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M")
    plt.title(f"Sentiment Analysis of Tweets for {media}")
    plt.xlim([x_vals.max(),x_vals.min()])
    plt.ylim(-1,1)
    plt.ylabel("Tweet Polarity")
    plt.xlabel(f"Tweets Ago {now}")
    count += 1
 

plt.savefig("TweetSentiments.PNG")

```


![png](main_files/main_6_0.png)



```python
# bar chart of overall scores 


sns.set(style="white", context="talk")
f, ax2 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.barplot(x=results_df.index, y=results_df['Compound Score'], palette="Set3", ax=ax2)
ax2.set_ylabel("Compound Score")
ax2.set_ylim(-0.4,0.4)
ax2.set_title(f"Overall Sentiment Score {now}")
plt.savefig("OverallSentiments.PNG")
```


![png](main_files/main_7_0.png)



```python
pd.DataFrame.to_csv(user_tweets, "100tweets.csv", sep=',')
```


```python
!ipython nbconvert --to markdown main.ipynb
```
!rename main.ipynb README.md