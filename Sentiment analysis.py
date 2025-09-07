import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
pd.set_option('future.no_silent_downcasting', True)

FILENAME_stock = 'Stock Data APR_6'
FILENAME_WHITE_HOUSE = 'Whitehouse_scrape_APR_6.csv'

scaler = MinMaxScaler()

def add_sentiment_to_df(df):
    sent = []
    for i in range(len(df)):
        Blob = TextBlob(df.loc[i, 'Content'])
        polarity = Blob.sentiment.polarity
        sent.append(polarity)
    df['Sentiment'] = sent
    return df


def sentiment_list(df):
    dct = {}
    for i in range(len(df)):
        date = df.loc[i, 'Date']
        sentiment = df.loc[i, 'Sentiment']
        if date in dct:
            dct[date].append(sentiment)
        else:
            dct[date] = [sentiment]
    sent_dct = {}
    for key, value in dct.items():
        sent_dct[key] = ((statistics.mean(value) + 1) / 2) # normalizing
    return sent_dct

def keep_same(df1, dct):
    new_lst = []
    date_lst = []
    for i in range(len(df1)):
        # Format the timestamp as string in a specific format
        date_lst.append(str(df1.loc[i + 1, 'Date']))  # or another format
    for key, value in dct.items():
        date = str(key)
        if date in date_lst:
            new_lst.append(value)
    return new_lst

def difference(lst):
    dif_lst = []
    for i in range(2, len(lst)):
        dif = lst[i - 1] - lst[i]
        if dif < 0:
            dif_lst.append(1)
        else:
            dif_lst.append(0)
    return dif_lst
def main():

    # Reading in the data
    WHITE_HOUSE_data = pd.read_csv(FILENAME_WHITE_HOUSE)
    Stock_data = pd.read_csv(FILENAME_stock)

    # Cleaning the data
    Stock_data = Stock_data.drop(Stock_data.index[0])

    # Normalizing
    Stock_data['normalize'] = scaler.fit_transform(Stock_data[['Open']])
    # turning from a string to a float
    Stock_data.loc[:, 'normalize'] = Stock_data.loc[:, 'normalize'].astype(float)
    Stock_data.loc[:, 'Open'] = Stock_data.loc[:, 'Open'].astype(float)

    # Turning the dates in datetime data types
    WHITE_HOUSE_data['Date'] = pd.to_datetime(WHITE_HOUSE_data['Date'])
    Stock_data['Date'] = pd.to_datetime(Stock_data['Date'])

    #Adding a sentiment column to df
    WHITE_HOUSE_data = add_sentiment_to_df(WHITE_HOUSE_data)

    # Getting the sentiment list. Join same dates, then normalize
    sent_dct = sentiment_list(WHITE_HOUSE_data)

    #Keeping the same dates
    Same_date_Sent = keep_same(Stock_data, sent_dct)
    Same_date_Sent = np.array(Same_date_Sent).reshape(-1, 1)
    Same_date_Sent = scaler.fit_transform(Same_date_Sent)
    Same_date_Sent = Same_date_Sent.flatten().tolist()

    #Creating the df
    df = pd.DataFrame({
        'Date': Stock_data['Date'],
        'stock_price': Stock_data['Open'],
        'stock_n': Stock_data['normalize'],
        'sentiment_n': Same_date_Sent})


    df["Daily Return"] = ((df['stock_price'].diff()))

    df["Sent change"] = (df['sentiment_n'].diff())
    df.replace([np.inf, -np.inf], 0.207244, inplace=True)



    df['Daily Return'] = (scaler.fit_transform(df[['Daily Return']]) - 0.5)
    #df['Sent change'] = scaler.fit_transform(df[['Sent change']])




    #print(df)
    plt.plot(df['stock_n'], label = "Stock price")
    plt.plot(df['sentiment_n'], label = "Sentiment")
    plt.xticks([0, 26, 53], ["2025-01-21", "2025-02-26", "2025-04-04"])
    plt.title('Normalized stock vs Normalized Sentiment Polarity value')
    plt.legend()
    plt.show()





    # fix this plot
    plt.plot(df['Daily Return'], label = 'Daily Return')
    plt.plot(df['Sent change'], label = 'Sentiment change')
    plt.xticks([1, 26, 53], ["2025-01-22", "2025-02-26", "2025-04-04"])
    plt.title("Daily change of stock and Sentiment polarity value Normalized")
    plt.legend()
    plt.show()

    # finding the correlation    FIX
    d1 = difference(df['Daily Return'])
    d2 = difference(df['Sent change'])


    corr = []
    for j in range(9):
        count = 0
        for i in range(len(d1)):
            if d1[i - j] == d2[i]:
                count += 1
        corr.append(count / (len(d1) - j))
    print(corr)
    plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8], corr)
    plt.title('Correlation between Sentiment Change and Stock Returns')
    plt.ylabel('Percent same')
    plt.xlabel('Lagging factor')
    plt.tight_layout()
    plt.show()

    vectorizer = TfidfVectorizer(stop_words="english")
    vec_catch = vectorizer.fit_transform(WHITE_HOUSE_data['Content'])

    # Reduce dimensions
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vec_catch.toarray())

    # Assign PCA results back to WHITE_HOUSE_data
    WHITE_HOUSE_data[["pca1", "pca2"]] = reduced
    print(WHITE_HOUSE_data)


    # Run KMeans
    K = 2
    km = KMeans(n_clusters=K)
    km.fit(WHITE_HOUSE_data[["pca1", "pca2"]])

    # Plot
    sns.scatterplot(
        x="pca1", y="pca2",
        data=WHITE_HOUSE_data,
        hue=km.labels_,
        palette="plasma", s=150
    )
    plt.title("KMeans Clustering")
    plt.show()

    # Create a new scatter plot colored by stock movement
    # Create a mapping of dates to stock movements
    stock_movement_dict = dict(zip(df['Date'], df['stock_price'].diff() > 0))
    
    # Map each press release date to its corresponding stock movement
    WHITE_HOUSE_data['Stock_Movement'] = WHITE_HOUSE_data['Date'].map(stock_movement_dict)
    
    # Filter out rows where Stock_Movement is NaN
    valid_data = WHITE_HOUSE_data.dropna(subset=['Stock_Movement'])
    
    # Create separate scatter plots for up and down movements
    up_data = valid_data[valid_data['Stock_Movement'] == True]
    down_data = valid_data[valid_data['Stock_Movement'] == False]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(
        up_data["pca1"], 
        up_data["pca2"],
        c='green',
        label='Stock Up',
        s=150,

    )
    plt.scatter(
        down_data["pca1"],
        down_data["pca2"],
        c='red',
        label='Stock Down',
        s=150,

    )
    plt.title("K-Means Features Colored by Stock Movement")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

    print(df)
main()



