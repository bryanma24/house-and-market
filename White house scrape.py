from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://www.whitehouse.gov/presidential-actions/"
url1 = "https://www.foxnews.com/category/politics/elections/presidential/trump-transition"

def Get_links_WH(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    soup = soup.find_all("h2")
    URLS = []
    for links in soup:
        for link in links.find_all('a'):
            URLS.append(link.get('href'))
    return URLS

def get_text(list):
    date_text = []
    paragraphs = []
    for link in list:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        date_div = soup.find('div', class_='wp-block-post-date')
        date_text.append(date_div.find('time').text)
        paragraphs.append([p.get_text(strip=True) for p in soup.find_all('p')])
    df = pd.DataFrame({'Date': date_text, 'Content': paragraphs})
    return df

def page(url1):
    links = []
    url_lst = [url1]
    url2 = url1 + 'page/1'
    for i in range(1, 100):
        updated_url = url2[:-1] + str(i)
        if len(updated_url) > 0:
            url_lst.append(updated_url)
        else:
            break
    for i in range(len(url_lst)):
        links.append(Get_links_WH(url_lst[i]))

    flat_links = [item for sublist in links for item in sublist]
    return get_text(flat_links)
def main():

    PRES_action_df = page('https://www.whitehouse.gov/presidential-actions/')

    Articals_df = page('https://www.whitehouse.gov/articles/')

    Brefings_df = page('https://www.whitehouse.gov/briefings-statements/')

    Facts_df = page('https://www.whitehouse.gov/fact-sheets/')

    Remaks_df = page('https://www.whitehouse.gov/remarks/')

    # Step 1: Combine them all
    combined_df = pd.concat([PRES_action_df, Articals_df, Brefings_df, Facts_df, Remaks_df], ignore_index=True)

    # Step 2: Convert the first column to datetime (assuming it's the date column)
    combined_df.iloc[:, 0] = pd.to_datetime(combined_df.iloc[:, 0], format='%B %d, %Y')

    # Step 3: Sort by the date column
    combined_df = combined_df.sort_values(by=combined_df.columns[0], ascending=False).reset_index(drop=True)

    #This was made on april 6th, at 9:20am
    combined_df.to_csv('Whitehouse_scrape_APR_6.csv', index=False)


#<div class="wp-block-post-date"><time datetime="2025-03-22T00:14:57-04:00">March 22, 2025</time></div> </div>
main()