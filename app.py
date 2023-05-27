import os

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import tiktoken

APIFI_ENDPOINT="https://api.apify.com/v2/acts/compass~crawler-google-places/run-sync-get-dataset-items"


@st.cache_data
def scrape_reviews(google_maps_url):
    # Define API details

    apifi_api_token = os.getenv("APIFI_API_TOKEN")
    querystring = {"token": apifi_api_token, "timeout": "120", "format": "json"}

    # Define the payload with the provided URL
    payload = {
        "exportPlaceUrls": False,
        "includeWebResults": False,
        "language": "en",
        "maxCrawledPlacesPerSearch": 1,
        "maxImages": 0,
        "maxReviews": 5000,
        "oneReviewPerRow": False,
        "onlyDataFromSearchPage": False,
        "scrapeResponseFromOwnerText": True,
        "scrapeReviewId": True,
        "scrapeReviewUrl": True,
        "scrapeReviewerId": True,
        "scrapeReviewerName": False,
        "scrapeReviewerUrl": True,
        "startUrls": [{"url": google_maps_url}],
        "reviewsSort": "newest",
        "reviewsFilterString": "",
        "searchMatching": "all",
        "allPlacesNoSearchAction": ""
    }
    headers = {"Content-Type": "application/json"}

    # Make the request to the API
    response = requests.request("GET", APIFI_ENDPOINT, json=payload, headers=headers, params=querystring)

    # Convert the response to JSON
    data = response.json()

    return data


# Title of the app
st.title("What should I eat?")

# Ask user for the Google Maps URL
google_maps_url = st.text_input('Enter the Google Maps URL', value='https://goo.gl/maps/Sp3rLZwVZe57GiuXA')
is_submitted = st.button('Submit')


# If URL is provided
@st.cache_data
def summarize(data, model="gpt-4"):
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    reviews = data['reviews'][0:50]
    reviews_text = []
    for review in reviews:
        if not review['text']:
            if not review['textTranslated']:
                continue
            else:
                reviews_text.append(review['textTranslated'])
        else:
            reviews_text.append(review['text'])

    restaurant = f"{data['title']} {data['description']}"

    prompt_template = f"""
    You are a world class food critic. You will be presented with a restaurant and a list of reviews. You must extract 
    the following information
    * The top 3 dishes at the restaurant based on the reviews. Also include the number of times each dish was mentioned, and
    include at least 5 reviews for each dish.
    * The worst 3 dishes at the restaurant based on the reviews, and include reviews that criticize the dish.
    * The best parts about the restaurant
    * The worst parts about the restaurant
    
    Make sure to respond in markdown format. You should have 4 sections. You should have bullet points under each.
    
    {restaurant}
    {reviews_text}
    """

    print(prompt_template)

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(model=model,
                                                   messages=[{"role": "user", "content": prompt_template}])

    # print the chat completion
    return chat_completion.choices[0].message.content


if google_maps_url and is_submitted:
    data = scrape_reviews(google_maps_url)
    data = data[0]

    summary = summarize(data)
    st.markdown(summary)

    st.header('Reviews Table')
    # If there are reviews in the data
    if 'reviews' in data and data['reviews']:

        # Convert reviews to a DataFrame
        reviews_df = pd.json_normalize(data['reviews'])

        # Display the table in the app
        st.dataframe(reviews_df)

    else:
        st.write('No reviews found for this location.')

    st.header('Raw JSON')
    st.json(data)
