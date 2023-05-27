import os
from time import sleep
from typing import List

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import tiktoken

APIFI_SYNC_ENDPOINT = "https://api.apify.com/v2/acts/compass~crawler-google-places/run-sync-get-dataset-items"

APIFI_ASYNC_ENDPOINT = (
    "https://api.apify.com/v2/acts/compass~crawler-google-places/runs"
)

DEFAULT_RESTAURANT='https://goo.gl/maps/sNG2jZVTD4t2QFfA7'

apifi_api_token = os.getenv("APIFI_API_TOKEN")
querystring = {"token": apifi_api_token, "timeout": "120", "format": "json"}

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def scrape_reviews(google_maps_url):
    # Define API details

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
        "allPlacesNoSearchAction": "",
    }
    headers = {"Content-Type": "application/json"}

    # Make the request to the API
    response = requests.request(
        "GET", APIFI_SYNC_ENDPOINT, json=payload, headers=headers, params=querystring
    )

    # Convert the response to JSON
    data = response.json()

    return data


# If URL is provided
@st.cache_data
def extract_review_info(data, model="gpt-3.5-turbo"):
    reviews = data["reviews"]

    reviews_text = []
    for review in reviews:
        if not review["text"]:
            if not review["textTranslated"]:
                continue
            else:
                reviews_text.append(review["textTranslated"])
        else:
            reviews_text.append(review["text"])


    extracted_review_info = []

    # create a review batch of 25 reviews, and ask OpenAI to extract information from it
    for i in range(0, len(reviews_text), 25):

        status_text.write(f"Extracting information from reviews {i} to {i+25}")
        review_batch_text_by_newline = '\n'.join(reviews_text[i:i + 25])
        prompt_template = f"""
        You are a world class food critic. You will look at a collection of restaurant reviews and extract 
        the following information from each review and place them in the correct category
        * Any dishes mentioned in a review in a positive way. Extract the dish name, the number of times it was mentioned positively, and a single quote of the most glowing review of the dish.
        * Any dishes mentioned in a negative way. Extract the dish name, the number of times it was mentioned negatively, and a single quote of the most negative review of the dish.
        * Any criticisms of the restaurant that are not related to food. Only return criticisms.
        Make sure to respond in markdown format. You should have 3 sections. "Best dishes", "Worst dishes", "Criticisms of the restaurant"
        
        I will provide an example
        
        Example
        ===
        Input: 
        The best fish soup in singapore by far!
        The line was very long
        The service was very slow
        The fish soup was very good
        The dumplings were terrible
        The dumplings were ok
        The fish soup was very tasty
        The service was mediocre but the fish soup was pretty good
        I enjoyed the tempe, but want to try the fish soup next time
        
        Output:
        Best dishes:
        * fish soup (3 mentions)
            * The best fish soup in singapore by far!
        * tempe (1 mention)
            * I enjoyed the tempe, but want to try the fish soup next time
            
        Worst dishes:
        * dumplings (2 mentions)
            * The dumplings were terrible
        
        Criticisms of the restaurant:
        * The line was very long
        * The service was very slow
        * The service was mediocre but the fish soup was pretty good
        
        ===
        Input:
        {review_batch_text_by_newline}
        """

        # create a chat completion
        chat_completion = openai.ChatCompletion.create(
            model=model, messages=[{"role": "user", "content": prompt_template}]
        )

        extracted_review_info.append(chat_completion.choices[0].message.content)

    return extracted_review_info


@st.cache_data
def summarize(restaurant, extracted_review_info: List[str], model="gpt-4"):
    status_text.write("Summarizing reviews with GPT-4")

    combined_extracted_review_info = '\n\n'.join(extracted_review_info)
    prompt_template = f"""
    You are a world class food critic. You will be given multiple documents containing summaries of reviews for a 
    restaurant. Each of these summaries contain the best dishes, the worst dishes, and criticisms of the restaurant.
    You will merge these documents into one final summary. Your final summary will have 3 sections
    * The top 3 dishes, along with how many times it was mentioned in total and the top quotes for that dish
    * The worst 3 dishes, along with how many times it was mentioned in total and the top quotes for that dish
    * The criticisms of the restaurant. Make sure that only valid criticisms are included.
    
    You must respond in markdown.

    {restaurant}
    {combined_extracted_review_info}
    """

    print(prompt_template)

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt_template}]
    )

    # print the chat completion
    return chat_completion.choices[0].message.content


@st.cache_data(allow_output_mutation=True, suppress_st_warning=True)
def start_scrape_job(google_maps_url):
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
        "allPlacesNoSearchAction": "",
    }
    headers = {"Content-Type": "application/json"}

    # Make the request to the API
    response = requests.request(
        "POST", APIFI_ASYNC_ENDPOINT, json=payload, headers=headers, params=querystring
    )

    # Convert the response to JSON
    response_json = response.json()

    return response_json["data"]["id"]


def get_run_status(act_id):
    url = f"https://api.apify.com/v2/acts/compass~crawler-google-places/runs/{act_id}"
    querystring = {"token": apifi_api_token}
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = response.json()
    status = response_json["data"]["status"]
    return status


def get_dataset_id(act_id):
    url = f"https://api.apify.com/v2/acts/compass~crawler-google-places/runs/{act_id}"
    querystring = {"token": apifi_api_token}
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = response.json()
    dataset_id = response_json["data"]["defaultDatasetId"]
    return dataset_id


def stream_logs(act_id):
    url = f"https://api.apify.com/v2/logs/{act_id}"
    response = requests.get(url, stream=True, params={"token": apifi_api_token})

    latest_status = "Retrieving reviews..."
    # Get the last line in the logs and return it if it starts with INFO
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if "INFO" in decoded_line:
                pattern = r"Extracting reviews: (\d+\/\d+)"
                import re
                match = re.search(pattern, decoded_line)
                if match:
                    # extract the review info
                    review_info = match.group(1)
                    # create and return the result string
                    latest_status = f"Fetching reviews {review_info}..."
    return latest_status


# Title of the app
st.title("What should I eat?")

# Ask user for the Google Maps URL
google_maps_url = st.text_input(
    "Enter the Google Maps URL", value="https://goo.gl/maps/Sp3rLZwVZe57GiuXA"
)
is_submitted = st.button("Submit")
heading = st.header("Status")
status_text = st.empty()


@st.cache_data
def get_dataset(dataset_id):
    # https://api.apify.com/v2/datasets/nJPUOKQf6QMapsAS4/items?token=***
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    querystring = {"token": apifi_api_token}
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = response.json()
    return response_json


if google_maps_url and is_submitted:
    status_text.write("Starting review crawling...")
    # start the scraping job
    act_id = start_scrape_job(google_maps_url)
    sleep(0.5)

    if act_id is None:
        status_text.write("Something went wrong. Please try again.")
        st.stop()

    status_text.write("Started review crawling...")

    busy_processing = True
    while busy_processing:
        # Are we done?
        status = get_run_status(act_id)

        if status == "RUNNING":
            # Get latest log status
            log_message = stream_logs(act_id)

            # Print current status to app
            status_text.write(log_message)
            continue

        if status == "FAILED" or status == "ABORTED" or status == "TIMED_OUT":
            raise Exception("Something went wrong. Please try again.")

        if status == "SUCCEEDED":
            break

    # retrieve job data once done
    status_text.write("Fetching crawled reviews...")
    dataset_id = get_dataset_id(act_id)
    dataset = get_dataset(dataset_id)
    data = dataset[0]
    status_text.write(f"Fetched {len(data['reviews'])} reviews...")

    review_info = extract_review_info(data)

    restaurant = f"{data['title']} {data['description']}"
    summary = summarize(restaurant, review_info)
    st.markdown(summary)

    st.header("Reviews Table")
    # If there are reviews in the data
    if "reviews" in data and data["reviews"]:
        # Convert reviews to a DataFrame
        reviews_df = pd.json_normalize(data["reviews"])

        # Display the table in the app
        st.dataframe(reviews_df)

    else:
        st.write("No reviews found for this location.")

    st.header("Raw JSON")
    st.json(data)
