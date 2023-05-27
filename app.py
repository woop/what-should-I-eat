import os
import uuid
from time import sleep
from typing import List

import pandas as pd
import requests
import streamlit as st
import tiktoken

APIFI_SYNC_ENDPOINT = "https://api.apify.com/v2/acts/compass~crawler-google-places/run-sync-get-dataset-items"

APIFI_ASYNC_ENDPOINT = (
    "https://api.apify.com/v2/acts/compass~crawler-google-places/runs"
)

DEFAULT_RESTAURANT = "https://goo.gl/maps/FHGpxBirpdJRTjbd7"

apifi_api_token = os.getenv("APIFI_API_TOKEN")
querystring = {"token": apifi_api_token, "timeout": "120", "format": "json"}

import openai


@st.cache_data
def find_useful_reviews(review_batch, model):
    prompt_template = f"""
You are going to be given a list of reviews. Extract all reviews that are positive about a specific dish,
negative about a specific dish, or criticize the restaurant. Return the reviews in three markdown lists. I will provide
an example

Example
===
Input
- Best location
- Hard to find parking
- Best fish soup super tasty
- Best fish soup
- The dumplings suck
- The fish soup is great
- The dumplings are terrible
- The location is great
- Slow service

Output

# Positive
- Best fish soup super tasty
- The fish soup is great
- Best fish soup

# Negative
- The dumplings suck
- The dumplings are terrible

# Criticisms
- Hard to find parking
- Slow service

===
Input:
{review_batch}

Output:"""

    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt_template}]
    )
    useful_reviews = chat_completion.choices[0].message.content
    return useful_reviews


def extract_review_info(data, model="gpt-4"):
    reviews = data["reviews"]

    reviews_text = []
    for review in reviews:
        if not review["textTranslated"]:
            if not review["text"]:
                continue
            else:
                reviews_text.append(review["text"])
        else:
            reviews_text.append(review["textTranslated"])

    review_batch = ""
    reviews_text_pointer = 0
    # all_useful_reviews = ''
    while True:
        enc = tiktoken.encoding_for_model(model)
        current_encoding_len = len(enc.encode(review_batch))
        additional_encoding_len = len(enc.encode(reviews_text[reviews_text_pointer]))
        combined_encoding_len = current_encoding_len + additional_encoding_len
        if combined_encoding_len > 5500:
            return review_batch
        else:
            review_batch += '\n\n- ' + reviews_text[reviews_text_pointer]
            reviews_text_pointer += 1
            if reviews_text_pointer >= len(reviews_text):
                return review_batch

@st.cache_data
def call_openai_for_review_info(model, prompt_template):
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt_template}]
    )
    extracted_review_from_chat = chat_completion.choices[0].message.content
    return extracted_review_from_chat


def summarize(restaurant, useful_reviews: str, model="gpt-4"):
    status_text.write("Summarizing reviews with GPT-4")

    prompt_template = f"""===
{restaurant}
===
{useful_reviews}
===
You are a world class food critic. Above are a list of reviews from users for a restaurant.

Create a markdown summary of the restaurant. It has 3 headings. "Best dishes", "Worst dishes", and "Criticisms".
Under the best and worst dishes categories, list the top 3 dishes and the bottom 3 dishes. Add all quotes related to
those dishes underneath those dishes. Also add the amount of mentions of that dish. Under criticisms, group all
criticisms together into categories and list the related quotes under each category. Do not return anything else."""

    # create a chat completion
    summarized_message = summarize_review_infos(model, prompt_template)
    return summarized_message


@st.cache_data
def critique_summary(model, prompt_template_critique):
    chat_completion_critique = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt_template_critique}]
    )
    chat_completion_critique_response = chat_completion_critique.choices[
        0
    ].message.content
    return chat_completion_critique_response


@st.cache_data
def summarize_review_infos(model, prompt_template):
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt_template}]
    )
    summarized_message = chat_completion.choices[0].message.content
    return summarized_message


@st.cache_data()
def start_scrape_job(google_maps_url, bust_cache_string=None):
    if bust_cache_string:
        print("busting cache with " + bust_cache_string)
    # Define the payload with the provided URL
    payload = {
        "exportPlaceUrls": False,
        "includeWebResults": False,
        "language": "en",
        "maxCrawledPlacesPerSearch": 1,
        "maxImages": 0,
        "maxReviews": 200,
        "oneReviewPerRow": False,
        "onlyDataFromSearchPage": False,
        "scrapeResponseFromOwnerText": False,
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
st.text("A tool to help you find the best (and worst) dishes at a restaurant")

# Ask user for the Google Maps URL
openai_api_key_input = st.empty()

# see if OPENAI_API_KEY env var is set
if "OPENAI_API_KEY" in os.environ:
    openai_api_key_input = st.text_input(
        "Enter your OpenAI API key", type="password", value=os.environ["OPENAI_API_KEY"]
    )
else:
    openai_api_key_input = st.text_input("Enter your OpenAI API key", type="password")

openai.api_key = openai_api_key_input

google_maps_url = st.text_input("Enter the Google Maps URL", value=DEFAULT_RESTAURANT)
is_submitted = st.button("Submit")
restaurant_name = st.empty()
status_heading = st.empty()
status_subheading = st.empty()
status_text = st.empty()


@st.cache_data
def get_previous_runs():
    url = f"https://api.apify.com/v2/actor-runs"

    querystring = {"token": apifi_api_token}
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = response.json()
    if "data" in response_json:
        return response_json["data"]["items"]
    else:
        return []


@st.cache_data
def get_dataset(dataset_id):
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    querystring = {"token": apifi_api_token}
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = response.json()
    return response_json


if google_maps_url and is_submitted:
    status_heading.header("Status")
    status_subheading.text("Please note that it takes about 5 minutes to compute results!")

    status_text.write("Starting review crawling...")
    # start the scraping job
    act_id = start_scrape_job(google_maps_url)
    sleep(1)

    status = get_run_status(act_id)
    if status == "FAILED" or status == "ABORTED" or status == "TIMED_OUT":
        bust_cache_random_string = str(uuid.uuid4())
        act_id = start_scrape_job(google_maps_url, bust_cache_random_string)
        sleep(1)

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
    restaurant_description = data["description"]
    restaurant_title = data["title"]
    if restaurant_description:
        restaurant_title = restaurant_title + " - " + restaurant_description
    restaurant_name = st.header(restaurant_title)

    status_text.write(f"Fetched {len(data['reviews'])} reviews...")

    review_info = extract_review_info(data)

    summary = summarize(restaurant_title, review_info)
    status_text.write(f"Done! ({len(data['reviews'])} reviews)")
    restaurant_name = st.empty()
    st.markdown(summary)

    st.header("Reviews Table")
    # If there are reviews in the data
    if "reviews" in data and data["reviews"]:
        # Convert reviews to a DataFrame
        reviews_df = pd.json_normalize(data["reviews"])

        # Remove rows in the "text" column that are "None"
        reviews_df = reviews_df[reviews_df["text"].notna()]

        # Display the table in the app
        st.dataframe(reviews_df)

    else:
        st.write("No reviews found for this location.")

    st.header("Raw JSON")
    st.json(data)
