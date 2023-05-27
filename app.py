import os
from time import sleep
from typing import List

import streamlit as st
import pandas as pd
import requests

APIFI_SYNC_ENDPOINT = "https://api.apify.com/v2/acts/compass~crawler-google-places/run-sync-get-dataset-items"

APIFI_ASYNC_ENDPOINT = (
    "https://api.apify.com/v2/acts/compass~crawler-google-places/runs"
)

DEFAULT_RESTAURANT = "https://goo.gl/maps/FHGpxBirpdJRTjbd7"

apifi_api_token = os.getenv("APIFI_API_TOKEN")
querystring = {"token": apifi_api_token, "timeout": "120", "format": "json"}

import openai


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

    for i in range(0, len(reviews_text), 25):
        status_text.write(
            f"Extracting best dishes, worst dishes, and critiques from reviews {i} to {i + 25} out of "
            f"{len(reviews_text)} of using GPT-3.5-turbo"
        )
        review_batch_text_by_newline = "\n".join(reviews_text[i : i + 25])
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
        extracted_review_from_chat = call_openai_for_review_info(model, prompt_template)
        extracted_review_info.append(extracted_review_from_chat)

    return extracted_review_info


@st.cache_data
def call_openai_for_review_info(model, prompt_template):
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt_template}]
    )
    extracted_review_from_chat = chat_completion.choices[0].message.content
    return extracted_review_from_chat


def summarize(restaurant, extracted_review_info: List[str], model="gpt-4"):
    status_text.write("Summarizing reviews with GPT-4")

    combined_extracted_review_info = ""
    for i, review_info in enumerate(extracted_review_info):
        combined_extracted_review_info += (
            f"Review summary {i} of {len(extracted_review_info)}\n"
        )
        combined_extracted_review_info += "===\n"
        combined_extracted_review_info += review_info
        combined_extracted_review_info += "\n\n"

    prompt_template = f"""
    You are a world class food critic. You will be given multiple documents containing summaries of reviews for a 
    restaurant. Each of these summaries contain the best dishes, the worst dishes, and criticisms of the restaurant.
    
    Create a final report that merges each section together. The title should be the restaurant name and should not say
    Final Report.
     
     Your final report will have 3 sections
    * The top 3 dishes, with all positive quotes and mentions of the dish from all documents
    * The worst 3 dishes, with all negative quotes and mentions of the dish from all documents
    * The criticisms of the restaurant from all documents. Pay special attention to make sure they are actually negative
    
    You must respond in markdown.

    {restaurant}
    {combined_extracted_review_info}
    """

    # create a chat completion
    summarized_message = summarize_review_infos(model, prompt_template)

    status_text.write("Summarization complete, double checking results...")

    prompt_template_critique = f"""
    You are a world class food critic. Please critique the summary of the restaurant above.
    Are the quotes for the best dishes and worst dishes accurate? Are the lists ranked correctly?
    Are the criticisms of the restaurant valid?
    If not, please correct them or remove them
    You must respond in markdown and maintain any existing formatting.
    Only respond with the updated sections. Do not add a description of your changes.

    {summarized_message}
    """

    # double check that the summary is valid
    chat_completion_critique_response = critique_summary(
        model, prompt_template_critique
    )

    # print the chat completion
    return chat_completion_critique_response


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


@st.cache_data
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
openai_api_key_input = st.empty()

# see if OPENAI_API_KEY env var is set
if "OPENAI_API_KEY" in os.environ:
    openai_api_key_input = st.text_input(
        "Enter your OpenAI API key", type="password", value=os.environ["OPENAI_API_KEY"]
    )
else:
    openai_api_key_input = st.text_input("Enter your OpenAI API key", type="password")

google_maps_url = st.text_input("Enter the Google Maps URL", value=DEFAULT_RESTAURANT)
is_submitted = st.button("Submit")
restaurant_name = st.empty()
status_heading = st.empty()
status_text = st.empty()


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
