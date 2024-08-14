import os
from dotenv import load_dotenv
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key from environment
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to search for papers on arXiv
def search_papers(keyword):
    search_url = f"https://arxiv.org/search/?query={keyword}&searchtype=all&abstracts=show&order=-announced_date_first&size=50"
    try:
        response = requests.get(search_url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')
        papers = soup.find_all('li', class_='arxiv-result')
        if not papers:
            return None, f"No papers found for keyword: {keyword}"
        
        # Extract the first paper's URL and title
        first_paper = papers[0]
        abstract_link = first_paper.find('p', class_='list-title is-inline-block')
        if abstract_link and abstract_link.find('a'):
            paper_url = abstract_link.find('a')['href']
            if not paper_url.startswith('http'):
                paper_url = "https://arxiv.org" + paper_url
            paper_title = first_paper.find('p', class_='title').text.strip()
            return paper_url, paper_title
        else:
            return None, "Abstract link not found in the search result."
    except requests.exceptions.RequestException as e:
        return None, f"An error occurred while searching for papers: {e}"

# Function to fetch paper abstract
def get_paper_abstract(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')
        abstract_block = soup.find('blockquote', class_='abstract mathjax')
        if abstract_block:
            abstract = abstract_block.text.strip()
            return abstract
        else:
            return "Abstract not found. Please check the URL or try another paper."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Function to use OpenAI GPT for text generation
def generate_text(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("LLM Task Executor")

# Example Task: Paper Talk from URL
keyword = st.text_input("Enter keyword for search (optional):", "environment")

if st.button("Search and Fetch Abstract"):
    if not keyword:
        st.error("Please enter a keyword.")
    else:
        with st.spinner("Searching for papers..."):
            paper_url, paper_title = search_papers(keyword)
            if paper_url:
                st.success(f"Found paper: {paper_title}")
                st.write(f"Paper URL: {paper_url}")  # Log the paper URL for debugging
                abstract = get_paper_abstract(paper_url)
                st.text_area("Abstract", abstract, height=200)
            else:
                st.error(paper_title)

# To keep the Streamlit app interactive and prevent it from closing
st.write("Running in interactive mode. Check terminal for output.")

# Example Task: Paper Talk from URL using OpenAI GPT
url = st.text_input("Enter paper URL (or ID):", "")
if st.button("Who should read this paper?"):
    if not url.startswith("http"):
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Processing..."):
            abstract = get_paper_abstract(url)
            if "Abstract not found" not in abstract and "An error occurred" not in abstract:
                prompt = f"Title: {url}\nAbstract: {abstract}\nWho should read this paper?"
                result = generate_text(prompt)
                st.text_area("Analysis", result, height=200)
            else:
                st.error("Failed to fetch the abstract. Please check the URL.")
