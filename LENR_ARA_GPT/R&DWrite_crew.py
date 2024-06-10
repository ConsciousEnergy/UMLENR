import os
import dotenv
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
use_openai = True
#os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
#use_groq = True

os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3'  # Adjust based on available model
use_local_model = True

# Initialize FAISS index
dimension = 768  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Function to generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Function to store data in FAISS
def store_memory(id, text):
    embeddings = generate_embeddings(text)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return id, text  # Return a tuple for tracking

# Function to retrieve data from FAISS
def retrieve_memory(query, top_k=5):
    embeddings = generate_embeddings(query)
    faiss.normalize_L2(embeddings)
    distances, indices = index.search(embeddings, top_k)
    return indices

# Define a function for web scraping using BeautifulSoup
def web_search(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        results.append(g.get_text())
    return results

# Example usage of web_search function
search_results = web_search("latest advancements in Condensed Matter Nuclear Fusion 2024")
print("Web Search Results:", search_results)

# Define your agents with roles and goals
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in condensed matter nuclear fusion and LENR.',
    backstory="""You are a helpful assistant that acts as Technical Researcher who conducts in-depth exploration on topics given by a user prompt, gathering credible sources and summarizing key insights.
    You are known for your analytical skills and attention to detail.""",
    verbose=True,
    allow_delegation=False,
    tools=[]  # No external tools used for search, relying on BeautifulSoup
)

writer = Agent(
    role='Science and Technology Content Strategist',
    goal='Craft compelling content on condensed matter nuclear fusion and LENR advancements based on research insights from our Senior Research Analyst.',
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives, ensuring all statements are supported by citations and references.""",
    verbose=True,
    allow_delegation=True
)

seo_editor = Agent(
    role='SEO and Readability Expert',
    goal='Enhance the article for readability and SEO, ensuring the content is engaging and optimized for search engines.',
    backstory="""You are an expert in SEO and content readability. You ensure that the article meets best practices for SEO, readability, and overall engagement. You also ensure the article has at least 6 paragraphs and 800+ words, and includes Title, Metadescription, resource/citation links, and Keywords with comma separation.""",
    verbose=True,
    allow_delegation=True
)

validator = Agent(
    role='Technical Validation Expert',
    goal='Ensure the accuracy and reliability of the research findings and written content on condensed matter nuclear fusion and LENR.',
    backstory="""You are an expert in nuclear physics and engineering, tasked with validating the technical accuracy and reliability of the research findings and written content.
    You ensure that all information is correct, well-cited, and based on credible sources.""",
    verbose=True,
    allow_delegation=False
)

crew_manager = Agent(
    role='Crew Manager',
    goal='Ensure all tasks are completed and facilitate communication between the user and the agent chain.',
    backstory="""You are the Crew Manager, responsible for overseeing the workflow, ensuring that all tasks are completed on time, and coordinating communication between the user and the agents. You handle any issues that arise during the process and ensure smooth operation.""",
    verbose=True,
    allow_delegation=False
)

# Create tasks for your agents

# Task for the crew manager to ensure all tasks are completed and handle iterations
task1 = Task(
    description="""Oversee the workflow to ensure all tasks are completed. Facilitate communication between the user and the agents.
    Handle any issues, such as agents getting stuck in a loop, and ensure smooth operation of the process. Manage iterations based on user feedback.""",
    expected_output="Coordination report and final iteration document",
    agent=crew_manager
)

task2 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in condensed matter nuclear fusion and LENR in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts. Ensure to gather credible sources and references.""",
    expected_output="Full analysis report in bullet points with sources and references",
    agent=researcher
)

task3 = Task(
    description="""Using the insights provided, develop an engaging technical paper that highlights the most significant advancements in condensed matter nuclear fusion and LENR.
    Your paper should be informative yet accessible, catering to a scientific audience. Ensure to include all relevant citations and references.""",
    expected_output="Full technical paper of at least 8 pages with citations and references",
    agent=writer
)

task4 = Task(
    description="""Review and edit the technical paper for readability and SEO best practices. Ensure the content is engaging and optimized for search engines.
    The final article should have at least 6 paragraphs and 800+ words, and include Title, Metadescription, resource/citation links, and Keywords with comma separation.""",
    expected_output="SEO optimized and readable technical paper with Title, Metadescription, resource/citation links, and Keywords",
    agent=seo_editor
)

task5 = Task(
    description="""Review and validate the technical paper to ensure accuracy and reliability of the information presented. Check that all citations and references are correct and complete.
    Provide feedback and corrections as necessary.""",
    expected_output="Validated technical paper with corrections and complete citations",
    agent=validator
)

# Define a process for the crew

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[crew_manager, researcher, writer, seo_editor, validator],
    tasks=[task1, task2, task3, task4, task5],
    verbose=2,  # You can set it to 1 or 2 for different logging levels
)

# Function to iterate based on user feedback
def iterate_with_feedback(file_path):
    feedback = collect_human_feedback(file_path)
    if feedback:
        print("Applying feedback to improve the document...")
        # Store feedback in memory
        feedback_id, feedback_text = store_memory("user_feedback", feedback)
        # Here, you would re-run the necessary agents/tasks based on the feedback
        # For demonstration purposes, we just print the feedback
        print(feedback)
        # Save the feedback to the file for the next iteration
        with open(file_path, 'a') as file:
            file.write("\n\n# Feedback received:\n")
            file.write(feedback)

# Placeholder for human feedback loop
def collect_human_feedback(file_path):
    print(f"Please review the technical paper at {file_path} and provide your feedback.")
    feedback = input("Enter your feedback: ")
    return feedback

# Get your crew to work!
result = crew.kickoff()

# Save the initial validated technical paper to a text file
output_directory = 'working_directory'
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, 'validated_technical_paper.txt')

with open(output_file_path, 'w') as file:
    file.write(result)

print("######################")
print(result)
print(f"Validated technical paper saved to {output_file_path}")

# Store initial result in memory
initial_id, initial_text = store_memory("initial_paper", result)

# Iterate with human feedback
iterate_with_feedback(output_file_path)
