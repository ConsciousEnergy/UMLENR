#CrewAI Python Script for using Multi-Agent Crews to create Simulation for the UMLENR Project

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
search_results = web_search("latest advancements in Low-Energy Nuclear Reactions 2024")
print("Web Search Results:", search_results)

# Define your agents with roles and goals
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in LENR.',
    backstory="""You are a helpful assistant that acts as a Technical Researcher who conducts in-depth exploration on topics given by a user prompt, gathering credible sources and summarizing key insights.
    You are known for your analytical skills and attention to detail.""",
    verbose=True,
    allow_delegation=False,
    tools=[]  # No external tools used for search, relying on BeautifulSoup
)

developer = Agent(
    role='Python Developer',
    goal='Create an animated Python simulation for LENR based on research insights from the Senior Research Analyst.',
    backstory="""You are an expert Python Developer, known for your skills in creating complex simulations. You will develop an animated simulation for LENR, utilizing the latest research insights.""",
    verbose=True,
    allow_delegation=True
)

validator = Agent(
    role='Technical Validation Expert',
    goal='Ensure the accuracy and reliability of the simulation and research findings.',
    backstory="""You are an expert in nuclear physics and engineering, tasked with validating the technical accuracy and reliability of the simulation and research findings.
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
    description="""Conduct a comprehensive analysis of the latest advancements in LENR in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts. Ensure to gather credible sources and references.""",
    expected_output="Full analysis report in bullet points with sources and references",
    agent=researcher
)

task3 = Task(
    description="""Develop an animated Python simulation for LENR based on the research insights provided.
    Ensure the simulation accurately represents the latest findings and technological advancements.""",
    expected_output="Python script for animated LENR simulation",
    agent=developer
)

task4 = Task(
    description="""Review and validate the Python simulation to ensure accuracy and reliability.
    Provide feedback and corrections as necessary.""",
    expected_output="Validated Python script with corrections",
    agent=validator
)

# Define a process for the crew

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[crew_manager, researcher, developer, validator],
    tasks=[task1, task2, task3, task4],
    verbose=2,  # You can set it to 1 or 2 for different logging levels
)

# Function to iterate based on user feedback
def iterate_with_feedback(file_path):
    feedback = collect_human_feedback(file_path)
    if feedback:
        print("Applying feedback to improve the simulation...")
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
    print(f"Please review the simulation at {file_path} and provide your feedback.")
    feedback = input("Enter your feedback: ")
    return feedback

# Get your crew to work!
result = crew.kickoff()

# Save the initial validated simulation to a Python file
output_directory = 'working_directory'
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, 'validated_simulation.py')

with open(output_file_path, 'w') as file:
    file.write(result)

print("######################")
print(result)
print(f"Validated simulation script saved to {output_file_path}")

# Store initial result in memory
initial_id, initial_text = store_memory("initial_simulation", result)

# Iterate with human feedback
iterate_with_feedback(output_file_path)
