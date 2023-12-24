from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough

import dotenv
import os
import openai
import time
from openai import OpenAI
import warnings


# Suppress the specific warning related to relevance scores
warnings.filterwarnings("ignore", category=UserWarning, message="No relevant docs were retrieved using the relevance score threshold 0.65")

# load api key
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
# defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# if you saved the key under a different environment variable name, you can do something like:
# client = OpenAI(
#   api_key=os.environ.get("CUSTOM_ENV_NAME"),
# )
# DEPRECATED BECAUSE OPENAI DONT SUPPORT EPUBS
def loadEPub(filepath):
    # load all .epub files from the input folder
    documents = []
    files = os.listdir(filepath)
    for file_name in files:
        if file_name.endswith(".epub"):
            loader = UnstructuredEPubLoader(filepath + "/" + file_name)
            loaded = loader.load()
            for item in loaded:
                documents.append(item)
    
    # split the input
    text_splitter = CharacterTextSplitter(chunk_size=15000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    print("Split documents into " + str(len(texts)) + " chunks.")
    return texts


def loadPDF(filepath):
    #Load PDF using pypdf into array of documents, 
    # where each document contains the page content and metadata with page number.
    documents = []
    for file_name in os.listdir(filepath):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(filepath + "/" + file_name)
            pages = loader.load_and_split()
            #documents.append(pages)
            for page in pages:
                documents.append(page)

    return documents


def loadTXT(filepath):
    # load the document and split it into chunks
    documents = []
    for file_name in os.listdir(filepath):
        if file_name.endswith(".txt"):
            loader = TextLoader(filepath + "/" + file_name)
            loaded = loader.load()
            for item in loaded:
                documents.append(item)

#     # split it into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     docs = text_splitter.split_documents(documents)
#     return docs


# # create the vector storage
# embeddings = OpenAIEmbeddings()
# texts = loadEPub("inputASD")
# #texts = loadEPub("test_input")
# #texts = loadPDF("test_input")   
# vectorstore = Chroma.from_documents(texts, embeddings)
# retriever = vectorstore.as_retriever()

# rag_prompt = hub.pull("rlm/rag-prompt")

# template = '''
# Answer the question based only on the following context:
# {context}
# '''

# # create the conversation memory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Use GPT-3.5 Turbo for RAG
# gpt = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

# # Define the RAG chain
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()} 
#     | rag_prompt 
#     | gpt  # Use the appropriate GPT model here
# )

# # Define your similarity threshold (adjust as needed)
# THRESHOLD = 0.65

# print("Welcome to the ASD Bot. Ask me a question about ASD. Type 'exit' to quit.")

# List all files in the "inputASD" directory
directory = "inputASD"
file_ids = []
for filename in os.listdir(directory):
    if filename.endswith(".pdf" or ".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "rb") as file:
            uploaded_file = client.files.create(
                file=file,
                purpose='assistants'
            )
            file_ids.append(uploaded_file.id)


# Add the file to the assistant
assistant = client.beta.assistants.create(
  instructions='''
  Your task is to assist users in understanding different management strategies for Autism. 
  You should rely exclusively on the specific research files and documents provided to you. 
  Do not use any information from your training or external sources. 
  Answer questions based solely on the contents of these provided files. 
  If a question arises that cannot be answered with the information in these files, respond with 'I don't know.' 
  Your goal is to ensure that all information given is accurate, research-based, and directly sourced from these documents. 
  Avoid speculation or guessing. Focus on delivering factual, proven methods as indicated by the uploaded information. 
  Please confirm your understanding of these instructions.''',
  model="gpt-4-1106-preview",
  tools=[{"type": "retrieval"}],
  file_ids=file_ids  # Use the list of uploaded file IDs
)

# Your specific assistant ID
assistant_id = 'asst_c1JUnCuRh0pljGz1X2WAtS25'

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


# Pretty printing helper
def pretty_print(messages):
    #print("# Messages")
    # skip the first message, which is the user's question
    for m in messages:
        if m.role == "user":
            continue
        else:
            print(f"{m.role}: {m.content[0].text.value}\n")
    print()


print("Welcome to the ASD Bot. Ask me a question about ASD. Type 'exit' to quit.\n")
thread = client.beta.threads.create()
# Start a loop to continuously prompt the user for input
while True:
    user_input = input("Ask the assistant: ")
    print()

    # Check if the user wants to exit the conversation
    if user_input.lower() in ['exit', 'quit']:
        break

    # Create a message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )
    
    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
        # this will override the default prompt instructions we have given
        #instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    # Step 5: Check the Run status
    # By default, a Run goes into the queued state. You can periodically retrieve the Run to check on its status to see if it has moved to completed.
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )

    # Wait for completion
    wait_on_run(run, thread)

    # Retrieve all the messages added after our last user message
    messages = client.beta.threads.messages.list(
        thread_id=thread.id, 
        order="asc", 
        after=message.id
    )

    # Pretty print the messages
    pretty_print(get_response(thread))


# while True:
#     query = input("> ")
#     if query == "exit":
#         break
#     if query.strip() == "":
#         continue

#     # Call the function and store the results in a variable
#     # returns 3 documents that are at least 0.65 similar to the query
#     results = vectorstore.similarity_search_with_relevance_scores(query, k=3, score_threshold=THRESHOLD)

#     # Print out the relevance scores for all documents
#     # for document, score in results:
#     #     print("Relevance Score:", score)

#     # Check if relevant documents were found
#     if results:
#         # use rag_chain to search vector store for context
#         answer = rag_chain.invoke(query)
#         print("\nANSWER:", answer.content)
#         print()

#     else:
#         # If no relevant documents were found, indicate that the query is not in the dataset

#         print("\nAnswer: That information is not in my dataset.\n")
#         print("this line is just for testing purposes")
