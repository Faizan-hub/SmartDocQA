import os  # For accessing environment variables
import streamlit as st  # For creating the web application interface
from groq import Groq  # Import the Groq library for LLM interactions
from sentence_transformers import SentenceTransformer  # For creating embeddings from sentences
from ingest_and_process import process_documents  # For processing documents from a directory
from embedding import create_embeddings  # For generating embeddings for the processed documents
from retrieval import find_most_similar_page, generate_answer_with_llm  # For retrieving relevant documents and generating answers


# Load the SentenceTransformer model for embeddings
@st.cache_resource  # Cache the model to avoid reloading on each run
def load_models():
    """
    Load and cache the SentenceTransformer model for generating embeddings.

    Returns:
        tuple: A tuple containing the embedding model and the Groq client.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the SentenceTransformer model

    key = os.getenv("GROQ_API_KEY")  # Retrieve the Groq API key from environment variables

    # Initialize Groq client with the API key
    client = Groq(api_key=key)
    return embedding_model, client  # Return both the model and the client


# Load documents and their embeddings from a specified directory
@st.cache_data  # Cache the data to avoid reprocessing
def load_data(directory, _embedding_model):
    """
    Load documents from a given directory and create their embeddings.

    Args:
        directory (str): The path to the directory containing documents.
        _embedding_model (SentenceTransformer): The model used to create embeddings.

    Returns:
        tuple: A tuple containing the list of documents and their corresponding embeddings.
    """
    documents = process_documents(directory)  # Process documents from the specified directory
    document_embeddings = create_embeddings(_embedding_model, documents)  # Create embeddings for the documents
    return documents, document_embeddings  # Return both documents and their embeddings


# Streamlit user interface setup
def main():
    """
    Main function to set up the Streamlit application UI and handle user interaction.
    """
    # Load models and data
    embedding_model, client = load_models()  # Load the embedding model and Groq client
    documents, document_embeddings = load_data("data", embedding_model)  # Load documents and their embeddings

    # Set up Streamlit interface
    st.title("Document Query Answering System")  # Application title
    st.write("Enter your question about the documents:")  # Instruction for the user

    user_query = st.text_input("Your Question:")  # Text input for user queries

    if st.button("Submit"):  # Button to submit the user query
        if user_query:  # Check if the user has entered a question
            # Create embedding for the user query
            query_embedding = embedding_model.encode(user_query).reshape(1, -1)
            # Find the most similar document to the user query
            best_page_info, score = find_most_similar_page(query_embedding, documents, document_embeddings)

            if best_page_info:  # If a relevant document is found
                context = best_page_info['content']  # Get the content of the best matching document
                answer = generate_answer_with_llm(client, context, user_query)  # Generate an answer using LLM
                st.subheader("Results:")  # Section title for results
                st.write(f"**Most relevant document:** {best_page_info['filename']}")  # Display document filename
                st.write(f"**Answer:** {answer}")  # Display the generated answer
            else:
                st.write("No relevant document found.")  # Handle case where no relevant document is found
        else:
            st.write("Please enter a question.")  # Prompt to enter a question if the input is empty


# Run the Streamlit app
if __name__ == "__main__":
    main()  # Execute the main function to run the app
