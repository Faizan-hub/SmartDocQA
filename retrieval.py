import numpy as np  # For numerical operations with arrays
from sklearn.metrics.pairwise import cosine_similarity  # For calculating cosine similarity between embeddings

def find_most_similar_page(query_embedding, documents, document_embeddings):
    """
    Find the most similar document page based on the query embedding.

    Args:
        query_embedding (np.ndarray): The embedding of the user query.
        documents (list): A list of documents containing page contents.
        document_embeddings (list): A list of document embeddings.

    Returns:
        tuple: Contains the best page information and its similarity score.
    """
    best_score = -1  # Initialize the best score
    best_page_info = None  # Placeholder for the best page info

    # Iterate through each document and its pages
    for doc_idx, document in enumerate(document_embeddings):
        for page_idx, page in enumerate(document["embeddings"]):
            page_embedding = np.array(page["embedding"]).reshape(1, -1)  # Reshape for cosine similarity
            score = cosine_similarity(query_embedding.reshape(1, -1), page_embedding)[0][0]  # Calculate score

            # Update best score and page info if the current score is higher
            if score > best_score:
                best_score = score
                content = documents[doc_idx]["pages"][page_idx]["content"]
                best_page_info = {
                    "filename": document['filename'],
                    "content": content
                }

    return best_page_info, best_score  # Return the best page info and score

def generate_answer_with_llm(client, context, query):
    """
    Generate an answer using Groq's LLM based on the provided context.

    Args:
        context (str): The context from which to generate the answer.
        query (str): The user's question.

    Returns:
        str: The generated answer or an error message.
    """
    # Construct the prompt for the LLM
    prompt = (
        f"Context:\n{context}\n\n"
        f"You are a helpful assistant that answers questions based only on the provided context. "
        f"Answer the following question:\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    try:
        # Call the Groq API using the new interface
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",  # Specify the desired model
            max_tokens=500,  # Adjust as needed for the answer length
            temperature=0.0  # Lower temperature for more deterministic responses
        )
        answer = chat_completion.choices[0].message.content.strip()  # Extract the generated answer
        return answer
    except Exception as e:
        print(f"Error while calling Groq API: {e}")  # Print the error message
        return "I'm sorry, but I couldn't generate an answer."  # Return a friendly error message
