def create_embeddings(model, documents):
    """Create embeddings for the pages of the provided documents using the specified model.

    Args:
        model: The model used to create embeddings (e.g., SentenceTransformer).
        documents (list): A list of documents, each containing pages with content.

    Returns:
        list: A list of documents with their corresponding page embeddings.
    """
    all_embeddings = []  # List to hold all embeddings for each document

    # Iterate over each document to create embeddings
    for document in documents:
        document_embeddings = []  # List to hold embeddings for the current document

        # Iterate over each page in the document
        for page in document['pages']:
            # Create an embedding for the page content using the model
            page_embedding = model.encode(page['content'])
            # Append the embedding along with page information to the document's list
            document_embeddings.append({
                "page": page['page'],
                "embedding": page_embedding.tolist()  # Convert to list for JSON serialization
            })

        # Append the document's filename and its embeddings to the all_embeddings list
        all_embeddings.append({
            "filename": document['filename'],
            "embeddings": document_embeddings
        })

    return all_embeddings  # Return the list of all document embeddings
