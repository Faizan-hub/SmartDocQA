import os  # For handling directory and file operations
from pdf2image import convert_from_path  # For converting PDF pages to images
from pypdf import PdfReader  # For reading PDF files
import pytesseract  # For performing Optical Character Recognition (OCR)
from concurrent.futures import ThreadPoolExecutor  # For parallel processing


def process_page_for_ocr(img):
    """Helper function to perform OCR on a single image.

    Args:
        img: The image on which to perform OCR.

    Returns:
        str: Extracted text from the image.
    """
    return pytesseract.image_to_string(img)  # Perform OCR and return the extracted text


def process_documents(directory):
    """Process all PDF documents in a given directory, extracting text from each page.

    Args:
        directory (str): The path to the directory containing PDF files.

    Returns:
        list: A list of documents, each containing pages and their extracted content.
    """
    all_documents = []  # List to hold all documents and their pages

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):  # Process only PDF files
            print(f"Processing: {filename}")
            reader = PdfReader(os.path.join(directory, filename))  # Read the PDF file
            images_to_process = []  # List to hold indices of pages requiring OCR
            document_pages = []  # List to hold pages of the current document

            # Process each page in the PDF
            for i, page in enumerate(reader.pages):
                # Try extracting text first
                page_text = page.extract_text()
                if page_text:
                    # Append page content to the document_pages list
                    document_pages.append({"page": i + 1, "content": page_text.strip()})
                else:
                    # If no text is found, add the page index for OCR later
                    images_to_process.append(i)

            # Convert the pages that need OCR to images
            if images_to_process:
                print("Converting pages to images for OCR...")
                images = convert_from_path(os.path.join(directory, filename), dpi=150)  # Convert pages to images

                # Use ThreadPoolExecutor for parallel OCR processing
                with ThreadPoolExecutor() as executor:
                    ocr_results = executor.map(process_page_for_ocr, (images[i] for i in images_to_process))
                    for i, ocr_text in enumerate(ocr_results):
                        # Append OCR results to the document_pages list
                        document_pages.append({"page": images_to_process[i] + 1, "content": ocr_text.strip()})
                        print(f"OCR processed for page {images_to_process[i] + 1}.")

            # Add the processed document's pages to the all_documents list
            all_documents.append({"filename": filename, "pages": document_pages})

    # Return all documents as a structured list
    return all_documents
