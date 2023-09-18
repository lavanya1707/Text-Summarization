import streamlit as st
import PyPDF2
from transformers import BartForConditionalGeneration, BartTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to preprocess text (remove newlines)
def preprocess_text(text):
    cleaned_text = text.replace('\n', ' ')
    return cleaned_text

# Function to summarize text using BART (abstractive)
def summarize_text_abstractive(input_text, max_length=150):
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to summarize text using LexRank (extractive)
def summarize_text_extractive(input_text, num_sentences=3):
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

# Streamlit app title
st.title('Text Summarization')

# Option to upload a PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Input text area for manual input
input_text = st.text_area('Enter or paste your text:', "")

# Button to perform text extraction and summarization
if st.button('Extract and Summarize'):
    if pdf_file is not None:
        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        # Preprocess the extracted text
        cleaned_text = preprocess_text(pdf_text)
        # Summarize the text (both extractive and abstractive)
        extractive_summary = summarize_text_extractive(cleaned_text)
        abstractive_summary = summarize_text_abstractive(cleaned_text)
        # Display both summaries
        st.subheader('Extractive Summary:')
        st.write(extractive_summary)
        st.subheader('Abstractive Summary:')
        st.write(abstractive_summary)
    elif input_text:
        # Summarize the manually entered text (both extractive and abstractive)
        extractive_summary = summarize_text_extractive(input_text)
        abstractive_summary = summarize_text_abstractive(input_text)
        # Display both summaries
        st.subheader('Extractive Summary:')
        st.write(extractive_summary)
        st.subheader('Abstractive Summary:')
        st.write(abstractive_summary)
    else:
        st.warning('Please upload a PDF file or enter text for summarization.')
