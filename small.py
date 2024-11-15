from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from textstat import text_standard, lexicon_count

load_dotenv()  


os.environ["GOOGLE_API_KEY"] = "AIzaSyBzT-hxkBkifqwEzuw5gHM4MwCcub1h1-8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


model_options = {
    "Gemini Pro": "gemini-pro"}

def load_model(model_name):
    if model_name == "gemini-pro":
        return genai.GenerativeModel(model_options[model_name]).start_chat(history=[])


st.set_page_config(page_title="Gemini Chatbot Demo")
st.header("Chatbot Application")

selected_model_name = st.selectbox("Choose a model:", options=list(model_options.keys()))
selected_model = load_model(selected_model_name)


user_input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

def get_response(question, model_name):
    if model_name == "gemini-pro":
        chat = selected_model  
        response = chat.send_message(question, stream=True)
        return "".join([chunk.text for chunk in response])


response_text = ""
if submit:
    response_text = get_response(user_input, selected_model_name)
    st.subheader("The Response is:")
    st.write(response_text)

def check_quality(response_text):
    response_length = len(response_text)
    word_count = lexicon_count(response_text, removepunct=True)
    reading_level = text_standard(response_text, float_output=False)
    return response_length, word_count, reading_level

quality_check = st.button("Check Answer Quality")

if quality_check and response_text:
    length, word_count, complexity = check_quality(response_text)

    st.subheader("Quality of Response:")
    st.write(f"Length of Response: {length} characters")
    st.write(f"Word Count: {word_count} words")
    st.write(f"English Complexity: {complexity}")
