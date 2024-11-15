import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from textstat import text_standard, lexicon_count

model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

st.set_page_config(page_title="T5 Text Generation Demo")
st.title("T5 Model Text Generation")

st.write("Enter your input text below:")
user_input = st.text_area("Input Text")

if st.button("Generate"):
    with st.spinner("Generating..."):
       
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=50)

     
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.subheader("Output:")
        st.write(output_text)

def check_quality(output_text):
    response_length = len(output_text)
    word_count = lexicon_count(output_text, removepunct=True)
    reading_level = text_standard(output_text, float_output=False)
    return response_length, word_count, reading_level

quality_check = st.button("Check Answer Quality")

if quality_check and output_text:
    length, word_count, complexity = check_quality(output_text)

    st.subheader("Quality of Response:")
    st.write(f"Length of Response: {length} characters")
    st.write(f"Word Count: {word_count} words")
    st.write(f"English Complexity: {complexity}")
