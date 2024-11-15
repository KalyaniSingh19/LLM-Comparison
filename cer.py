import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from textstat import text_standard, lexicon_count

model_name = "cerebras/Cerebras-GPT-111M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.set_page_config(page_title="Cerebras-GPT Text Generation Demo")
st.title("Cerebras-GPT Model Text Generation")

st.write("Enter your prompt below:")
user_input = st.text_area("Input Prompt")

if st.button("Generate"):
    with st.spinner("Generating..."):
       
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.subheader("Generated Text:")
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
