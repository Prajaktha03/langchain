import os

from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title("Celebrity Search Results")

input_text = st.text_input("Search the celebrity you want")

first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity: {name}"
)

llm = OpenAI(temperature=0.9)

chain = LLMChain(
    prompt=first_input_prompt,
    llm=llm,
    verbose=True
)

if input_text:
    response = chain.run(name=input_text)
    st.write(response)
