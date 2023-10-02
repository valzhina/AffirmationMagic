# os module provides a way to work with Operating System(win/mac..)
import os

# Used to buils the app
import streamlit as st

# Used to build LLM Workflow
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


#App Framework
st.title('👩‍🚀 Fun GPT Creator')
 
# Place for prompt
prompt = st.text_input('Your affirmation goes here')

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me an affirmation for {topic}'
 )

# Script templetes  
script_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template = 'Write me a script based on this TITLE: {title} while leveraging this wikipedia research:{wikipedia_research}'
 )

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
 

#Creates an instance of OpenAI service(Llms) temperature sets creativity level
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template,verbose=True, output_key='title',
memory=title_memory) 
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', 
memory=script_memory)
# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
# output_variables=['title', 'script'], verbose=True)

wiki = WikipediaAPIWrapper()

#Screen response 
if prompt:
    # response = sequential_chain({'topic':prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research=wiki_research) 

    st.write(title) 
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research) 