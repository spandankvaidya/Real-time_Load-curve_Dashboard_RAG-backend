# app/chatbot_logic.py

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP THE LLM ---
# The LLM is initialized once when the module is loaded.
# It relies on the GROQ_API_KEY being set in the environment.
llm = ChatGroq(
    model='gemma2-9b-it',
    api_key=os.environ.get("GROQ_API_KEY")
)

# --- 2. DEFINE A SIMPLE PROMPT TEMPLATE ---
# This template instructs the LLM on how to behave.
GENERAL_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the following question to the best of your ability.

    Question: {input}
    """
)

# --- 3. CREATE THE CORE LOGIC CHAIN ---
# We combine the prompt and the LLM into a single, reusable chain.
general_chain = GENERAL_PROMPT | llm

# --- 4. DEFINE THE ANSWER GENERATION FUNCTION ---
# This async function is what your API endpoint will call.
async def generate_answer(question: str):
    """
    Generates an answer by invoking the LLM chain with the user's question.
    """
    print(f"-> Received question: {question}")
    print("-> Invoking the general knowledge LLM chain...")
    
    try:
        # Asynchronously call the chain. The 'ainvoke' method is for async environments like FastAPI.
        response = await general_chain.ainvoke({"input": question})
        
        # The actual text response is in the 'content' attribute of the response object.
        answer = response.content
        print(f"-> LLM generated answer successfully.")
        return answer
    except Exception as e:
        # This will catch errors if the API key is invalid or the Groq service is down.
        print(f"ERROR: An error occurred while invoking the LLM: {e}")
        # It's good practice to return a user-friendly error message.
        return "Sorry, I am unable to process your request at the moment. Please try again later."
