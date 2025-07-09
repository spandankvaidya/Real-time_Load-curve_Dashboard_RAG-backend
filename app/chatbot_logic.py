import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load the Groq API key from environment variables.
#    This is securely fetched from the 'GROQ_API_KEY' variable you set on Render.
try:
    groq_api_key = os.environ['GROQ_API_KEY']
except KeyError:
    # This provides a clear error in your Render logs if the key is missing.
    raise Exception("FATAL ERROR: GROQ_API_KEY environment variable not set.")

# 2. Initialize the ChatGroq model.
#    We use 'gemma-9b-it', the instruction-tuned version, which is perfect for a chatbot.
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma-9b-it",
    temperature=0.7  # Adjust for more or less creative responses
)

# 3. Create a system prompt to give the chatbot its persona and instructions.
#    This is crucial for guiding the LLM's behavior.
system_prompt = (
    "You are a helpful and knowledgeable 'Power Grid Assistant'. "
    "Your primary role is to answer questions related to power grids, electricity consumption, "
    "load forecasting, and renewable energy. You are part of a dashboard that shows "
    "real-time load curve predictions. Be concise, clear, and professional. "
    "If you don't know an answer, it's better to say so than to make up information."
)

# 4. Create the full prompt template that will be sent to the LLM.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"), # The user's question will be injected here.
    ]
)

# 5. Create a simple chain using LangChain Expression Language (LCEL).
#    This chains the components together: prompt -> LLM -> string output.
#    This is the simple "invoke" logic you requested.
chain = prompt | llm | StrOutputParser()

def get_chatbot_response(question: str) -> str:
    """
    Invokes the LLM chain with the user's question and returns the response.
    Includes basic error handling.
    """
    try:
        # The .invoke method sends the request to the Groq API.
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        print(f"Error invoking chatbot LLM: {e}")
        return "Sorry, I encountered an error and cannot process your request at the moment."