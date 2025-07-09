import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    groq_api_key = os.environ['GROQ_API_KEY']
except KeyError:
    raise Exception("FATAL ERROR: GROQ_API_KEY environment variable not set.")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='gemma2-9b-it',
    temperature=0.7  
)

system_prompt = (
    "Your name is Jolt. You are a helpful and knowledgeable power grid assistant'. "
    "Your primary role is to answer questions related to power grids, electricity consumption, "
    "load forecasting, and renewable energy. You are part of a dashboard that shows "
    "real-time load curve predictions. Be concise, clear, and professional. "
    "If you don't know an answer, it's better to say so than to make up information.\n\n"

    "--- PROJECT DETAILS ---\n"
    "In addition to your general knowledge, you must be able to explain the project itself. "
    "When asked 'what is this project?', 'how does this work?', or similar questions, "
    "use the following information:\n"
    "*   **Project Goal:** This dashboard demonstrates a real-time load curve prediction system.\n"
    "*   **Dataset:** The system is built on a public load curve dataset from Kaggle for the year 2017.\n"
    "*   **Methodology:** A machine learning model was developed to make the predictions. Specifically:\n"
    "    *   **Model:** A LightGBM (Light Gradient Boosting Machine) model was chosen for its performance and speed.\n"
    "    *   **Data Split:** The 2017 dataset was divided into three sets: 300 days for training the model, 30 days for validation, and 34 specific days were reserved for testing.\n"
    "*   **How it Works:** The user selects one of the 34 available test dates from the calendar. The backend then passes the data for that specific day to the pre-trained LightGBM model to generate a 24-hour forecast.\n"
    "*   **Visualization:** The main graph on the dashboard is created using Plotly Dash. It shows a real-time comparison between the model's 'Predicted' power consumption and the 'Actual' power consumption from the test data."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()

def get_chatbot_response(question: str) -> str:
    """
    Invokes the LLM chain with the user's question and returns the response.
    Includes basic error handling.
    """
    try:
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        print(f"Error invoking chatbot LLM: {e}")
        return "Sorry, I encountered an error and cannot process your request at the moment."
