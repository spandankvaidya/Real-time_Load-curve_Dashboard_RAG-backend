# Real-time_Load-curve_Dashboard_RAG-backend

This repository contains the backend service for the Real-Time Power Forecast Dashboard. It's a robust Python application built with FastAPI that serves both the machine learning predictions and the AI chatbot functionality.

![Backend Screenshot](https://raw.githubusercontent.com/spandankvaidya/Real-time_Load-curve_Dashboard_RAG-backend/main/Dashboard_screenshot.png)

## Core Features

-   **High-Performance API:** Built with **FastAPI** for fast, asynchronous request handling.
-   **Dynamic Graph Generation:** A stateless **Plotly Dash** application is mounted on FastAPI. It generates and serves the live prediction graph on-the-fly based on the date provided in the URL (`/dashboard/:date`).
-   **AI Chatbot "Jolt"**: Powered by **Groq**'s LPU™ Inference Engine using the `gemma2-9b-it` model. It features a context-aware system prompt implemented with **LangChain** to answer questions about the project.
-   **Machine Learning Integration:** Utilizes a pre-trained **LightGBM** model to generate power load forecasts for selected test days.
-   **Efficient Data Processing:** Leverages **Polars** for high-speed data loading and feature engineering from the source CSV files.
-   **Cloud-Ready:** Configured for seamless deployment on **Render** using a `Procfile` and Gunicorn.

## Tech Stack & Tools

<div>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" />
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  </a>
  <a href="https://dash.plotly.com/">
    <img src="https://img.shields.io/badge/Plotly_Dash-3DDC84?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly Dash" />
  </a>
  <a href="https://www.langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-A473E8?style=for-the-badge" alt="LangChain" />
  </a>
  <a href="https://groq.com/">
    <img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge" alt="Groq" />
  </a>
  <a href="https://lightgbm.readthedocs.io/">
    <img src="https://img.shields.io/badge/LightGBM-8E44AD?style=for-the-badge" alt="LightGBM" />
  </a>
  <a href="https://pola.rs/">
    <img src="https://img.shields.io/badge/Polars-1D2B3A?style=for-the-badge" alt="Polars" />
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/Numpy-4D77CF?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" />
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  </a>
  <a href="https://gunicorn.org/">
      <img src="https://img.shields.io/badge/gunicorn-%23499848.svg?style=for-the-badge&logo=gunicorn&logoColor=white" alt="Gunicorn" />
  </a>
  <a href="https://render.com/">
    <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=black" alt="Render" />
  </a>
</div>

## Author

-   **spandankvaidya** - [GitHub Profile](https://github.com/spandankvaidya)

## License

This project is licensed under the MIT License.
