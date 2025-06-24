# AI Resume Critique Tool
This is a Streamlit web application that uses AI to analyze your resume against a specific job description. It provides a detailed breakdown of your match score, identifies key skills, highlights missing skills, and offers a conversational AI chatbot (powered by Groq and Llama 3) to give you personalized feedback.


## Features

-   **📄 Resume Parsing**: Upload your resume in either PDF or DOCX format.
-   **🎯 Job Description Analysis**: Paste a job description to tailor the analysis.
-   **📊 Match Scoring**: Get an overall match score based on a weighted analysis of:
    -   **Skill Match**: Direct comparison of skills found in your resume vs. the job description.
    -   **Domain Match**: Determines the industry/domain relevance of your resume to the job.
    -   **Keyword Match**: Measures the overlap of important keywords between your resume and the job spec.
-   **💡 Skill Insights**:
    -   View a categorized list of skills detected in your resume.
    -   Quickly see which required skills are missing from your resume.
-   **🤖 AI Chatbot Expert**:
    -   Interact with an AI-powered resume expert (using Llama 3 via the fast Groq API).
    -   Ask for specific advice, suggestions for improvement, or how to better frame your experience.

## Tech Stack

-   **Framework**: Streamlit
-   **Language**: Python
-   **NLP**: spaCy (`en_core_web_sm` model)
-   **LLM Integration**: LangChain, ChatGroq (Llama 3)
-   **File Parsing**: PyPDF2, docx2txt

---
