import streamlit as st
import os
import PyPDF2
import docx2txt
import spacy
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install the spaCy model by running: python -m spacy download en_core_web_sm")
    st.stop()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text

def extract_skills(doc):
    skill_patterns = {
        'programming_languages': ['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift', 'kotlin'],
        'web_technologies': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
        'databases': ['sql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis'],
        'ai_ml': ['machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision', 'tensorflow', 'pytorch'],
        'tools': ['git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'jira'],
        'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 'project management'],
        'marketing': ['seo', 'social media', 'content marketing', 'email marketing', 'analytics', 'crm'],
        'finance': ['accounting', 'financial analysis', 'excel', 'powerbi', 'tableau']
    }
    
    found_skills = {category: [] for category in skill_patterns}
    text_lower = doc.text.lower()
    
    for category, skills in skill_patterns.items():
        for skill in skills:
            if skill in text_lower:
                found_skills[category].append(skill)
    
    return found_skills

def compare_with_job(resume_text, job_description):
    resume_doc = nlp(resume_text.lower())
    job_doc = nlp(job_description.lower())
    
    resume_skills = extract_skills(resume_doc)
    job_skills = extract_skills(job_doc)
    
    skill_scores = {}
    all_skills_flat = []
    missing_skills = []
    
    for category in resume_skills.keys():
        resume_category_skills = set(resume_skills[category])
        job_category_skills = set(job_skills[category])
        
        if job_category_skills:
            score = len(resume_category_skills.intersection(job_category_skills)) / len(job_category_skills)
            skill_scores[category] = score
            missing_skills.extend(list(job_category_skills - resume_category_skills))
        else:
            skill_scores[category] = 0
            
        all_skills_flat.extend(resume_category_skills)
    
    domain_score = calculate_domain_relevance(resume_doc, job_doc)
    keyword_score = calculate_keyword_relevance(resume_doc, job_doc)
    skill_match_score = sum(skill_scores.values()) / len(skill_scores)
    
    final_score = {
        "overall_match": (skill_match_score * 0.5 + 
                         keyword_score * 0.3 + 
                         domain_score * 0.2),
        "skill_match": skill_match_score,
        "skill_scores_by_category": skill_scores,
        "keyword_match": keyword_score,
        "domain_match": domain_score,
        "skills_found": resume_skills,
        "missing_skills": list(set(missing_skills))
    }
    
    return final_score

def calculate_domain_relevance(resume_doc, job_doc):
    domains = {
        'tech': ['software', 'programming', 'developer', 'engineering', 'technical', 'ai', 'ml', 'data'],
        'marketing': ['marketing', 'advertising', 'brand', 'social media', 'content', 'seo'],
        'finance': ['finance', 'accounting', 'banking', 'investment', 'financial'],
        'healthcare': ['medical', 'healthcare', 'clinical', 'patient', 'health'],
        'sales': ['sales', 'business development', 'account management', 'client', 'customer'],
        'hr': ['human resources', 'recruiting', 'talent', 'hr', 'training'],
        'operations': ['operations', 'project management', 'process', 'quality', 'logistics']
    }
    
    job_text = job_doc.text.lower()
    job_domain = None
    max_matches = 0
    
    for domain, keywords in domains.items():
        matches = sum(1 for keyword in keywords if keyword in job_text)
        if matches > max_matches:
            max_matches = matches
            job_domain = domain
    
    if not job_domain:
        return 0.5
    
    resume_text = resume_doc.text.lower()
    domain_keywords = domains[job_domain]
    domain_matches = sum(1 for keyword in domain_keywords if keyword in resume_text)
    
    return domain_matches / len(domain_keywords) if domain_keywords else 0

def calculate_keyword_relevance(resume_doc, job_doc):
    important_pos = ['NOUN', 'PROPN', 'ADJ', 'VERB']
    
    job_keywords = [token.text.lower() for token in job_doc 
                   if token.pos_ in important_pos and not token.is_stop]
    
    resume_text = resume_doc.text.lower()
    matches = sum(1 for keyword in job_keywords if keyword in resume_text)
    
    return matches / len(job_keywords) if job_keywords else 0

def initialize_chatbot():
    try:
        groq_api_key = os.environ['GROQ_API_KEY']
        model = 'llama3-8b-8192'
        
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model
        )
        
        system_prompt = '''You are an expert resume reviewer. Analyze resumes and provide professional, 
        constructive feedback on structure, content, and improvements. Be specific and actionable in your advice.'''
        
        memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])
        
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )
        
        return conversation
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Resume Critique Tool", layout="wide")
    
    st.title("AI Resume Critique Tool")
    st.write("Upload your resume and job description to get AI-powered feedback!")
    
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=['pdf', 'docx'])
    
    with col2:
        job_description = st.text_area("Paste the job description you're targeting")
    
    if uploaded_file and job_description:
        if st.button("Process Resume", type="primary"):
            with st.spinner("Analyzing resume..."):
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_docx(uploaded_file)
                
                comparison = compare_with_job(resume_text, job_description)
                st.session_state.analysis_results = comparison
                st.session_state.processed = True
    
    if st.session_state.processed and st.session_state.analysis_results:
        comparison = st.session_state.analysis_results
        
        st.header("Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Match", f"{comparison['overall_match']*100:.1f}%")
        with col2:
            st.metric("Skill Match", f"{comparison['skill_match']*100:.1f}%")
        with col3:
            st.metric("Domain Match", f"{comparison['domain_match']*100:.1f}%")
        
        st.subheader("Skills Analysis")
        for category, skills in comparison['skills_found'].items():
            if skills:
                score = comparison['skill_scores_by_category'].get(category, 0)
                st.write(f"**{category.replace('_', ' ').title()}** ({score*100:.1f}% match)")
                st.write(", ".join(skills))
        
        if comparison['missing_skills']:
            st.subheader("Missing Skills")
            st.write(", ".join(comparison['missing_skills']))
        
        st.header("Chat with AI Resume Expert")
        if "conversation" not in st.session_state:
            st.session_state.conversation = initialize_chatbot()
        
        if st.session_state.conversation:
            user_question = st.text_input("Ask for specific feedback or suggestions:")
            if user_question:
                with st.spinner("Getting AI feedback..."):
                    context = f"""
                    Resume Analysis:
                    - Overall Match: {comparison['overall_match']*100:.1f}%
                    - Skills Found: {', '.join([skill for skills in comparison['skills_found'].values() for skill in skills])}
                    - Missing Skills: {', '.join(comparison['missing_skills'])}
                    
                    Question: {user_question}
                    """
                    
                    response = st.session_state.conversation.predict(human_input=context)
                    st.write("AI Expert:", response)

if __name__ == "__main__":
    main()
