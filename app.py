import os
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="InnovateCorp RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 InnovateCorp RAG Assistant")
st.write("Ask questions about the InnovateCorp Knowledge Transfer Guide.")


# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()


# ---------------------------------------------------------
# INITIALIZE GEMINI
# ---------------------------------------------------------

@st.cache_resource
def initialize_models():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemma-4-31b-it",
        google_api_key=GOOGLE_API_KEY
    )

    return embeddings, llm


embeddings, llm = initialize_models()


# ---------------------------------------------------------
# INNOVATECORP KT GUIDE
# ---------------------------------------------------------

kt_guide_content = """
Welcome to InnovateCorp! This Knowledge Transfer (KT) guide is designed to help new employees navigate their initial weeks and understand key aspects of our operations. Our core values are Innovation, Collaboration, and Customer Focus.

**Team Structure:** You will be joining the 'Project Alpha' team, reporting to Sarah Chen, the Senior Project Manager. Your direct teammates include David Lee (Lead Developer), Maria Rodriguez (UI/UX Designer), and Tom Jackson (QA Engineer). Our team meetings are held every Monday at 10 AM in Conference Room 3, and daily stand-ups are at 9:30 AM via Google Meet.

**Key Tools & Software:** For project management, we use Jira for task tracking and Confluence for documentation. Our primary communication tool is Slack for instant messaging and Google Workspace for email and calendars. Development work is primarily done using Python and JavaScript, with code hosted on GitHub. Access to these tools will be granted within your first three days.

**Onboarding Process:** Your first week will focus on setup and introductions. You'll receive your laptop and login credentials on day one. HR will conduct an orientation session on Tuesday covering company policies, benefits, and payroll. You'll have one-on-one meetings with your team members throughout the week. By the end of your second week, you should have access to all necessary systems and have completed mandatory compliance training modules.

**Important Resources:** The company's internal knowledge base can be found at `internal.innovatecorp.com/kb`. This includes FAQs, best practices, and troubleshooting guides. For IT support, please submit a ticket via `support.innovatecorp.com` or call extension 5555. Health and wellness benefits information is available on the HR portal.

**Culture & Expectations:** InnovateCorp encourages a proactive and collaborative environment. We value open communication and continuous learning. Don't hesitate to ask questions; your team is here to support your growth. Performance reviews are conducted quarterly, and professional development courses are available through our 'InnovateLearn' platform.
"""


# ---------------------------------------------------------
# CREATE DOCUMENT
# ---------------------------------------------------------

@st.cache_resource
def create_vector_store():

    kt_documents = [
        Document(page_content=kt_guide_content)
    ]

    # Semantic chunking
    semantic_splitter = SemanticChunker(
        embeddings
    )

    kt_semantic_chunks = semantic_splitter.create_documents(
        [kt_guide_content]
    )

    # Create FAISS vector store
    vector_store = FAISS.from_documents(
        kt_semantic_chunks,
        embeddings
    )

    return vector_store


vector_store = create_vector_store()


# ---------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ---------------------------------------------------------
# RAG PROMPT
# ---------------------------------------------------------

rag_prompt = ChatPromptTemplate.from_template(
    """
You are an HR assistant for InnovateCorp using the company's
Knowledge Transfer guide.

Use ONLY the retrieved context below to answer the user's question.

If the answer is not available in the context, politely explain
that you don't know or that the information is not available
in the manual.

Do not invent information.

Retrieved Context:
{context}

Question:
{question}

Answer:
"""
)


# ---------------------------------------------------------
# FORMAT DOCUMENTS
# ---------------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        f"Content: {doc.page_content}"
        for doc in docs
    )


# ---------------------------------------------------------
# RAG FUNCTION
# ---------------------------------------------------------

def ask_rag(question):

    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)

    rag_chain = (
        rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return answer


# ---------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input(
    "Ask something about InnovateCorp..."
)


if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the knowledge base..."):

            answer = ask_rag(question)

        st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
