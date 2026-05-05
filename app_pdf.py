import streamlit as st
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate



st.set_page_config(page_title="Atlas Geográfico", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #121216; color: #ececf1; }
    h1 { color: #f4f4f8; font-weight: 600; font-size: 1.5rem; border-bottom: 1px solid #333; padding-bottom: 10px; }
    [data-testid="stChatMessage"] { background-color: #1c1c22 !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("COMPASS")

# ==========================================
# 2. CARGA DEL PDF (Ultra Corto)
# ==========================================
@st.cache_resource
def cargar_pdf():
    pdf_path = Path(__file__).resolve().parent / "geografia_mundial_llm.pdf"
    loader = PyPDFLoader(str(pdf_path))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    fragmentos = text_splitter.split_documents(loader.load())
    
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma.from_documents(fragmentos, embeddings)

# ==========================================
# 3. MOTOR RAG Y PROMPT
# ==========================================
with st.spinner("Cartografiando el documento..."):
    db = cargar_pdf()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="qwen3.5:9b", temperature=0.1)
    
    # Prompt estricto y con temática geográfica
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Eres 'Atlas', un erudito en geografía mundial e historia. "
            "Responde la pregunta del viajero basándote ÚNICAMENTE en este texto.\n"
            "Si el texto no menciona la respuesta, di exactamente: 'Mis mapas no contienen esa información.'\n"
            "No inventes datos.\n\n"
            "TEXTO:\n{context}\n\n"
            "PREGUNTA:\n{question}\n\n"
            "ATLAS:"
        )
    )

# ==========================================
# 4. CHAT INTERACTIVO
# ==========================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"rol": "assistant", "contenido": "Saludos, viajero. Soy Atlas. ¿Qué coordenadas o historias del mundo exploraremos hoy?"}]

# Dibujar chat anterior
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])

# Escribir nueva pregunta
if pregunta := st.chat_input("Escribe tu pregunta geográfica aquí..."):
    
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)
        
    with st.chat_message("assistant"):
        with st.spinner("Consultando los mapas..."):
            
            # Buscar en el PDF y armar respuesta
            docs = retriever.invoke(pregunta)
            contexto = "\n".join(d.page_content for d in docs)
            prompt_final = prompt_template.format(context=contexto, question=pregunta)
            
            respuesta = llm.invoke(prompt_final)
            st.write(respuesta)
            
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})