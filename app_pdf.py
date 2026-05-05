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
# 2. CARGA DEL PDF (Fragmentos más precisos)
# ==========================================
@st.cache_resource
def cargar_pdf():
    pdf_path = Path(__file__).resolve().parent / "geografia_mundial_llm.pdf"
    loader = PyPDFLoader(str(pdf_path))

    # Reducimos el chunk a 400 para que el modelo pequeño no se pierda leyendo
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    fragmentos = text_splitter.split_documents(loader.load())
    
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma.from_documents(fragmentos, embeddings)

# ==========================================
# 3. MOTOR RAG Y PROMPT (Anti-Alucinaciones)
# ==========================================
with st.spinner("Cartografiando el documento..."):
    db = cargar_pdf()
    
    # Traemos 6 fragmentos pequeños y súper precisos
    retriever = db.as_retriever(search_kwargs={"k": 6})
    
    # Temperatura en 0.0 es innegociable para RAG
    llm = Ollama(model="llama3.1:8b", temperature=0.0)
    
    # Prompt estructurado con XML y formato de reglas estrictas
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Actúas como Atlas, un geógrafo. "
            "Responde a la pregunta basándote ÚNICA Y EXCLUSIVAMENTE en la información "
            "dentro de las etiquetas <contexto>.\n\n"
            "REGLAS:\n"
            "1. Si la respuesta no está en el <contexto>, debes decir EXACTAMENTE: 'Mis mapas no contienen esa información.'\n"
            "2. NO uses conocimiento externo.\n"
            "3. NO inventes datos.\n\n"
            "<contexto>\n"
            "{context}\n"
            "</contexto>\n\n"
            "Pregunta: {question}\n\n"
            "Respuesta de Atlas:"
        )
    )

# ==========================================
# 4. CHAT INTERACTIVO (MODIFICADO)
# ==========================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"rol": "assistant", "contenido": "Saludos, viajero. Soy Atlas. ¿Qué coordenadas o historias del mundo exploraremos hoy?"}]

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])

if pregunta := st.chat_input("Escribe tu pregunta geográfica aquí..."):
    
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)
        
    with st.chat_message("assistant"):
        with st.spinner("Consultando los mapas..."):
            
            docs = retriever.invoke(pregunta)
            
            # MEJORA 4: Validación de seguridad por si el Retriever falla o no trae nada
            if not docs:
                respuesta = "Mis mapas no contienen esa información."
            else:
                # Separar los fragmentos con doble salto de línea para que el modelo los lea mejor
                contexto = "\n\n".join(d.page_content for d in docs)
                prompt_final = prompt_template.format(context=contexto, question=pregunta)
                
                respuesta = llm.invoke(prompt_final)
            
            st.write(respuesta)
            
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})

