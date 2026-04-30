import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import re

st.title("Chat Beta 1.1")

# 1. Función para preparar el PDF (Se ejecuta solo una vez)
@st.cache_resource
def preparar_base_de_datos():
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "Documentos/Biblioteca_B.pdf"

    # A. Leer el PDF
    loader = PyPDFLoader(str(pdf_path))  # Asegúrate de que exista este archivo
    documentos = loader.load()

    # Limpieza para mejorar recuperación semántica:
    # - Une palabras partidas por guion al final de línea
    # - Normaliza saltos de línea múltiples
    # - Mantiene párrafos separados para no perder contexto
    for doc in documentos:
        texto = doc.page_content
        texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
        texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
        texto = re.sub(r"\n{2,}", "\n\n", texto)
        texto = re.sub(r"\s{2,}", " ", texto).strip()
        doc.page_content = texto
    
    # B. Picar el documento en fragmentos (mejor para español y texto técnico)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    fragmentos = text_splitter.split_documents(documentos)
    
    # C. Convertir el texto a números (Embeddings) y guardarlo en una Base de Datos Vectorial (Chroma)
    modelo_vectores = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    base_de_datos = Chroma.from_documents(fragmentos, modelo_vectores)
    
    return base_de_datos

# 2. Cargar el motor y la base de datos
with st.spinner("Leyendo el PDF y preparando el cerebro..."):
    vector_db = preparar_base_de_datos()
    # Conectamos con tu Ollama local
    llm = Ollama(model="qwen3.5:4b", temperature=0)
    # Prompt explícito (evita problemas de mapeo query/question en cadenas prearmadas)
    prompt_qa = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Eres un experto en conducción y preparación para pruebas de conducción en Chile.\n"
            "Responde la pregunta usando SOLO el CONTEXTO recuperado.\n"
            "Si el CONTEXTO contiene la respuesta, respóndela de forma directa y clara en español.\n"
            "Si la respuesta NO está en el CONTEXTO, responde exactamente: NO_LO_SE.\n\n"
            "CONTEXTO:\n{context}\n\n"
            "PREGUNTA:\n{question}\n\n"
            "RESPUESTA:"
        ),
    )
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
    )

# 3. Interfaz de Usuario
pregunta = st.text_input("Haz una pregunta sobre el PDF:")

if st.button("Buscar en el documento"):
    if pregunta:
        with st.spinner("Buscando en el documento y pensando..."):
            docs = retriever.invoke(pregunta)
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            prompt = prompt_qa.format(context=context, question=pregunta)
            resultado = llm.invoke(prompt)
            
            st.success("Respuesta:")
            st.write(resultado)

            with st.expander("Ver fragmentos usados (fuentes)"):
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"**Fuente {i}**")
                    st.write(doc.page_content)

            with st.expander("Ver prompt enviado al modelo (debug)"):
                st.code(prompt)
    else:
        st.warning("Por favor escribe una pregunta.")