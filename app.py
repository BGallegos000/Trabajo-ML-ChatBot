import streamlit as st
from langchain_community.llms import Ollama

# 1. Título simple
st.title("Chatbot Base")

# 2. Conexión al modelo local (El Chef)
# Asegúrate de que "ollama run llama3.2:1b" esté corriendo en otra ventana de CMD
llm = Ollama(model="llama3.2:1b")

# 3. Inicializar la memoria (Para que Streamlit no olvide la charla al recargar)
if "historial" not in st.session_state:
    st.session_state.historial = []

# 4. Dibujar el historial en pantalla
for msj in st.session_state.historial:
    with st.chat_message(msj["rol"]):
        st.write(msj["texto"])

# 5. Capturar input y procesar
prompt = st.chat_input("Escribe tu mensaje...")

if prompt:
    # A. Guardar y mostrar lo que el usuario escribió
    st.session_state.historial.append({"rol": "user", "texto": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # B. Pedirle al modelo que genere la respuesta
    with st.chat_message("assistant"):
        # invoke le manda el texto a Ollama y espera la respuesta
        respuesta_modelo = llm.invoke(prompt) 
        st.write(respuesta_modelo)

    # C. Guardar la respuesta del modelo en el historial
    st.session_state.historial.append({"rol": "assistant", "texto": respuesta_modelo})