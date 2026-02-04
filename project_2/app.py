import pathlib, streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from mlx_lm import load, generate

class MLXLLM(LLM):
    model_id: str
    model: Any = None
    tokenizer: Any = None
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        # Load the model and tokenizer (cached if already downloaded)
        self.model, self.tokenizer = load(model_id)
    @property
    def _llm_type(self) -> str:
        return "mlx"
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Use the native mlx_lm generate function
        return generate(self.model, self.tokenizer, prompt=prompt, verbose=False, max_tokens=512)

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

@st.cache_resource
def init_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="thenlper/gte-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    llm = MLXLLM(model_id="mlx-community/gemma-3-4b-it-4bit")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
    )

chain = init_chain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("What is in your mind?")
if question:
    with st.spinner("Thinking..."):
        response = chain(
            {
                "question": question,
                "chat_history": st.session_state.history,   # <- supply it
            }
        )
    st.session_state.history.append((question, response["answer"]))


for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(bot)
