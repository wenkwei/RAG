import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from io import BytesIO

st.set_page_config(page_title="安全AI助手", page_icon="🔒")
st.title("🔒 安全知识库问答机器人")

with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传PDF文件", type="pdf")
    st.markdown("""
    **使用指南**：
    1. 上传PDF文件
    2. 提出相关问题
    3. 查看AI回答及来源
    """)

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key 未设置")
    st.stop()

@st.cache_resource
def process_pdf(uploaded_file):
    try:
        pdf_stream = BytesIO(uploaded_file.getvalue())
        loader = PyPDFLoader(pdf_stream)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"PDF处理失败：{str(e)}")
        return None

@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="gemini-pro", temperature=0)

if uploaded_file:
    with st.spinner("处理PDF中..."):
        vectorstore = process_pdf(uploaded_file)
    if not vectorstore:
        st.stop()

    retriever = vectorstore.as_retriever()
    llm = get_llm()

    system_prompt = """
    你是专业助手，仅根据提供的文档回答问题。
    不知道就说：“根据文档无法回答”，不要编造。
    上下文：{context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 新版 LangChain 链式写法（不会报错！）
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.header("💬 开始提问")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("输入你的问题..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                ans = rag_chain.invoke(user_input)
                st.markdown(ans)

        st.session_state.messages.append({"role": "assistant", "content": ans})

else:
    st.info("👈 请先上传 PDF 文件")
