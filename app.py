import streamlit as st
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 页面设置
st.set_page_config(page_title="售前客服知识库", page_icon="📱")
st.title("📱 手机售前客服问答机器人")

# 侧边栏
with st.sidebar:
    st.header("📂 上传知识库文件")
    uploaded_file = st.file_uploader("支持 PDF / TXT 文件", type=["pdf", "txt"])
    st.markdown("""
    **使用说明**
    1. 上传手机客服记录（.txt 或 .pdf）
    2. 输入问题即可自动查询知识库
    3. AI 只根据你上传的内容回答
    """)

# API Key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("❌ 请在 Streamlit Secrets 中配置 GOOGLE_API_KEY")
    st.stop()

# 处理文件（支持 PDF + TXT）
@st.cache_resource
def process_file(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        documents = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        # PDF
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

        # TXT
        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
            documents = loader.load()

        os.unlink(tmp_path)

        # 切分
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        # 向量库
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"处理失败：{str(e)}")
        return None

# LLM
@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="gemini-pro", temperature=0)

# 主逻辑
if uploaded_file:
    with st.spinner("⚙️ 正在解析文件..."):
        db = process_file(uploaded_file)
    if not db:
        st.stop()

    retriever = db.as_retriever()
    llm = get_llm()

    system_prompt = """
    你是手机售前客服智能助手，只根据提供的知识库内容回答。
    不知道就说“根据知识库暂无相关信息”，不许编造。

    知识库内容：
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 聊天界面
    st.subheader("💬 开始提问")
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    for msg in st.session_state.msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("请输入您的问题...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.msgs.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("查找中..."):
                ans = rag_chain.invoke(user_input)
                st.markdown(ans)

        st.session_state.msgs.append({"role": "assistant", "content": ans})

else:
    st.info("👈 请先上传 PDF 或 TXT 格式的客服记录文件")
