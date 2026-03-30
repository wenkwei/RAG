import streamlit as st
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from io import BytesIO

# 页面设置
st.set_page_config(page_title="安全知识库问答机器人", page_icon="🔒")
st.title("🔒 安全知识库问答机器人")

# 侧边栏
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传PDF文件", type="pdf")
    st.markdown("""
    **使用指南**：
    1. 上传PDF文件
    2. 提出相关问题
    3. 查看AI回答及来源
    """)

# 获取API Key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key 未设置")
    st.stop()

# 处理PDF函数（已修复临时文件问题）
@st.cache_resource
def process_pdf(uploaded_file):
    try:
        # 将上传的BytesIO写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # 用临时文件路径加载PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)  # 加载后立即删除临时文件
        
        # 文本切分
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # 向量库构建
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"PDF处理失败：{str(e)}")
        return None

# 初始化LLM
@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="gemini-pro", temperature=0)

# 主逻辑
if uploaded_file:
    with st.spinner("📄 正在处理PDF，请稍候..."):
        vectorstore = process_pdf(uploaded_file)
    if not vectorstore:
        st.stop()

    retriever = vectorstore.as_retriever()
    llm = get_llm()

    # 提示词模板
    system_prompt = """
    你是专业的安全知识库助手，仅根据提供的文档上下文回答问题。
    如果上下文中没有相关信息，请直接说："根据提供的文档，我无法回答这个问题"，严禁编造内容。

    上下文：
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 文档格式化函数
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 新版RAG链式调用（完全兼容最新LangChain）
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 聊天界面
    st.header("💬 开始提问")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理用户输入
    if user_input := st.chat_input("请输入你的问题..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                answer = rag_chain.invoke(user_input)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 请先在左侧上传 PDF 文件开始使用")
