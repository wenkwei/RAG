import streamlit as st
import os
from dotenv import load_dotenv
# 关键修复：使用 langchain_community
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO

# 1. 页面设置
st.set_page_config(page_title="我的安全AI助手", page_icon="🔒")
st.title("🔒 安全知识库问答机器人")

# 2. 侧边栏：上传文件
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传PDF文件", type="pdf")
    
    # 显示使用说明
    st.markdown("""
    **使用指南**：
    1. 上传PDF文件
    2. 提出相关问题
    3. 查看AI回答及来源
    """)

# 3. 获取 API Key（支持本地环境变量 + Streamlit Cloud 密钥）
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key 未设置，请在 Streamlit Cloud 中配置")
    st.stop()

# 4. 优化的文件处理函数
@st.cache_resource(show_spinner=False)
def process_pdf(uploaded_file):
    try:
        pdf_stream = BytesIO(uploaded_file.getvalue())
        loader = PyPDFLoader(pdf_stream)
        documents = loader.load()
        
        # 文本切分
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # 创建向量库
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"PDF处理失败：{str(e)}")
        return None

# 5. 初始化模型
@st.cache_resource(show_spinner=False)
def get_llm():
    return GoogleGenerativeAI(model="gemini-pro", temperature=0)

# 6. 主逻辑
if uploaded_file:
    with st.spinner("📄 正在处理PDF，请稍候..."):
        vectorstore = process_pdf(uploaded_file)
    
    if not vectorstore:
        st.stop()
    
    retriever = vectorstore.as_retriever()
    llm = get_llm()

    # 提示词
    system_prompt = """
    你是一个专业的AI助手。
    请根据提供的上下文回答问题。
    如果不知道答案，请直接说“根据提供的文档，我无法回答这个问题”，不要编造。

    上下文：
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 构建RAG链
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 聊天界面
    st.header("💬 开始提问")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("请输入你的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt_input)
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = rag_chain.invoke({"input": prompt_input})
                answer = response["answer"]
                st.markdown(answer)

                with st.expander("📎 查看参考来源"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**片段 {i+1}** | 页码：{doc.metadata.get('page', '未知')}")
                        st.write(doc.page_content[:350] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 请先在左侧上传 PDF 文件开始使用")
