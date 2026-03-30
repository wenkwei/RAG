import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    
    if uploaded_file:
        st.success("文件已上传，处理中...")
        # 显示进度条
        progress = st.progress(0)
        progress.progress(50)
    
    # 显示使用说明
    st.markdown("""
    **使用指南**：
    1. 上传PDF文件
    2. 提出相关问题
    3. 查看AI回答及来源
    """)

# 3. 安全获取API Key（从环境变量）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("API Key未设置，请联系管理员")
    st.stop()

# 4. 优化的文件处理函数（内存存储）
@st.cache_resource
def process_pdf(uploaded_file):
    # 直接从内存读取PDF
    pdf_stream = BytesIO(uploaded_file.getvalue())
    loader = PyPDFLoader(pdf_stream)
    documents = loader.load()
    
    # 文本切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 创建向量数据库
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore

# 5. 初始化模型（缓存优化）
@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="gemini-pro", temperature=0)

# 6. 主逻辑
if uploaded_file:
    try:
        # 获取向量数据库
        vectorstore = process_pdf(uploaded_file)
        retriever = vectorstore.as_retriever()
        
        # 获取模型
        llm = get_llm()
        
        # 定义提示词
        system_prompt = (
            "你是一个乐于助人的AI助手。请使用以下提供的上下文来回答用户的问题。"
            "如果上下文中没有相关信息，请直接说你不知道，不要编造答案。"
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 构建问答链
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # 7. 聊天界面
        st.header("开始提问")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 显示历史聊天记录
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 处理用户输入
        if prompt_input := st.chat_input("基于上传的文件提问..."):
            # 显示用户问题
            with st.chat_message("user"):
                st.markdown(prompt_input)
            st.session_state.messages.append({"role": "user", "content": prompt_input})

            # 显示AI回答
            with st.chat_message("assistant"):
                with st.spinner("正在思考..."):
                    response = rag_chain.invoke({"input": prompt_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # 详细显示参考来源
                    with st.expander("查看详细参考来源"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**来源 #{i+1}** (页码: {doc.metadata.get('page', '未知')})")
                            st.markdown(f"{doc.page_content[:300]}...")
    
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
    except Exception as e:
        st.error(f"处理过程中出错: {str(e)}")
        st.stop()

# 8. 首次使用提示
else:
    st.info("请先在左侧侧边栏上传PDF文件开始使用")
