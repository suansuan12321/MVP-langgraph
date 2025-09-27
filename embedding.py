import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import asyncio
import time
import logging
from pymilvus import connections, list_collections

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

## 环境变量
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")

file_paths = input("请输入文件路径（可多个，空格分隔）:").split()
file_paths = [fp for fp in file_paths if fp.strip()]

async def load_file_content(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = await asyncio.to_thread(loader.load)
        return docs
    elif ext in [".txt", ".md"]:
        return await asyncio.to_thread(lambda: open(file_path, "r", encoding="utf-8").read())
    elif ext in [".doc", ".docx"]:
        loader = Docx2txtLoader(file_path)
        docs = await asyncio.to_thread(loader.load)
        return docs
    else:
        raise ValueError(f"暂不支持的文件类型: {ext}")

async def embed_docs(file_paths, chunk_size=300, chunk_overlap=50):

     ## 加载文档-并发加载
    async def load_and_wrap(file_path):
        docs = await load_file_content(file_path)
        if isinstance(docs, str):
            from langchain_core.documents import Document
            docs = [Document(page_content=docs, metadata={"source": file_path})]
        return docs
    start_time = time.time()
    all_docs_nested = await asyncio.gather(*(load_and_wrap(fp) for fp in file_paths))
    all_docs = [doc for docs in all_docs_nested for doc in docs]
    end_time = time.time()
    logging.info(f"加载文档时间：{end_time - start_time}秒")

    ## 分割文档
    start_time = time.time()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = await asyncio.to_thread(text_splitter.split_documents, all_docs)
    end_time = time.time()
    logging.info(f"分割文档时间：{end_time - start_time}秒")

    ## 嵌入并入向量库
    start_time = time.time()
    if openai_api_key:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    end_time = time.time()
    logging.info(f"嵌入并入向量库时间：{end_time - start_time}秒")
    return split_docs, embeddings

def check_collection_exists(collection_name):
    connections.connect(host="localhost", port="19530")
    exist_collections = list_collections()
    return collection_name in exist_collections

async def main():
    split_docs, embeddings = await embed_docs(file_paths)
    collection_name = input("请输入数据库集合名称：")
    connection_args = {"host": "localhost", "port": "19530"}

    # 检查集合是否已存在
    exists = await asyncio.to_thread(check_collection_exists, collection_name)
    if exists:
        print(f"集合 '{collection_name}' 已存在。")
        print("请选择操作：Y:覆盖重写  N:跳过本次写入")
        choice = input("请输入字母选择（Y或N）：").strip()
        if choice.strip().lower() == "y":
            drop_old = True
        else:
            print("已跳过本次写入。")
            return
    else:
        drop_old = False

    start_time = time.time()
    await asyncio.to_thread(
        Milvus.from_documents,
        split_docs,
        embeddings,
        connection_args=connection_args,
        drop_old=drop_old,
        collection_name=collection_name
    )
    end_time = time.time()
    logging.info(f"嵌入并入向量库时间：{end_time - start_time}秒")

if __name__ == "__main__":
    asyncio.run(main())