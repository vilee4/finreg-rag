import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 文本分块
# chunk_size 每个文本块的大小
# chunk_overlap 块之间重叠的大小
def chunk_text(text:str, chunk_size=300, chunk_overlap=50):
    # 递归尝试不同的分隔符，直到将文本分割成符合要求的小块 
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""],  # 分隔符
        keep_separator=True,
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    return chunks

if __name__=="__main__":
    # 1.读取文本
    text_path = "../data/east_sample.txt"
    json_path = "../data/chunks.json"
    # 相对路径拼接成绝对路径      
    # """获取项目内的完整路径"""
    projetcPath = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(projetcPath,text_path)
    full_json_path = os.path.join(projetcPath,json_path)
    if not os.path.exists(full_path):
        print("未找到east_sample.txt,请确认文件路径")
        exit(1)

    with open(full_path,"r",encoding="utf-8") as f:
        text = f.read()
    
    print("原始文本长度：",len(text),"字符")

    # 2. 文本分块
    chunks = chunk_text(text)
    print(f"\n✅ 共生成 {len(chunks)} 个文本块\n")

    # 3.打印前2块示例
    for i,chunk in enumerate(chunks[:3]):
        print(f"-----块{i+1}-----")
        print(chunk.strip()[:400] + ("..." if len(chunk) > 400 else ""))
        print()

    # 4.保存为JSON
    chunks_data = [{"id": i, "text": chunk.strip()} for i, chunk in enumerate(chunks)]
    with open(full_json_path,"w",encoding="utf-8") as f:
        json.dump(chunks_data,f,ensure_ascii=False,indent=2)
    
    print("💾 已保存分块结果到 data/chunks.json")