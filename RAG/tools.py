import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


def set_faiss_db_from_json(db_name: str, docs: Document, mode: str = "overwrite"):
    save_path = os.path.join("faiss_dbs", db_name)
    os.makedirs(save_path, exist_ok=True)

    # --- Embeddings ---

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    faiss_index_path = os.path.join(save_path, "index.faiss")
    faiss_pkl_path = os.path.join(save_path, "index.pkl")

    if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
        if mode == "append":
            print(f"🟢 دیتابیس '{db_name}' پیدا شد — در حال افزودن داده‌های جدید...")
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(docs)
            vectorstore.save_local(save_path)
            print(f"✅ داده‌های جدید به '{db_name}' اضافه شد.")
        elif mode == "overwrite":
            print(f"🟠 دیتابیس '{db_name}' بازنویسی می‌شود...")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(save_path)
            print(f"✅ دیتابیس '{db_name}' با داده‌های جدید جایگزین شد.")
        else:
            raise ValueError("mode باید یکی از 'append' یا 'overwrite' باشد.")
    else:
        print(f"🔹 دیتابیس '{db_name}' وجود ندارد — در حال ساخت جدید...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(save_path)
        print(f"✅ دیتابیس جدید '{db_name}' ساخته شد.")


def ask_faiss_question(db_name: str, query: str, k: int = 3):
    """
    یک سوال از دیتابیس FAISS می‌پرسد و نتایج نزدیک‌ترین متون را برمی‌گرداند.

    پارامترها:
    db_name : str : نام دیتابیس در پوشه faiss_dbs
    query   : str : سوالی که می‌خواهی بپرسی
    k       : int : تعداد نتایج نزدیک که می‌خواهی نمایش داده شود
    """
    # آماده‌سازی embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # لود دیتابیس
    vectorstore_path = f"faiss_dbs/{db_name}"
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    # جستجو
    results = vectorstore.similarity_search(query, k=k)

    # نمایش نتایج
    for i, doc in enumerate(results):
        print(f"🔹 نتیجه {i + 1}:\n{doc.page_content}\n{'-' * 50}")
    return results

def answer_with_ai(faiss_results, user_query):
    client = OpenAI()
    context = "\n\n".join([doc.page_content for doc in faiss_results])

    prompt = f"""
شما یک دستیار هوشمند هستید. با توجه به اطلاعات زیر، به سوال کاربر پاسخ کامل:

اطلاعات:
{context}

سوال کاربر:
{user_query}

پاسخ:
"""

    print("ai",prompt)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # پاسخ دقیق و واقعی
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return answer


def json_to_docs_internet(json_data: dict) -> list:

    docs = []

    for service in json_data.get("internet_services", []):
        service_type = service.get("type", "")
        service_desc = service.get("description", "")

        for plan in service.get("plans", []):

            # --- ساخت متن ---
            text = f"""
سرویس: {str(service_type)}
توضیحات: {str(service_desc)}
نام پلن: {str(plan.get('name', ''))}
حداقل سرعت: {str(plan.get('min_speed', ''))}
حداکثر سرعت: {str(plan.get('max_speed', ''))}
سرعت آپلود: {str(plan.get('upload_speed', ''))}
قابل جابه‌جایی: {plan.get('movable', '')}
پوشش: {str(plan.get('coverage', ''))}
مودم: {plan.get('movable', '')}
ترافیک شبانه: {plan.get('night_traffic', '')}
حجم ترافیک: {str(plan.get('traffic_gb', ''))}
مدت زمان: {str(plan.get('duration_days', ''))}
IP: {str(plan.get('ip', ''))}
قیمت: {str(plan.get('price', ''))}
"""

            # --- تقسیم متن به chunk ---
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_text(text)

            # --- ساخت Document ---
            for chunk in chunks:
                docs.append(Document(page_content=chunk))

    return docs