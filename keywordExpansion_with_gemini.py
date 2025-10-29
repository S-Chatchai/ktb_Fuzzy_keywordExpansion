import re
import pandas as pd
from itertools import chain
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import json
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyDIyLTH6O8ccHCW26AhFFxVC_eD5apumkc")  # อย่าลืมใส่ API Key ของคุณเอง

# -------------------------------
# Load Excel
# -------------------------------
input_file = "../inputfile/thai_complaints_10000rows_batched.xlsx"
df = pd.read_excel(input_file)
texts = df['รายละเอียด'].tolist()[:100]

# -------------------------------
# Keywords
# -------------------------------
existing_keywords = [
    "ลงทะเบียน","ทวงหนี้","ปรับปรุงหนี้","ผ่อนไม่ได้","ปฏิเสธ"
]

# -------------------------------
# Model
# -------------------------------
model_st = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', use_auth_token=False)
threshold = 0.79

# -------------------------------
# Encode keywords แค่ครั้งเดียว
# -------------------------------
kw_embeddings = model_st.encode(existing_keywords, convert_to_tensor=True)

# -------------------------------
# Helper: clean text
# -------------------------------
def clean_text(text):
    # ลบเครื่องหมายพิเศษและตัวเลข
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\\|]', '', text)
    text = re.sub(r'[\d๐-๙]', '', text)
    # ลบ white space ทั้งหมด
    text = re.sub(r'\s+', '', text)
    return text

# -------------------------------
# Helper: generate n-grams
# -------------------------------
def generate_ngrams(tokens, n=2):
    return [''.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# -------------------------------
# Batch processing and data collection
# -------------------------------
batch_size_text = 100
batch_size_encode = 64
expanded_results = []
pair_set = set()

for start_text in tqdm(range(0, len(texts), batch_size_text), desc="Processing sentences"):
    batch_texts = texts[start_text:start_text + batch_size_text]

    all_tokens = []
    token_map = []

    for idx, text in enumerate(batch_texts):
        text_clean = clean_text(str(text))
        tokens = word_tokenize(text_clean, engine="newmm")
        
        # uni-gram + bi-gram + tri-gram
        bigrams = generate_ngrams(tokens, n=2)
        trigrams = generate_ngrams(tokens, n=3)
        ngram_tokens = list(chain(tokens, bigrams, trigrams))

        all_tokens.extend(ngram_tokens)
        token_map.extend([idx] * len(ngram_tokens))

    # Encode tokens ของ batch
    for i in tqdm(range(0, len(all_tokens), batch_size_encode), desc="Encoding tokens", leave=False):
        batch_tokens = all_tokens[i:i + batch_size_encode]
        token_embeddings = model_st.encode(batch_tokens, convert_to_tensor=True, batch_size=len(batch_tokens))

        sims = util.cos_sim(token_embeddings, kw_embeddings)

        for j, token in enumerate(batch_tokens):
            sim_values = sims[j]
            max_sim, idx_kw = torch.max(sim_values, dim=0)
            max_sim = float(max_sim)
            matched_kw = existing_keywords[idx_kw]

            if max_sim >= threshold and token != matched_kw:
                sentence_idx = token_map[i + j]
                sentence = batch_texts[sentence_idx]
                key_pair = (matched_kw, token)
                if key_pair not in pair_set:
                    pair_set.add(key_pair)
                    expanded_results.append({
                        "sentence": sentence,
                        "existing_keyword_matched": matched_kw,
                        "new_keyword": token,
                        "similarity": max_sim
                    })

# -------------------------------
# Save results to Excel first
# -------------------------------
df_expanded = pd.DataFrame(expanded_results)
excel_output_file = "../outputfile/expanded_keywords_for_review_test_no_clean100.xlsx"
df_expanded.to_excel(excel_output_file, index=False)
print(f"Saved initial results to {excel_output_file} for your review!")

# -------------------------------
# Prepare data for Gemini API and create JSON
# -------------------------------
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
results_to_save = {}

# Group keywords by their matched existing keyword
expanded_keywords_by_kw = df_expanded.groupby('existing_keyword_matched')['new_keyword'].apply(set).to_dict()

for existing_kw, new_words_set in expanded_keywords_by_kw.items():
    if not new_words_set:
        continue

    # Clean and consolidate the words
    cleaned_words = [re.sub(r'[^ก-๙a-zA-Z]', '', word).strip() for word in new_words_set]
    cleaned_words = [word for word in cleaned_words if len(word) > 1]  # Filter out very short or empty words
    unique_words = list(set(cleaned_words))
    
    # Create the prompt for Gemini
    prompt_list = [f'"{word}"' for word in unique_words]

    prompt = f"""คุณเป็นผู้เชี่ยวชาญภาษาไทย โปรดวิเคราะห์รายการคำต่อไปนี้ว่าคำใดมีความหมายเหมือนหรือใกล้เคียงกับคำว่า "{existing_kw}" ตามหลักเกณฑ์ต่อไปนี้:

รวมคำที่ซ้ำกันหรือคล้ายกัน: หากพบคำที่สะกดคล้ายกันแต่มีความหมายเดียวกัน (เช่น "ลงทะเบียนแอ" และ "ลงทะเบียนแอป") ให้ใช้คำที่สมบูรณ์กว่าเพียงคำเดียว

ความหมายไม่สูมบูรณ์: หากพบคำที่ความหมายไม่สมบูรณ์ (เช่น "ว่าลงทะเบียน") ให้ตัดออกเลย

ลบคำที่ไม่สมบูรณ์หรือมีอักขระแปลก: ตัดคำที่ไม่สมบูรณ์ (เช่น "ลงทะเบียน ตัว") หรือคำที่มีอักขระที่ไม่ใช่ตัวอักษรไทยออก

กรองคำที่ไม่เกี่ยวข้อง: ไม่ต้องแสดงคำที่ไม่มีความหมายใกล้เคียงหรือเกี่ยวข้องโดยตรงกับคำว่า "{existing_kw}" ในผลลัพธ์สุดท้าย

***ไม่ต้องสร้างคำใหม่ขึ้นมาเอง


ตรวจสอบคำต่อไปนี้: {', '.join(prompt_list)}

ตอบกลับเฉพาะ JSON format:
{{
  "related_keywords": ["คำที่ผ่านการตรวจสอบแล้ว", ...]
}}
"""
    try:
        response = gemini_model.generate_content(prompt)
        gemini_response_text = response.text.replace('```json', '').replace('```', '').strip()
        gemini_result = json.loads(gemini_response_text)
        results_to_save[existing_kw] = gemini_result.get("related_keywords", [])
    except Exception as e:
        print(f"An error occurred for keyword '{existing_kw}': {e}")
        results_to_save[existing_kw] = []

# Save the final results to a JSON file
json_output_file = "../outputfile/expanded_keywords_gemini_no_clean100.json"
with open(json_output_file, 'w', encoding='utf-8') as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=4)

print(f"Saved final, processed results to {json_output_file}!")
