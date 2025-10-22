import pandas as pd
from rapidfuzz import fuzz
from pythainlp.tokenize import word_tokenize

# -------------------
# ฟังก์ชันตัดคำภาษาไทย
# -------------------
def segment_thai(text):
    """ตัดคำภาษาไทยและคืนค่าเป็น list ของคำ"""
    if not text or pd.isna(text):
        return []
    words = word_tokenize(str(text), engine='newmm')
    return words

# -------------------
# ฟังก์ชันสร้าง n-grams
# -------------------
def create_ngrams(words, n):
    """สร้าง n-grams จาก list ของคำ"""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ''.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

# -------------------
# ฟังก์ชัน fuzzy match แบบ dynamic n-gram
# -------------------
def find_typo_with_ngrams_dynamic(text, keyword, fuzz_thresh=80, min_length_ratio=0.9):
    text_words = segment_thai(text)
    keyword_words = segment_thai(keyword)
    n = len(keyword_words)
    keyword_joined = ''.join(keyword_words)

    max_score = 0
    best_match = None

    # สร้าง n-grams ขนาด n-1, n, n+1
    for size in range(max(1, n-1), n+2):
        ngrams = create_ngrams(text_words, size)
        for ngram in ngrams:
            score = fuzz.ratio(ngram, keyword_joined)
            if len(ngram)/len(keyword_joined) >= min_length_ratio:
                if score > max_score:
                    max_score = score
                    best_match = ngram

    # fallback: ตรวจทั้งข้อความ
    if max_score < fuzz_thresh:
        text_joined = ''.join(text_words)
        score = fuzz.ratio(text_joined, keyword_joined)
        if score > max_score and len(text_joined)/len(keyword_joined) >= min_length_ratio:
            max_score = score
            best_match = text_joined

    is_match = max_score >= fuzz_thresh
    return is_match, max_score, best_match

# -------------------
# ตรวจว่าข้อความมีครบทุกคำในแต่ละ "เซตคีย์เวิร์ด"
# -------------------
def find_typo_multiple_keyword_sets_dynamic(text, keyword_sets, fuzz_thresh=80, min_length_ratio=0.9):
    """
    ตรวจสอบข้อความว่าตรงกับทุกคำในแต่ละเซต keyword หรือไม่
    ถ้ามีเซตใดที่ตรงครบทุกคำ → ถือว่า relevant
    """
    for kw_set in keyword_sets:
        matches = []
        for kw in kw_set:
            is_match, score, matched_text = find_typo_with_ngrams_dynamic(
                text, kw, fuzz_thresh=fuzz_thresh, min_length_ratio=min_length_ratio
            )
            matches.append((kw, is_match, score, matched_text))

        # ถ้าเจอครบทุกคำในเซตนี้
        if all(m[1] for m in matches):
            best_score = sum(m[2] for m in matches) / len(matches)
            matched_keywords = [m[0] for m in matches]
            matched_texts = [m[3] for m in matches]
            return True, matched_keywords, best_score, matched_texts

    # ถ้าไม่ครบทุกคำในทุกเซตเลย
    return False, None, 0, None

# -------------------
# ฟังก์ชันลบ blacklist
# -------------------
def remove_blacklist_exact(text, blacklist):
    for bl in blacklist:
        text = text.replace(bl, "")
    return text

# -------------------
# ตั้งค่าชื่อไฟล์, keywords และ blacklist
# -------------------
input_file = "CMPC_Monthly_20250531.xlsx"
output_file = "../outputfile/CMPC_Monthly_relevant1.xlsx"

keyword_sets = [
    ["คุณสู้เราช่วย"],
    # ["เราช่วย"],
    # ['ไถ่ถอน']
    ["ไถ่ถอน", "โฉนด"]
]

BLACKLIST = ["การ", "ความ"]
FUZZY_THRESHOLD = 80
MIN_LENGTH_RATIO = 0.7 #0.9

# -------------------
# อ่านไฟล์ Excel
# -------------------
print("กำลังอ่านไฟล์...")
df = pd.read_excel(input_file)

if 'รายละเอียด' not in df.columns:
    raise ValueError("ไม่พบคอลัมน์ 'รายละเอียด' ในไฟล์ Excel")

# -------------------
# ลบ blacklist ก่อนตัดคำ
# -------------------
df['รายละเอียด_cleaned'] = df['รายละเอียด'].apply(
    lambda x: remove_blacklist_exact(str(x), BLACKLIST)
)

# -------------------
# ตัดคำ (สำหรับดูในไฟล์)
# -------------------
print("กำลังตัดคำ...")
df['รายละเอียด_ตัดคำ'] = df['รายละเอียด_cleaned'].apply(
    lambda x: ' | '.join(segment_thai(str(x)))
)

# -------------------
# ตรวจจับคำผิด (แบบต้องครบทั้งเซต)
# -------------------
print(f"กำลังตรวจหาคำที่ตรงกับ {len(keyword_sets)} เซตคีย์เวิร์ด (threshold={FUZZY_THRESHOLD})...")
relevance_results = df['รายละเอียด_cleaned'].apply(
    lambda x: find_typo_multiple_keyword_sets_dynamic(
        str(x), keyword_sets, fuzz_thresh=FUZZY_THRESHOLD, min_length_ratio=MIN_LENGTH_RATIO
    )
)

df['Relevant'] = [r[0] for r in relevance_results]
df['Matched_keyword'] = [', '.join(r[1]) if r[1] else None for r in relevance_results]
df['Fuzzy_score'] = [r[2] for r in relevance_results]
df['Matched_word_in_text'] = [', '.join(r[3]) if r[3] else None for r in relevance_results]

# -------------------
# กรองเฉพาะแถวที่เกี่ยวข้อง
# -------------------
df_relevant = df[df['Relevant']]

# -------------------
# สรุปผล
# -------------------
print(f"\n{'='*60}")
print(f"📊 สรุปผลลัพธ์")
print(f"{'='*60}")
print(f"✅ พบข้อมูลที่เกี่ยวข้อง: {len(df_relevant)} / {len(df)} แถว")
if len(df) > 0:
    print(f"📈 อัตราการจับได้: {len(df_relevant)/len(df)*100:.2f}%")

if len(df_relevant) > 0:
    print(f"\n🔍 จำนวนแยกตาม Keyword Set:")
    for kw_set in keyword_sets:
        kw_label = ' + '.join(kw_set)
        count = df_relevant[df_relevant['Matched_keyword'].str.contains(kw_set[0], na=False)].shape[0]
        print(f"   • {kw_label}: {count} แถว")

# -------------------
# บันทึกไฟล์
# -------------------
df_relevant.to_excel(output_file, index=False)
print(f"\n💾 บันทึกไฟล์สำเร็จ: {output_file}")

# -------------------
# ตัวอย่างผลลัพธ์
# -------------------
if len(df_relevant) > 0:
    print(f"\n📋 ตัวอย่าง 5 แถวแรก:")
    sample_cols = ['Matched_keyword', 'Matched_word_in_text', 'Fuzzy_score']
    print(df_relevant[sample_cols].head().to_string(index=False))
