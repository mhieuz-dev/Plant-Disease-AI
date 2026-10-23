import requests
import json

BASE_URL = "http://localhost:8000"

# 20 câu hỏi test - giữ nguyên thứ tự này
questions = [
    ("cafe_Leaf rust",   "Bệnh gỉ sắt cà phê dùng thuốc gì và phun liều lượng bao nhiêu?"),
    ("cafe_Leaf rust",   "Bệnh gỉ sắt cà phê phòng ngừa như thế nào?"),
    ("cafe_Miner",       "Sâu vẽ bùa cà phê xuất hiện vào mùa nào và xử lý ra sao?"),
    ("cafe_Miner",       "Sâu vẽ bùa cà phê dùng thuốc gì để trị?"),
    ("cafe_Phoma",       "Bệnh đốm lá Phoma cà phê có triệu chứng gì?"),
    ("cafe_Phoma",       "Bệnh đốm lá Phoma cà phê điều trị bằng thuốc nào?"),
    ("cafe_Leaf rust",   "Khi nào nên phun thuốc phòng bệnh gỉ sắt cho cà phê?"),
    ("rice_Leaf Blast",  "Bệnh đạo ôn lá lúa dùng thuốc gì và phun mấy lần?"),
    ("rice_Leaf Blast",  "Bệnh đạo ôn lá lúa phòng ngừa bằng cách nào?"),
    ("rice_Leaf scald",  "Bệnh cháy bìa lá lúa do nguyên nhân gì gây ra?"),
    ("rice_Leaf scald",  "Bệnh cháy bìa lá lúa điều trị như thế nào?"),
    ("rice_Sheath Blight","Bệnh khô vằn lúa dùng thuốc gì để xử lý?"),
    ("rice_Sheath Blight","Bệnh khô vằn lúa phát triển mạnh trong điều kiện thời tiết nào?"),
    ("rice_Leaf Blast",  "Liều lượng phun thuốc trị đạo ôn lá lúa là bao nhiêu?"),
    ("tieu_Footrot",     "Bệnh chết nhanh hồ tiêu do nấm gì gây ra và điều trị thế nào?"),
    ("tieu_Footrot",     "Bệnh chết nhanh hồ tiêu phòng ngừa bằng cách nào?"),
    ("tieu_Pollu_Disease","Bệnh thán thư hồ tiêu có triệu chứng gì và dùng thuốc gì?"),
    ("tieu_Slow-Decline","Bệnh chết chậm hồ tiêu khác bệnh chết nhanh ở điểm nào?"),
    ("tieu_Slow-Decline","Bệnh chết chậm hồ tiêu điều trị như thế nào?"),
    ("tieu_Footrot",     "Khi phát hiện bệnh chết nhanh hồ tiêu cần làm gì ngay lập tức?"),
]

def run_test(mode_name):
    print(f"\n{'='*60}")
    print(f"CHẠY: {mode_name}")
    print('='*60)
    results = []
    
    # Tạo 1 session dùng chung
    r = requests.post(f"{BASE_URL}/conversation/new")
    conv_id = r.json()["conversation_id"]
    
    for i, (class_name, question) in enumerate(questions, 1):
        r = requests.post(f"{BASE_URL}/chat", json={
            "conversation_id": conv_id,
            "question": f"[Cây đang mắc bệnh: {class_name}]\n{question}"
        })
        data = r.json()
        answer   = data.get("answer", "")
        citations = data.get("citations", [])
        
        print(f"\nCâu {i}: {question}")
        print(f"Trả lời: {answer[:400]}")
        print(f"Citations: {citations}")
        print("-"*40)
        
        results.append({
            "q": i,
            "question": question,
            "answer": answer,
            "citations": citations
        })
    
    return results

if __name__ == "__main__":
    results = run_test("RAG MODE")
    
    # Lưu ra file để chấm điểm
    with open("rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu rag_results.json")