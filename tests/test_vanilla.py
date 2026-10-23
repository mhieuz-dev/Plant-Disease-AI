import requests
import json

BASE_URL = "http://localhost:8000"

# 20 câu hỏi - KHÔNG có prefix class để RAG không chạy (vanilla mode)
questions = [
    "Bệnh gỉ sắt cà phê dùng thuốc gì và phun liều lượng bao nhiêu?",
    "Bệnh gỉ sắt cà phê phòng ngừa như thế nào?",
    "Sâu vẽ bùa cà phê xuất hiện vào mùa nào và xử lý ra sao?",
    "Sâu vẽ bùa cà phê dùng thuốc gì để trị?",
    "Bệnh đốm lá Phoma cà phê có triệu chứng gì?",
    "Bệnh đốm lá Phoma cà phê điều trị bằng thuốc nào?",
    "Khi nào nên phun thuốc phòng bệnh gỉ sắt cho cà phê?",
    "Bệnh đạo ôn lá lúa dùng thuốc gì và phun mấy lần?",
    "Bệnh đạo ôn lá lúa phòng ngừa bằng cách nào?",
    "Bệnh cháy bìa lá lúa do nguyên nhân gì gây ra?",
    "Bệnh cháy bìa lá lúa điều trị như thế nào?",
    "Bệnh khô vằn lúa dùng thuốc gì để xử lý?",
    "Bệnh khô vằn lúa phát triển mạnh trong điều kiện thời tiết nào?",
    "Liều lượng phun thuốc trị đạo ôn lá lúa là bao nhiêu?",
    "Bệnh chết nhanh hồ tiêu do nấm gì gây ra và điều trị thế nào?",
    "Bệnh chết nhanh hồ tiêu phòng ngừa bằng cách nào?",
    "Bệnh thán thư hồ tiêu có triệu chứng gì và dùng thuốc gì?",
    "Bệnh chết chậm hồ tiêu khác bệnh chết nhanh ở điểm nào?",
    "Bệnh chết chậm hồ tiêu điều trị như thế nào?",
    "Khi phát hiện bệnh chết nhanh hồ tiêu cần làm gì ngay lập tức?",
]

def run_vanilla():
    print(f"\n{'='*60}")
    print("CHẠY: VANILLA LLM (không RAG)")
    print('='*60)
    results = []

    # Mỗi câu tạo session riêng để không bị ảnh hưởng lẫn nhau
    for i, question in enumerate(questions, 1):
        r = requests.post(f"{BASE_URL}/conversation/new")
        conv_id = r.json()["conversation_id"]

        r = requests.post(f"{BASE_URL}/chat", json={
            "conversation_id": conv_id,
            "question": question  # Không có prefix class -> RAG không chạy
        })
        data = r.json()
        answer    = data.get("answer", "")
        citations = data.get("citations", [])

        print(f"\nCâu {i}: {question}")
        print(f"Trả lời: {answer[:400]}")
        print(f"Citations: {citations}")
        print("-"*40)

        results.append({
            "q": i,
            "question": question,
            "answer": answer,
            "citations": citations,
        })

    return results

if __name__ == "__main__":
    results = run_vanilla()
    with open("vanilla_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu vanilla_results.json")
