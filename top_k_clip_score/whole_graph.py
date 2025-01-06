import json
import matplotlib.pyplot as plt

# JSON 파일 경로 리스트
json_files = [
    "vggsound_sparse_test_clip_topk_results.json", 
    "vggsound_sparse_test_clip_6s_topk_results.json", 
    "vggsound_sparse_test_clip_3s_topk_results.json"
]

# 범례 이름 리스트
legends = ["10s", "6s", "3s"]

# 그래프 그리기
plt.figure(figsize=(10, 6))

for i, json_file in enumerate(json_files):
    with open(json_file, "r") as f:
        data = json.load(f)
        avg_topk_scores = data.get("avg_topk_scores", [])
        scores = list(range(1, len(avg_topk_scores) + 1))  # x축
        plt.plot(scores, avg_topk_scores, label=legends[i])  # 범례 추가

# 그래프 스타일 설정
plt.xlabel("k", fontsize=12)
plt.ylabel("Average Top-k Scores", fontsize=12)
plt.title("Avg Top-k Scores for CLIP score", fontsize=14)
plt.legend(loc="best", fontsize=10)  # 범례 추가
plt.grid(True)

# 그래프 저장 및 표시
plt.savefig("avg_topk_scores_plot_with_legend.png")
plt.show()
