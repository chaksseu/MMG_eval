import os
import csv

# 기본 디렉토리 설정
base_dir = "/workspace/dataset/1226_output_BASE_epoch90_50steps"
audio_dir = os.path.join(base_dir, "audio")
video_dir = os.path.join(base_dir, "video")

# 오디오 및 비디오 파일 목록 가져오기
audio_files = set([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
video_files = set([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

# 파일의 베이스 이름 추출 (확장자 제외)
audio_basenames = {os.path.splitext(f)[0] for f in audio_files}
video_basenames = {os.path.splitext(f)[0] for f in video_files}


#print(audio_basenames)

# 오디오와 비디오에서 공통으로 존재하는 베이스 이름 찾기
matching_basenames = audio_basenames.intersection(video_basenames)

# CSV 파일 경로 설정
csv_path = os.path.join(base_dir, "file_pairs.csv")

# CSV 파일 작성
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 헤더 작성
    # writer.writerow(["video_path", "audio_path"])
    
    # 매칭된 파일들 쌍으로 작성
    for basename in sorted(matching_basenames):
        video_path = os.path.join(video_dir, basename + ".mp4")
        audio_path = os.path.join(audio_dir, basename + ".wav")
        writer.writerow([video_path, audio_path])

print(f"CSV 파일이 성공적으로 생성되었습니다: {csv_path}")
