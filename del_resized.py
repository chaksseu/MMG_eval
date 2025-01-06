import os

def remove_resized_suffix(directory):
    # 지정된 디렉토리 내의 모든 파일을 순회
    for filename in os.listdir(directory):
        # 파일이 .mp4 확장자를 가지고 있는지 확인
        if filename.endswith("_resized.mp4"):
            # 새로운 파일명 생성 (예: video_resized.mp4 -> video.mp4)
            new_filename = filename.replace("_resized", "")
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_filename)
            try:
                os.rename(src, dst)
                print(f"Renamed: '{filename}' -> '{new_filename}'")
            except Exception as e:
                print(f"Failed to rename '{filename}': {e}")

if __name__ == "__main__":
    # 대상 디렉토리 경로를 입력하세요
    target_directory = "/workspace/dataset/vggsound_sparse_test_origin_32s_40f_256/vggsound_sparse_test_random_32s_40frames_256/video"  # 예: "C:/Users/Username/Videos"

    # 디렉토리가 존재하는지 확인
    if os.path.isdir(target_directory):
        remove_resized_suffix(target_directory)
    else:
        print(f"Directory does not exist: {target_directory}")
