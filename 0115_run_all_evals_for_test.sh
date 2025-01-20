#!/usr/bin/env bash
###############################################################################
# 이 스크립트는 FAD, CLAP, FVD, CLIP 평가를
# 순차적으로 실행하고 결과를 하나의 JSON 파일로 통합하여 저장합니다.
###############################################################################

# BASE_1_28s
# BASE_7_68s
MODEL_TYPE="BASE_7_68s" 

# /workspace/dataset/0114_BASE_1_28s
# /workspace/dataset/0114_BASE_7_68s
PRED_BASE="/workspace/dataset/0114_BASE_7_68s"

# /workspace/dataset/0115_vggsound_sparse_test_random_128s_16f
# /workspace/dataset/0115_vggsound_sparse_test_random_768s_96f
TARGET_RANDOM="/workspace/dataset/0115_vggsound_sparse_test_random_768s_96f"

DATE="0115"
DEVICE="cuda:0"




FINAL_JSON="${DATE}_${MODEL_TYPE}_all_results.json"


# JSON 생성에 사용할 임시 파일
TEMP_JSON=$(mktemp)
# 스크립트 종료 시 임시 파일 삭제
trap 'rm -f "$TEMP_JSON"' EXIT

# 초기 JSON 구조 생성
echo '{}' | jq '.' > "$TEMP_JSON"

###############################################################################
# JSON에 데이터 추가 함수
###############################################################################
add_to_json() {
    local json_path=$1
    local jq_filter=$2
    jq "$jq_filter" "$json_path" > "${json_path}.tmp" && mv "${json_path}.tmp" "$json_path"
}

###############################################################################
# 오디오/비디오 평가 함수 (기존 run_audio_eval, run_video_eval 통합 예시)
###############################################################################
run_evaluation() {
    local type=$1           # "audio" 또는 "video"
    local preds_folder=$2   # 예측 폴더
    local target_folder=$3  # 정답 폴더
    local metrics=($4)      # 메트릭 (예: "CLAP FAD" / "fvd clip")
    local results_file=$5   # 결과 파일 이름
    local key=$6            # JSON에 들어갈 key (curated / random)
    local device=$7

    echo "${type^} 평가 실행: $key..."

    if [[ "$type" == "audio" ]]; then
        python run_audio_eval.py \
            --preds_folder "$preds_folder" \
            --target_folder "$target_folder" \
            --metrics "${metrics[@]}" \
            --results_file "$results_file" \
            --device "$device"

        # 결과 파싱
        local clap_avg=$(awk -F'[=,]' '/CLAP: Average/ {print $2}' "$results_file" | xargs)
        local clap_std=$(awk -F'[=,]' '/CLAP: Average/ {gsub(/Std/, "", $3); print $3}' "$results_file" | xargs)
        local fad=$(awk '/^FAD:/ {print $2}' "$results_file")

        # JSON에 추가
        add_to_json "$TEMP_JSON" \
            ".audio += { \"$key\": { CLAP_avg: \"$clap_avg\", CLAP_std: \"$clap_std\", FAD: \"$fad\" } }"

    elif [[ "$type" == "video" ]]; then
        python run_video_eval.py \
            --preds_folder "$preds_folder" \
            --target_folder "$target_folder" \
            --metrics "${metrics[@]}" \
            --results_file "$results_file" \
            --device "$device"

        # 결과 파싱
        local fvd=$(awk -F':' '/^FVD:/ {print $2}' "$results_file" | xargs)
        local clip_avg=$(awk -F'[=,]' '/CLIP: Average/ {print $2}' "$results_file" | xargs)
        local clip_std=$(awk -F'[=,]' '/CLIP: Average/ {gsub(/Std/, "", $3); print $3}' "$results_file" | xargs)

        # JSON에 추가
        add_to_json "$TEMP_JSON" \
            ".video += { \"$key\": { FVD: \"$fvd\", CLIP_avg: \"$clip_avg\", CLIP_std: \"$clip_std\" } }"
    fi
}


# (2) Random
RANDOM_KEY="random"

RANDOM_AUDIO_PREDS="$PRED_BASE/audio"
RANDOM_AUDIO_TARGET="$TARGET_RANDOM/audio"
RANDOM_AUDIO_METRICS="CLAP FAD"
RANDOM_AUDIO_RESULTS="${DATE}_random_audio_eval_${MODEL_TYPE}_50steps.txt"

RANDOM_VIDEO_PREDS="$PRED_BASE/video"
RANDOM_VIDEO_TARGET="$TARGET_RANDOM/video"
RANDOM_VIDEO_METRICS="fvd clip"
RANDOM_VIDEO_RESULTS="${DATE}_random_video_eval_${MODEL_TYPE}_50steps.txt"

###############################################################################
# 실제 평가 실행
###############################################################################


# (2) Random 오디오 평가
run_evaluation "audio" \
    "$RANDOM_AUDIO_PREDS" \
    "$RANDOM_AUDIO_TARGET" \
    "$RANDOM_AUDIO_METRICS" \
    "$RANDOM_AUDIO_RESULTS" \
    "$RANDOM_KEY" \
    "$DEVICE"

# (4) Random 비디오 평가
run_evaluation "video" \
    "$RANDOM_VIDEO_PREDS" \
    "$RANDOM_VIDEO_TARGET" \
    "$RANDOM_VIDEO_METRICS" \
    "$RANDOM_VIDEO_RESULTS" \
    "$RANDOM_KEY" \
    "$DEVICE"

###############################################################################
# 최종 JSON 파일 저장
###############################################################################
cp "$TEMP_JSON" "$FINAL_JSON"
rm "$TEMP_JSON"

echo ""
echo "------------------------------------------------------------"
echo "✅ 모든 평가가 완료되었습니다! 통합된 결과는 다음 파일에 저장되었습니다:"
echo "   --> $FINAL_JSON"
echo "------------------------------------------------------------"