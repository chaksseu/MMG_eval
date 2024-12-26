import re
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_sentence(filename):

    # 파일 이름에서 밑줄을 공백으로 바꾸고 확장자를 제거
    sentence = filename.replace('_', ' ').replace('.wav', '')
    
    # 제외할 단어 목록
    excluded_words = ['batch', 'proc', 'sample', 'audio']
    
    # 제외할 단어 패턴 생성 (단어 뒤에 숫자가 올 수 있도록 수정)
    pattern_words = r'\b(?:' + '|'.join(excluded_words) + r')\d*\b'
    # 숫자 패턴 (단독으로 있는 숫자 제거)
    pattern_numbers = r'\b\d+\b'
    
    # 제외할 단어 제거
    sentence = re.sub(pattern_words, '', sentence, flags=re.IGNORECASE)
    
    # 숫자 제거
    sentence = re.sub(pattern_numbers, '', sentence)
    
    # 불필요한 공백 제거
    sentence = ' '.join(sentence.split())
    
    return sentence

def get_text_data(filenames):
    """
    주어진 파일 이름 리스트에서 최종 텍스트 데이터를 생성합니다.

    매개변수:
    -- filenames : 리스트, 오디오 파일의 이름 리스트.

    반환값:
    -- 텍스트 데이터 리스트.
    """
    all_sentences = set()  # 중복 제거를 위해 집합 사용
    
    for filename in filenames:
        sentence_clean = clean_sentence(filename)
        print(sentence_clean)
    

# 예제 파일 이름 리스트
filenames = [
    "batch_proc_sample_123_audio.wav",
    "batch0_sample3_dog barking_audio.wav",
    "batch7_sample1_dog barking_audio.wav",
    "dog_barking_batch_0_proc_0.wav",
    "dog_barking_batch_90_proc_2.wav",
    "ice_cracking_batch_26_proc_0.wav",
    "lions_roaring_batch_41_proc_5.wav",
    "people_eating_apple_batch_58_proc_1.wav",
    "people_sneezing_batch_152_proc_7.wav",
    "playing_tennis_batch_114_proc_0.wav",
    "skateboarding_batch_125_proc_5.wav",
    "batch5_sample2_lions roaring_audio.wav",
    "batch8_sample2_playing badminton_audio.wav"
]

# 최종 텍스트 데이터 생성
get_text_data(filenames)
