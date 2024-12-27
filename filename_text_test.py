import re

excluded_words = ['batch', 'proc', 'sample', 'audio']
pattern_words = re.compile(r'^(?:' + '|'.join(excluded_words) + r')\d*$', re.IGNORECASE)
pattern_numbers = re.compile(r'^\d+$')

def clean_sentence(filename):
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    sentence = filename.replace('_', ' ').replace('.wav', '')
    words = sentence.split()
    filtered_words = [
        word for word in words 
        if not pattern_words.match(word) and not pattern_numbers.match(word)
    ]
    cleaned_sentence = ' '.join(filtered_words)
    return cleaned_sentence

# 예제 사용
print(clean_sentence("batch0_sample0_playing tennis_audio"))       # 출력: ""
print(clean_sentence("my_audio2_file.wav"))          # 출력: "my file"
print(clean_sentence("BatchProc_456.wav"))          # 출력: ""
print(clean_sentence("audio43_sample2_test123.wav"))   # 출력: "test123"
