LABEL_GROUPS = {
    "기쁨": ["기쁨", "편안함", "농담"],
    "설렘": ["설렘", "애정"],
    "실망": ["서운함", "실망"],
    "후회": ["후회"],
    "슬픔": ["슬픔", "미안함"],
    "짜증": ["짜증", "화남", "질투"],
    "불안": ["불안", "의심"],
    "중립": ["중립"],
}

# 🔁 역방향 매핑 생성 (모델 감정 → 사용자표시 감정)
LABEL_DISPLAY_MAP = {
    sub_label: main_label
    for main_label, sub_list in LABEL_GROUPS.items()
    for sub_label in sub_list
}