import difflib

known_intents = {
    "productivity_check": [
        "что я делал сегодня",
        "покажи продуктивность",
        "я был продуктивен?"
    ],
    "request_info": [
        "как приготовить рис",
        "найди информацию",
        "что значит это слово"
    ],
    "block_distraction": [
        "не пускай в дискорд",
        "заблокируй ютуб",
        "не давай играть"
    ],
    "greeting": [
        "привет",
        "здорово",
        "даров",
        "я дома"
    ]
}

def match_intent(text: str) -> str:
    text = text.lower()
    for intent, patterns in known_intents.items():
        for pattern in patterns:
            ratio = difflib.SequenceMatcher(None, text, pattern).ratio()
            if ratio > 0.6 or pattern in text:
                return intent
    return "fallback"
