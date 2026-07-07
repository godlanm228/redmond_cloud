import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.utils = types.SimpleNamespace(quote=lambda s: s)
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

from logic.news_personalization import NewsCandidate, rank_personal_news


class PersonalNewsRankingTests(unittest.TestCase):
    def test_finance_crypto_ai_rank_above_general_noise(self):
        candidates = [
            NewsCandidate(
                "Bitcoin and ether ETF inflows rise as Fed rate hopes lift crypto",
                "https://coindesk.example/btc-etf",
                "",
                "CoinDesk",
                "crypto",
            ),
            NewsCandidate(
                "OpenAI releases new developer agent tools on GitHub",
                "https://techcrunch.example/openai-agent",
                "",
                "TechCrunch AI",
                "ai",
            ),
            NewsCandidate(
                "Football transfer recap and weekend watch guide",
                "https://sport.example/football",
                "",
                "BBC Sport",
                "sport",
            ),
        ]

        picks = rank_personal_news(candidates, limit=3)

        self.assertEqual([p.item.category for p in picks], ["crypto", "ai"])
        self.assertIn("крипта", picks[0].reason)

    def test_excludes_already_used_digest_urls(self):
        candidates = [
            NewsCandidate("Bitcoin ETF inflows rise", "https://x/btc", "", "CoinDesk", "crypto"),
            NewsCandidate("Unity adds new multiplayer tooling", "https://x/unity", "", "Game Developer", "gamedev"),
        ]

        picks = rank_personal_news(candidates, exclude_urls={"https://x/btc"})

        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0].item.url, "https://x/unity")


class DigestPersonalSectionTests(unittest.TestCase):
    def test_digest_adds_personal_section_without_duplicate_section_urls(self):
        from logic import digest

        old_collect = digest.collect_headlines
        old_translate = digest._translate_titles
        data = {
            "world": [("World headline", "https://x/world", "", "BBC World")],
            "finance": [
                ("Markets edge higher on Fed rate hopes", "https://x/finance-main", "", "CNBC"),
                ("Nasdaq earnings preview for AI chip stocks", "https://x/finance-ai", "", "CNBC"),
            ],
            "crypto": [
                ("Bitcoin ETF inflows rise again", "https://x/crypto-main", "", "CoinDesk"),
                ("Ethereum staking regulation update", "https://x/eth", "", "CoinDesk"),
            ],
            "tech": [
                ("Consumer gadget deal roundup", "https://x/gadget", "", "The Verge"),
                ("OpenAI developer agent release reaches GitHub", "https://x/openai", "", "TechCrunch"),
            ],
            "ai": [
                ("Anthropic ships new coding agent for developers", "https://x/agent", "", "TechCrunch AI"),
            ],
            "gamedev": [
                ("Unity engine update improves multiplayer workflows", "https://x/unity", "", "Game Developer"),
            ],
            "sport": [("Sport headline", "https://x/sport", "", "BBC Sport")],
        }

        try:
            digest.collect_headlines = lambda category, limit: data.get(category, [])[:limit]
            digest._translate_titles = lambda titles: titles

            out = digest.build_digest()
        finally:
            digest.collect_headlines = old_collect
            digest._translate_titles = old_translate

        self.assertIn("Может быть тебе интересно", out)
        self.assertIn("https://x/eth", out)
        self.assertNotIn("https://x/crypto-main) · CoinDesk —", out)


if __name__ == "__main__":
    unittest.main()
