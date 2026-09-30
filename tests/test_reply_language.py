"""The owner writes in German or English → the prompt names the reply language.

30.09.2026: a German message got a Russian reply; the generic system rule lost
to the Russian chat history and the Russian understanding block.
"""
import unittest

from logic.response_generator import _reply_language_hint


class ReplyLanguageHintTest(unittest.TestCase):
    def test_german_message_gets_the_hint(self):
        hint = _reply_language_hint("Heute 2 h für Statistik gelernt, morgen geht's ins Gym")
        self.assertIn("по-немецки", hint)

    def test_english_message_gets_the_hint(self):
        self.assertTrue(_reply_language_hint("remind me about the exam tomorrow"))

    def test_russian_message_has_no_hint(self):
        self.assertEqual(_reply_language_hint("Сегодня 2 часа учил статистику, завтра в зал"), "")

    def test_russian_message_with_link_and_tickers_has_no_hint(self):
        text = "глянь что там по BTC и ETH https://www.coingecko.com/en/coins/bitcoin"
        self.assertEqual(_reply_language_hint(text), "")

    def test_short_latin_word_has_no_hint(self):
        self.assertEqual(_reply_language_hint("ok"), "")


if __name__ == "__main__":
    unittest.main()
