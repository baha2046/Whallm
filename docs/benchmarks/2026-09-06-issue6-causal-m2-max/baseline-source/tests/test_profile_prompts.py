from __future__ import annotations

import unittest

from Scripts.prepare_r0_prompts import build_prompt, encode_prompt


class CharacterTokenizer:
    bos_token = None

    def encode(self, text, add_special_tokens):
        tokens = [ord(character) for character in text]
        return [0, *tokens] if add_special_tokens else tokens

    def decode(self, tokens, **_options):
        return "".join(chr(token) for token in tokens)


class ProfilePromptTests(unittest.TestCase):
    def test_build_prompt_has_the_requested_runtime_token_count(self):
        tokenizer = CharacterTokenizer()

        prompt, tokens = build_prompt(tokenizer, "ab", 32)

        self.assertEqual(tokens, encode_prompt(tokenizer, prompt))
        self.assertEqual(len(tokens), 32)


if __name__ == "__main__":
    unittest.main()
