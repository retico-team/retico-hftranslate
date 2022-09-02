import transformers

transformers.logging.set_verbosity_error()
from transformers import pipeline

import retico_core


class HFTranslate:

    TRANSLATION_MAP = {
        "en_fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr_en": "Helsinki-NLP/opus-mt-fr-en",
        "en_de": "Helsinki-NLP/opus-mt-en-de",
        "de_en": "Helsinki-NLP/opus-mt-de-en",
        "es_en": "Helsinki-NLP/opus-mt-es-en",
        "en_es": "Helsinki-NLP/opus-mt-en-es",
        "fr_de": "Helsinki-NLP/opus-mt-fr-de",
        "de_fr": "Helsinki-NLP/opus-mt-de-fr",
    }

    def __init__(self, from_lang="en", to_lang="de"):
        self.from_lang = from_lang
        self.to_lang = to_lang

        tr_key = f"{from_lang}_{to_lang}"
        if not self.TRANSLATION_MAP.get(tr_key):
            raise ValueError(f"Cannot translate from {from_lang} to {to_lang}.")
        self.translator = pipeline("translation", model=self.TRANSLATION_MAP[tr_key])

    def translate(self, text):
        translation = self.translator(text)
        return translation[0]["translation_text"]


class HFTranslateModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Hugging Face Translation Module"

    @staticmethod
    def description():
        return (
            "A module that translates between languages using Hugging Face Transfomers."
        )

    @staticmethod
    def input_ius():
        return [retico_core.text.TextIU]

    @staticmethod
    def output_iu():
        return retico_core.text.TextIU

    def __init__(self, from_lang="en", to_lang="de", **kwargs):
        super().__init__(**kwargs)

        self.from_lang = from_lang
        self.to_lang = to_lang
        self.hftranslator = None

        self._latest_text = ""
        self._latest_translation = ""
        self.current_output_ius = []
        self.latest_input_iu = None

    def setup(self):
        self.hftranslator = HFTranslate(self.from_lang, self.to_lang)

    def shutdown(self):
        self.hftranslator = None

    def current_text(self):
        txt = []
        for iu in self.current_ius:
            txt.append(iu.text)
        return " ".join(txt)

    def clean_pipeline(self):
        self.current_ius = []
        self.current_output_ius = []
        self._latest_translation = ""
        self._latest_text = ""

    def process_update(self, update_message):
        final = False
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.current_ius.append(iu)
                self.latest_input_iu = iu
                final = final or iu.committed
            elif ut == retico_core.UpdateType.REVOKE:
                self.revoke(iu)

        current_text = self.current_text()
        if final or current_text != self._latest_text:
            current_translation = self.hftranslator.translate(current_text)
            if current_translation == self._latest_translation:
                if final:
                    output_iu = self.create_iu(self.latest_input_iu)
                    output_iu.committed = True
                    output_iu.text = ""
                    self.clean_pipeline()
                    return retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
                return None

            self._latest_translation = current_translation
            self._latest_text = current_text

            um, new_tokens = self.get_increment(current_translation)
            for token in new_tokens:
                output_iu = self.create_iu(self.latest_input_iu)
                output_iu.text = token
                output_iu.committed = final
                self.current_output_ius.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)

            if final:
                self.clean_pipeline()

            return um

    def get_increment(self, new_text):
        """Compares the full text given by the asr with the IUs that are already
        produced and returns only the increment from the last update. It revokes all
        previously produced IUs that do not match."""
        um = retico_core.UpdateMessage()
        tokens = new_text.strip().split(" ")
        if tokens == [""]:
            return um, []

        new_tokens = []
        iu_idx = 0
        token_idx = 0
        while token_idx < len(tokens):
            if iu_idx >= len(self.current_output_ius):
                new_tokens.append(tokens[token_idx])
                token_idx += 1
            else:
                current_iu = self.current_output_ius[iu_idx]
                iu_idx += 1
                if tokens[token_idx] == current_iu.text:
                    token_idx += 1
                else:
                    current_iu.revoked = True
                    um.add_iu(current_iu, retico_core.UpdateType.REVOKE)
        self.current_output_ius = [
            iu for iu in self.current_output_ius if not iu.revoked
        ]

        return um, new_tokens
