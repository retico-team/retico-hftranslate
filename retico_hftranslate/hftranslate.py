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
        self.latest_input_iu = None

    def setup(self):
        self.hftranslator = HFTranslate(self.from_lang, self.to_lang)

    def shutdown(self):
        self.hftranslator = None

    def current_text(self):
        return " ".join([iu.text for iu in self.current_input])

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.current_input.append(iu)
                self.latest_input_iu = iu
            elif ut == retico_core.UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == retico_core.UpdateType.COMMIT:
                self.commit(iu)

        current_text = self.current_text()
        current_translation = self.hftranslator.translate(current_text)

        um, new_tokens = retico_core.text.get_text_increment(self, current_translation)
        for token in new_tokens:
            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.text = token
            self.current_output.append(output_iu)
            um.add_iu(output_iu, retico_core.UpdateType.ADD)

        if self.input_committed():
            for iu in self.current_output:
                self.commit(iu)
                um.add_iu(iu, retico_core.UpdateType.COMMIT)
            self.current_input = []
            self.current_output = []

        return um
