# retico-hftranslate
Local translation module based on the hugging face transformer library.

## Example

```python
from retico_core import *
import retico_wav2vecasr
import retico_speechbraintts
import retico_hftranslate

msg = []


def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == UpdateType.ADD:
            msg.append(x)
        if ut == UpdateType.REVOKE:
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed
    print(" " * 80, end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("")


m1 = audio.MicrophoneModule()
m2 = retico_wav2vecasr.Wav2VecASRModule(language="de")
m3 = retico_hftranslate.HFTranslateModule(from_lang="de", to_lang="en")
m5 = retico_speechbraintts.SpeechBrainTTSModule(language="en")
m6 = audio.SpeakerModule(rate=22050)
m4 = debug.CallbackModule(callback)

m1.subscribe(m2)
m2.subscribe(m3)
m3.subscribe(m4)
m3.subscribe(m5)
m5.subscribe(m6)

network.run(m1)

print("Press any key to exit")
input()

network.stop(m1)
```
