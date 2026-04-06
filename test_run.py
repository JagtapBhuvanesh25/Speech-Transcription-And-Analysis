import sys
from diarisation import SpeakerDiarizer
import traceback

d = SpeakerDiarizer()
try:
    print("Starting diarization...")
    t, s = d.diarize(r"uploads\WhatsApp_Ptt_2026-03-20_at_4.16.50_PM.mp3")
    print("Success!")
except Exception as e:
    traceback.print_exc()
