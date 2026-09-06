# Guide YOU

A companion and support tool for someone living with early-stage cognitive
decline, and for the people who care for them. It watches through a webcam,
listens for speech, recognises enrolled faces, and talks back.

**It is not a medical device.** It does not diagnose, assess, monitor health, or
give medical advice of any kind, and it must never be presented as doing so. Any
question about medication, symptoms, or treatment is deliberately redirected to a
carer or doctor and written to a log — it is never answered.

---

## ⚠️ Read this before enrolling anyone: your data is tied to this Windows account

Enrolled faces, voice prints, caregiver contact notes, and the alert log are
**encrypted at rest**, with the key bound to **this Windows user account on this
computer**.

**If any of the following happen, that data is permanently unreadable and
everyone must be enrolled again:**

- Windows is reinstalled
- The device is replaced, or the files are moved to another computer
- You switch to a different Windows user account
- The Windows user profile is reset or deleted

There is no passphrase, no recovery key, and no back door — not for you, and not
for anyone else either. That is the point. If you need the data to survive a
machine change, re-enrol on the new machine; copying the files across will not
work.

### What the encryption does and does not protect

| Protected against | Not protected against |
|---|---|
| The laptop being lost or stolen | Someone already signed in to this Windows account |
| The drive being removed and read elsewhere | |
| The files being copied off, or synced to cloud backup | |
| A different Windows account on the same machine | |

### Why there is no startup passphrase

A passphrase would be stronger, but it would make the device **fail closed**.
After a power cut, a crash, or a Windows update reboot, the app would sit at a
prompt that the person it supports cannot answer — so the help command, the
distress detection, and the whole safety net would be dead until a carer
arrived. That is exactly when they matter most. Availability is part of safety
here, so the key comes from the machine rather than from a human.

### Files holding personal data

| File | Contents | Encrypted |
|---|---|---|
| `known_faces/profiles.json` | Face encodings, voice prints, roles, contact notes | Yes |
| `alerts/alert_log.json` | Help requests, distress moments, deflected medical questions | Yes |
| `known_faces/*.jpg` | Reference photos from enrolment | **No** — plain images |
| `memory/memory_data.json` | Conversation history | **No** — still plaintext |
| `debug/last_capture.wav` | The most recent microphone recording | **No** — set `voice_input.debug_save_wav: false` to stop writing it |

None of these are committed to git. The last three are known plaintext gaps, not
oversights — worth closing if the device ever leaves the home.

---

## Setup

```powershell
python -m pip install -r visual_guidance_assistant/requirements.txt
python -m pip install --no-deps face_recognition
python -m pip install Click Pillow
```

`face_recognition` needs `--no-deps` because it declares a plain `dlib`
dependency; building real dlib on Windows needs Visual C++, and when it fails
pip rolls back **every other package in the same command**. `dlib-bin` (in
requirements.txt) is the prebuilt wheel and provides the module at runtime.

### Set your API key — use the `.env` file

Create a file called `.env` **in the project root** (next to this README):

```
ANTHROPIC_API_KEY=sk-ant-...
```

That's it. It's already excluded by `.gitignore`, it survives reboots, and the
app reads it however you launch it.

<details>
<summary>Why not <code>setx</code>? (worth knowing — it cost us an evening)</summary>

`setx ANTHROPIC_API_KEY "sk-ant-..."` writes the key to the Windows registry,
but a process only ever sees the environment block its **parent** handed it.
A terminal inside VS Code inherits VS Code's environment; VS Code inherited
Explorer's. If any ancestor in that chain started before the key was set, the
app sees nothing — and "just open a new terminal" does **not** fix it, because
the new terminal is still a child of the same stale editor.

The symptom is the assistant replying *"I can't reach my thinking service at the
moment"* to everything, while `echo %ANTHROPIC_API_KEY%` in some other window
shows the key perfectly. The `.env` file avoids the whole problem.

An environment variable, if genuinely present, still takes precedence over the
file — so `setx` continues to work if you prefer it, once the process tree has
actually been restarted.
</details>

Run:

```powershell
python visual_guidance_assistant\app.py
```

### One-time model download

```powershell
.venv\Scripts\python.exe visual_guidance_assistant\fetch_models.py
```

Fetches the YuNet face detector (~230KB) into `visual_guidance_assistant/models/`.
**This runs once.** The app never downloads anything at runtime — a care device
should not depend on the internet being reachable when somebody walks into the
room.

The model file is committed to the repository, so a fresh clone already has it
and works offline. You only need `fetch_models.py` if the file has been deleted,
or to update it. To install it by hand on a machine with no internet, copy
`models/face_detection_yunet_2023mar.onnx` from any other checkout — it is
fetched from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

**If the file is missing the app does not fail.** It prints one warning line and
falls back to dlib's HOG detector, which needs no model. Everything still works,
just far less reliably — see the numbers below.

### Why YuNet, measured on this hardware

Same 10 frames, same camera, identical pixels to both detectors:

| | YuNet | dlib HOG | dlib CNN |
|---|---|---|---|
| Located a face | **10/10** | 2/10 | 5/5 |
| Matched the enrolled person | **9/10** | 1/10 | 5/5 |
| Time per frame | **0.067s** | 0.249s | **72–78s** |

HOG failing to *locate* the face — not failing to match it — is why enrolment
appeared to succeed while "who am I" never worked. When HOG did find a face the
distance was a comfortable 0.38; it simply could not find one 9 times in 10.
dlib's CNN model is accurate but takes over a minute per frame on this CPU,
which is unusable for a live assistant.

After switching, the full cycle went from **0/10 to 10/10**.

## Configure before use

In `visual_guidance_assistant/config/settings.yaml`:

- **`care.home_location`** — set this. It is the only way the assistant can
  answer "where am I". Left blank it will honestly say it cannot tell, which is
  correct but less useful. It will **never** guess a location.
- `camera.index` — run `python visual_guidance_assistant\list_cameras.py` to see
  which index is which physical camera.

## Enrolling people

Say **"remember my face"**. You will be asked for a name, whether the person is
the one being supported (`patient`) or someone who helps (`caregiver`), and for
caregivers a relationship and contact note. Then the assistant asks out loud for
a short spoken phrase to learn the voice.

Anything not clearly "caregiver" is treated as `patient`, because that is the
gentler mode and the safe default when the answer is unclear.

## Teaching it to recognise family and friends

Hold a photo of someone up to the camera — on a phone screen is fine — and say
**"remember this person"**. You will be asked for their name and how they are
related. When they next visit in person, the assistant recognises them and can
greet them by name.

If it cannot see a face clearly it says so and tells you what to change
("hold it steadier, move it closer, tilt it to stop the screen glaring") rather
than failing quietly. Photographing a screen adds glare and moire that a live
face does not have, so expect to need a second go sometimes.

### ⚠️ This enrols someone who is not in the room

Photo enrolment records a person's face **without them being present, and
without them agreeing to it at that moment.** That is the point of the feature —
you cannot ask a daughter who lives two hours away to sit in front of the camera
— but it is worth being deliberate about.

What that face data does and does not do:

| | |
|---|---|
| Stays on this computer, encrypted | ✅ always |
| Used to greet them by name when they visit | ✅ that is the whole feature |
| Sent anywhere, ever | ❌ never — see "Alerts are logged locally" below |
| Grants any access to the system | ❌ **never** |

**A photo-enrolled person is granted nothing.** They cannot reach caregiver
mode, cannot have the alert log read to them, and are never treated as an
emergency contact — even if you enrol a photo of an actual carer. A photo is the
weakest identity claim in the system: it can be of anyone, held up by anyone. So
it buys a friendly greeting and nothing else. Caregiver access requires an
in-person enrolment via "remember my face".

Our suggestion, not a technical control: tell people you have added them. Most
will be glad the assistant knows who they are; being recorded without being told
is a different thing.

To remove someone, delete `known_faces/profiles.json` and re-enrol whoever
should stay. There is no per-person delete command yet.

## Voice commands

| Say | What happens |
|---|---|
| "remember my face" | Enrol or re-enrol whoever is in view, as a system user |
| "remember this person" / "add family member" | Enrol someone from a photo, to be recognised and greeted — grants no access |
| "who do you know" | Reads back system users and recognised people, separately |
| "what day is it" / "what time is it" | Answered from the system clock, identically every time |
| "where am I" | Answered from `care.home_location`, or an honest "I can't tell" |
| "I need help" / "call \<name\>" | Calm confirmation, and a logged alert |
| "read me today's alerts" | **Caregivers only**, and only when confirmed by face |

## Asking for help does not summon anyone

Requests are sorted into three tiers, and the bar for the top one is
deliberately high:

| Tier | Examples | What happens |
|---|---|---|
| **Everyday** | "help me find my glasses", "I need help going to the shop", "what should I have for lunch" | Answered in conversation. **Nothing logged, nobody contacted.** |
| **Distress** | "I don't know where I am", "I'm scared", "where is Sarah" | Warm reassurance, logged for a carer to read later |
| **Emergency** | "I've fallen", "chest pain", "call an ambulance", "call Sarah now" | Immediate carer-alert response, logged |

Anything ambiguous — including a bare "I need help" — is treated as **everyday**
and simply answered ("Of course, I'm here to help. What do you need?"). This is
on purpose. An alert that fires when someone asks for help finding the remote is
an alert that gets ignored, and then it is not there when somebody has actually
fallen.

Measured on the current rules: 0 false alarms across 10 everyday phrasings,
0 missed emergencies across 11, 0 distress misclassifications across 6.

## How it decides who it is talking to

Face recognition is primary. A voice match can supply a name but **can never on
its own grant caregiver mode** — the voice fingerprint is a weak signal, and the
two failure directions are not equally bad:

- a caregiver wrongly in patient mode gets a warmer, shorter answer. Harmless.
- a patient wrongly in caregiver mode gets medical questions answered instead of
  redirected. Not harmless.

When nothing is confident, patient mode is the default for the same reason.

**Sticky identity.** No detector finds a face in every single frame — people
look away, tilt their head, move. Without help, one missed frame drops the
person back to "unknown", and the assistant stops knowing who it is talking to
mid-conversation. So the last confident identification is held for
`face_recognition.sticky_seconds` (default 60). A frame that positively
identifies somebody *else* is believed immediately, so this can never keep
asserting the wrong person once the camera can actually see who is there. It
stacks with whichever detector is active, and matters most when the model file
is missing and the weaker HOG fallback is doing the work.

## Alerts are logged locally and sent nowhere

Every help request, distress phrase, and deflected medical question is written to
`alerts/alert_log.json` with `notified: false`. **Nothing is sent to anyone** —
no SMS, no email, no call. Outbound notification needs real credentials and an
informed conversation with the person being supported about who gets told what,
so it is deliberately not built. `safety/alert_log.py → notify()` is the single
place it would hook in.

This means **a logged alert is only seen if someone looks**. Ask "read me today's
alerts" as a caregiver, or open the file.

## Voice recognition is not calibrated

The speaker fingerprint (`voice_id/voice_embedding.py`) is MFCC statistics in
pure numpy. It is a supplementary hint, not an identifier: it drifts with
microphone position, room acoustics, illness and tiredness, and similar-sounding
family members can score close together. The match threshold is set
conservatively high and **has not been calibrated against two real enrolled
voices**. Check the scores before relying on it.
