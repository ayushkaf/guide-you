Place images of known people here. Name files as person's name (e.g., john.jpg). One face per image. Clear frontal face photos work best.

2026-09-05: profiles.json in this folder cannot be decrypted on this Windows
account/machine (it's DPAPI-encrypted, bound to whoever originally wrote it --
its internal "updated" timestamp is 2026-08-08, weeks before these photos were
copied here on 2026-09-04, so it came from a different account/machine during
a project migration and is not recoverable here; see utils/secure_store.py).
With no live profile data readable, the reference photos in this folder are
currently orphaned -- none of them are backed by a working profile record.
Tom.jpg and Sarah.jpg were found to be byte-for-byte identical files (same
MD5), which can't reflect two different real people, so they were deleted as
leftover placeholder/test data. The remaining photos (Aayush.jpg, Priya.jpg,
bardan.jpg, GuideYouLiveTest.jpg, StepFourTest.jpg) are left in place but are
equally orphaned -- anyone who needs to actually use this app will need to be
re-enrolled through it (e.g. "remember my face" / "remember this person"),
which will capture a fresh, correctly-associated photo and profile.
