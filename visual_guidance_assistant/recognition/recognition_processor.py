"""Thread that runs face recognition on frames."""
import threading
from recognition.face_recognizer import FaceRecognizer

class RecognitionProcessor(threading.Thread):
    def __init__(self, frame_queue, tracking_queue, enriched_tracking_queue, face_recognizer: FaceRecognizer, config):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.tracking_queue = tracking_queue
        self.enriched_tracking_queue = enriched_tracking_queue
        self.recognizer = face_recognizer
        self.running = False
        self.latest_frame = None
        self.latest_tracks = []
        self.frame_skip = 5
        self.frame_count = 0
    
    def stop(self):
        self.running = False
    
    def run(self):
        self.running = True
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                self.latest_frame = frame
            except:
                pass
            
            try:
                tracks = self.tracking_queue.get(timeout=0.5)
                self.latest_tracks = tracks
            except:
                pass
            
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0 and self.latest_frame is not None:
                enriched = self.recognizer.recognize_faces(self.latest_frame, self.latest_tracks)
                try:
                    self.enriched_tracking_queue.put_nowait(enriched)
                except:
                    pass
