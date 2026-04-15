"""
Face detection using MTCNN with Haar cascade fallback.
"""

import cv2
from PIL import Image
from configs.settings import FACE_SIZE, FACE_MARGIN, DEVICE


class FaceDetector:
    def __init__(self):
        self.mtcnn = None
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=FACE_SIZE,
                margin=int(FACE_SIZE * FACE_MARGIN),
                keep_all=False,
                device=DEVICE,
                post_process=False,
            )
        except ImportError:
            pass

        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.mtcnn is not None:
            try:
                pil_img = Image.fromarray(rgb)
                tensor = self.mtcnn(pil_img)
                if tensor is not None:
                    face_np = tensor.permute(1, 2, 0).byte().cpu().numpy()
                    return Image.fromarray(face_np)
            except Exception:
                pass

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            mx = int(w * FACE_MARGIN)
            my = int(h * FACE_MARGIN)
            x1, y1 = max(0, x - mx), max(0, y - my)
            x2 = min(rgb.shape[1], x + w + mx)
            y2 = min(rgb.shape[0], y + h + my)
            crop = rgb[y1:y2, x1:x2]
            return Image.fromarray(crop).resize(
                (FACE_SIZE, FACE_SIZE), Image.BILINEAR
            )

        h, w = rgb.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        crop = rgb[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]
        return Image.fromarray(crop).resize(
            (FACE_SIZE, FACE_SIZE), Image.BILINEAR
        )
