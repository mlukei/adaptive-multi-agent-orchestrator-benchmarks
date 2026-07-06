import os
import fire
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance


def _preprocess(img: Image.Image) -> Image.Image:
    """Upscale small images and sharpen contrast before OCR."""
    # Convert to greyscale — Tesseract works better on grey/binary
    img = img.convert("L")
    # Upscale if too small — Tesseract degrades sharply below ~300 DPI equivalent
    MIN_DIM = 1000
    w, h = img.size
    if max(w, h) < MIN_DIM:
        scale = MIN_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Sharpen and boost contrast
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img

# DEMO = (
#     'You can recognize an image by calling `ocr_recognize_file` with 1 argument.\n'
#     '1. file_path: the path to the image file.\n'
#     "You can call it by generating command: {'app': 'ocr', 'action': 'ocr_recognize_file', 'file_path': ...}"
# )

DEMO = (
    "recognize the text from an image file: "
    "{'app': 'ocr', 'action': 'recognize_file', 'file_path': [THE_PATH_TO_THE_IMAGE_FILE]}"
)

def construct_action(work_dir, args: dict, py_file_path='/apps/ocr_app/ocr_recognize_file.py'):
    import shlex
    return f'python3 {py_file_path} --file_path {shlex.quote(str(args["file_path"]))}'

def ocr_recognize_file(file_path):
    try:
        img = Image.open(file_path)
        img = _preprocess(img)
        # PSM 6 = assume a single uniform block of text (good for tables/lists)
        # OEM 1 = LSTM engine (most accurate)
        custom_config = r"--oem 1 --psm 6"
        text = pytesseract.image_to_string(img, config=custom_config)
    except:
        text = None
    return text

def main(file_path, debug=False):
    if not os.path.exists(file_path):
        return f'OBSERVATION: The file {file_path} does not exist. Failed to recognize text.'
    
    text = ocr_recognize_file(file_path)
    if debug:
        print(text)
    if text:
        observation = f'OBSERVATION: The text from {file_path} is:\n{text}'
    else:
        observation = f'OBSERVATION: Failed to recognize text from {file_path}'
    return observation


if __name__ == '__main__':
    fire.Fire(main)
