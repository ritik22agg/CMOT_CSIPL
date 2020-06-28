"""
Created on Wed Jun 24
@author: divya
PyLint Score: 10/10
"""
# install tesseract first
# from here
# https://github.com/tesseract-ocr/tesseract/wiki#installation
# then install pillow
# pip install Pillow
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr(filename):
    """
    We'll use Pillow's Image class to open the image
    and use pytesseract to detect the string in the image
    """
    text = pytesseract.image_to_string(Image.open(filename))
    #print(f'Extracted from image {text}')
    return text