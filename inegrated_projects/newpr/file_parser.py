"""
Created on Mon Jun 22 00:04:39 2020
@author: divya
Parses files added as attachment in email form - pdf, txt, png, jpg and jpeg
PyLint Score: 10/10
"""


from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
#from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
#from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

#for image parsing
from parseimage import ocr

def parse(mail_path):
    """
    extracts text from PDF file
    """
    extracted_text = ""
    # Open and read the pdf file in binary mode
    filep = open(mail_path, "rb")

    # Create parser object to parse the pdf content
    parser = PDFParser(filep)

    # Store the parsed content in PDFDocument object
    document = PDFDocument(parser)

    # Check if document is extractable, if not abort
    if not document.is_extractable:
        #raise PDFTextExtractionNotAllowed
        return ""

    # Create PDFResourceManager object that stores shared resources such as fonts or images
    rsrcmgr = PDFResourceManager()

    # set parameters for analysis
    laparams = LAParams()

    # Create a PDFDevice object which translates interpreted information into desired format
    # Device needs to be connected to resource manager to store shared resources
    # device = PDFDevice(rsrcmgr)
    # Extract the decive to page aggregator to get LT object elements
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)

    # Create interpreter object to process page content from PDFDocument
    # Interpreter needs to be connected to resource manager for shared resources and device
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Ok now that we have everything to process a pdf document, lets process it page by page
    for page in PDFPage.create_pages(document):
        # As the interpreter processes the page stored in PDFDocument object
        interpreter.process_page(page)
        # The device renders the layout from interpreter
        layout = device.get_result()
        # Out of the many LT objects within layout, we are interested in LTTextBox and LTTextLine
        for lt_obj in layout:
            if isinstance(lt_obj, (LTTextBox, LTTextLine)):
                extracted_text += lt_obj.get_text()

    #close the pdf file
    filep.close()

    return extracted_text

#text = parsePDF('pdf1.pdf')

def allowed_ext(mail_path):
    """
    boolean function
    returns true if file is of valid type
    we have handled pdf, txt, jpg, jpeg and png
    """
    if (mail_path.endswith('.png') or mail_path.endswith('.jpg') or mail_path.endswith('.jpeg')
            or (mail_path.endswith('.pdf')) or (mail_path.endswith('.txt'))):
        return True
    return False

def extract_text(mail_path):
    """
    extracts text from valid file types, else returns empty string
    """
    #print(type(mail_path))
    print(f'=====>File receiveed is {mail_path}')

    #return empty string if file is not txt or pdf or image
    extracted_text = ""

    if (mail_path.endswith('.png') or mail_path.endswith('.jpg') or mail_path.endswith('.jpeg')):
        extracted_text = ocr(mail_path)
        #email = extracted_text.split('\n')

    if mail_path.endswith('.pdf'):
        print('PDF received')
        extracted_text = parse(mail_path)
        #extracted_text = str.split('\n'))
        print(extracted_text)

    elif mail_path.endswith('.txt'):
        print('txt received')
        email = open(mail_path, "r")
        body = ''
        for line in email:
            body = body + line
        extracted_text = body
        email.close()

    return extracted_text