import base64
import os
import io
import shutil
import platform
import pytesseract
from PIL import Image
from typing import Optional, Any
from fastapi import UploadFile
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configure Tesseract path (Centralized)
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    homebrew_path = '/opt/homebrew/bin/tesseract'
    if os.path.exists(homebrew_path):
        pytesseract.pytesseract.tesseract_cmd = homebrew_path
    else:
        pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', '/usr/bin/tesseract')

async def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using AI Vision (Groq) with Tesseract fallback.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # 1. Try AI Vision Extraction First
    if groq_api_key:
        try:
            chat = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.2-11b-vision-preview",
                temperature=0.1,
            )
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = """
            # ROLE
            You are a Senior Recruitment Operations Specialist. Your task is to perform an exhaustive and high-fidelity extraction of job-related data from the provided image.

            # TASK
            Extract all textual content and infer the hierarchical structure of the job posting or advertisement.

            # EXTRACTION GUIDELINES
            1. **Text Fidelity**: Maintain exact terminology for technical requirements, certifications, and tools.
            2. **Structural Preservation**: Identify and preserve logical sections (e.g., Intro, Responsibilities, Requirements, Benefits).
            3. **Data Points to Prioritize**:
                - Role Title & Level (e.g., Senior, Junior, Associate)
                - Hiring Entity (Company, Unit, or Individual)
                - Detailed Compensation (Salary, Hourly, Stipend, Bonuses)
                - Location Specifics (Remote, Hybrid, On-site, City/Country)
                - Explicit Deadlines and Start Dates
                - Contact Information (Names, Emails)
            
            # OUTPUT FORMAT
            - Output the result using professional Markdown.
            - Omit any conversational preamble or descriptive meta-text.
            - transcribe literal text without guessing if a section is unclear.
            - Ensure bulleted lists are properly preserved.

            # CONSTRAINTS
            - DO NOT include summary notes or hallucinations.
            - DO NOT output anything except the extracted data.
            """

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            )
            response = await chat.ainvoke([message])
            extracted_text = response.content.strip()
            
            if extracted_text and len(extracted_text) > 0:
                print("Extraction successful using AI Vision.")
                return extracted_text
        except Exception as e:
            print(f"AI Vision extraction failed, falling back to Tesseract: {str(e)}")

    # 2. Fallback to Tesseract OCR
    try:
        if not shutil.which('tesseract') and not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            print("Waring: Tesseract not found. Returning empty text.")
            return ""
            
        img = Image.open(io.BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(img)
        print("Extraction successful using Tesseract OCR.")
        return extracted_text.strip()
    except Exception as e:
        print(f"Tesseract extraction failed: {str(e)}")
        return ""

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Unified file text extraction (Images, PDFs, Docx, Text).
    """
    name = (file.filename or "").lower()
    content = await file.read()
    
    # reset file for potential future reads (important for UploadFile)
    try:
        await file.seek(0)
    except Exception:
        pass

    # Image Handling
    if any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        return await extract_text_from_image(content)

    # PDF Handling
    if name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=content, filetype="pdf")
            return "\n".join(page.get_text("text") for page in doc)
        except Exception:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    # DOCX Handling
    if name.endswith(".docx") or name.endswith(".doc"):
        import docx2txt
        temp_path = f"/tmp/_extract_{os.getpid()}.docx"
        with open(temp_path, "wb") as f:
            f.write(content)
        try:
            return docx2txt.process(temp_path) or ""
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Plain Text Handling
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return content.decode("latin-1", errors="ignore")
