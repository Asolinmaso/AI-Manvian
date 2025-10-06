from typing import Literal
from fastapi import UploadFile
import os

async def extract_text_from_file(file: UploadFile) -> str:
	name = (file.filename or "").lower()
	content = await file.read()
	# reset file for potential future reads
	try:
		await file.seek(0)
	except Exception:
		pass

	if name.endswith(".pdf"):
		# Prefer PyMuPDF for better extraction; fallback to pypdf
		try:
			import fitz  # PyMuPDF
			import io
			doc = fitz.open(stream=content, filetype="pdf")
			texts = []
			for page in doc:
				texts.append(page.get_text("text"))
			return "\n".join(texts)
		except Exception:
			from pypdf import PdfReader
			import io
			reader = PdfReader(io.BytesIO(content))
			texts = []
			for page in reader.pages:
				try:
					texts.append(page.extract_text() or "")
				except Exception:
					continue
			return "\n".join(texts)
	elif name.endswith(".docx") or name.endswith(".doc"):
		import io
		import docx2txt
		with open("/tmp/_resume_upload.docx", "wb") as f:
			f.write(content)
		try:
			text = docx2txt.process("/tmp/_resume_upload.docx") or ""
		finally:
			try:
				os.remove("/tmp/_resume_upload.docx")
			except Exception:
				pass
		return text
	else:
		# assume text
		try:
			return content.decode("utf-8", errors="ignore")
		except Exception:
			return content.decode("latin-1", errors="ignore")
