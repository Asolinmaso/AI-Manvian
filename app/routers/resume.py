import os
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
try:
	from langchain_core.output_parsers import JsonOutputParser  # LC >=0.1
except Exception:  # fallback for older versions
	JsonOutputParser = None  # type: ignore
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.text_extract import extract_text_from_file

router = APIRouter(prefix="/resume", tags=["resume"])


class ParseResponse(BaseModel):
	# Keys align with FE profile model shape
	user: Dict[str, Any]
	about: Optional[Dict[str, Any]] = None
	education: Optional[list] = None
	experience: Optional[list] = None
	skills: Optional[list] = None
	languages: Optional[list] = None
	projects: Optional[list] = None
	certifications: Optional[list] = None
	recognitions: Optional[list] = None
	courses: Optional[list] = None
	services: Optional[list] = None
	products: Optional[list] = None
	keywords: Optional[list] = None


LLM_MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
	"You are an expert resume parser for a career platform. "
	"Extract clean, structured JSON suitable for directly updating a user's profile. "
	"IMPORTANT: Return ONLY valid JSON. No markdown, no explanations, no extra text. "
	"Use double quotes for all property names and string values. "
	"Dates can be month/year strings if unclear. Keep arrays even if single item. "
	"Ensure all JSON is properly formatted and parseable. "
)

JSON_SCHEMA_HINT = (
	"Output JSON with keys: user, about, education, experience, skills, languages, projects, "
	"certifications, recognitions, courses, services, products, keywords.\n"
	"IMPORTANT: Provide ALL required fields. Use defaults if not found in resume.\n"
	"- user: {{ username, email?, phone?, location? }}\n"
	"- about: {{ introduction, visionMission?, professionalGoals? }}\n"
	"- education: [{{ educationLevel (required), institutionName (required), boardUniversity (required, use 'Not specified' if missing), fieldOfStudy (required), educationStatus (required: 'Pursuing' or 'Completed'), startMonth (required, use '01' if missing), startYear (required), endMonth?, endYear?, gradeCgpaPercentage (required, use 'Not specified' if missing), links? }}]\n"
	"- experience: [{{ companyName, position, duration, typeOfEmployment?, workMode?, location?, currentlyWorkHere?, about?, skills?, links? }}]\n"
	"- skills: [{{ name, type (required: 'hard' or 'soft'), level (optional: 'Beginner', 'Intermediate', 'Advanced'), description? }}]\n"
	"- languages: [{{ name, proficiency (optional: 'Beginner', 'Intermediate', 'Advanced'), read?, write?, speak? }}]\n"
	"- projects: [{{ title, subTitle (required), description, duration (required), companyName?, projectType?, teamStructure?, location?, links? }}]\n"
	"- certifications: [{{ name, issuedBy, description?, links? }}]\n"
	"- recognitions: [{{ title, awardedBy, description?, links? }}]\n"
	"- courses: [{{ courseName (required), academyName (required), modeOfCourse (required), courseStatus (required: 'Pursuing' or 'Completed'), startMonth (required), endMonth (required), startYear (required), endYear (required), links? }}]\n"
	"- services: [{{ serviceName, description, numberOfProjects?, experienceYears?, links? }}]\n"
	"- products: [{{ productName, productType?, description, location?, links? }}]\n"
	"- keywords: [string]\n"
	"Default values to use when missing:\n"
	"- educationStatus: 'Completed' if degree is finished, 'Pursuing' if ongoing\n"
	"- startMonth: '01' if only year is known\n"
	"- gradeCgpaPercentage: 'Not specified' if not mentioned\n"
	"- boardUniversity: 'Not specified' if not clear\n"
	"- fieldOfStudy: Extract from degree name or use 'General Studies'\n"
	"- skills type: 'hard' for technical skills, 'soft' for interpersonal\n"
	"- courseStatus: 'Completed' if finished, 'Pursuing' if ongoing\n"
	"- modeOfCourse: 'Online', 'Offline', or 'Hybrid' based on context\n"
)


def fixCommonJSONIssues(json_text: str) -> str:
	"""Fix common JSON formatting issues"""
	import re
	
	# Fix single quotes to double quotes
	json_text = re.sub(r"'([^']*)'", r'"\1"', json_text)
	
	# Fix trailing commas
	json_text = re.sub(r',\s*}', '}', json_text)
	json_text = re.sub(r',\s*]', ']', json_text)
	
	# Fix unescaped quotes in strings
	json_text = re.sub(r'([^\\])"([^"]*)"([^,}\]]*)"', r'\1"\2\3"', json_text)
	
	return json_text

def createFallbackResponse(seed_user: dict) -> dict:
	"""Create a minimal valid response when AI fails"""
	return {
		"user": {
			"username": seed_user.get("username", "User"),
			"email": seed_user.get("email"),
			"phone": seed_user.get("phone"),
			"location": seed_user.get("location"),
		},
		"about": {
			"introduction": "Professional with diverse experience and skills.",
			"visionMission": "",
			"professionalGoals": "",
		},
		"education": [],
		"experience": [],
		"skills": [],
		"languages": [],
		"projects": [],
		"certifications": [],
		"recognitions": [],
		"courses": [],
		"services": [],
		"products": [],
		"keywords": [],
	}

def build_chain():
	# Enforce JSON output when supported
	llm = ChatGroq(
		temperature=0.1,
		groq_api_key=os.getenv("GROQ_API_KEY"),
		model_name=LLM_MODEL_NAME
	)
	prompt = ChatPromptTemplate.from_messages([
		("system", SYSTEM_PROMPT + "\n" + JSON_SCHEMA_HINT + "\nAlways include empty arrays for missing sections."),
		("human", "Here is the resume text to parse:\n\n{resume_text}\n\nExample valid JSON format:\n{{\n  \"user\": {{\"username\": \"John Doe\", \"email\": \"john@example.com\"}},\n  \"about\": {{\"introduction\": \"Experienced professional\"}},\n  \"education\": [{{\"educationLevel\": \"Bachelor's Degree\", \"institutionName\": \"University\", \"boardUniversity\": \"Not specified\", \"fieldOfStudy\": \"Computer Science\", \"educationStatus\": \"Completed\", \"startMonth\": \"01\", \"startYear\": \"2020\", \"gradeCgpaPercentage\": \"3.5\"}}],\n  \"skills\": [{{\"name\": \"JavaScript\", \"type\": \"hard\"}}],\n  \"keywords\": [\"programming\", \"web development\"]\n}}"),
	])
	if JsonOutputParser is not None:
		return prompt | llm | JsonOutputParser()
	# Fallback to string parser if JsonOutputParser isn't available
	return prompt | llm | StrOutputParser()


@router.post("/parse", response_model=ParseResponse)
async def parse_resume(
	file: UploadFile = File(...),
	username_hint: Optional[str] = Form(None),
	email_hint: Optional[str] = Form(None),
	phone_hint: Optional[str] = Form(None),
	location_hint: Optional[str] = Form(None),
	model: Optional[str] = Form(None),
):
	try:
		resume_text = await extract_text_from_file(file)
		if not resume_text or len(resume_text.strip()) < 20:
			raise HTTPException(status_code=400, detail="Could not extract text from resume")

		chain = build_chain() if not model else (ChatPromptTemplate.from_messages([
			("system", SYSTEM_PROMPT + "\n" + JSON_SCHEMA_HINT),
			("human", "Here is the resume text to parse using model {model}:\n\n{resume_text}"),
		]) | ChatGroq(model_name=model, temperature=0.2, groq_api_key=os.getenv("GROQ_API_KEY")) | StrOutputParser())

		seed_user = {
			"username": username_hint,
			"email": email_hint,
			"phone": phone_hint,
			"location": location_hint,
		}
		seed_user = {k: v for k, v in seed_user.items() if v}

		result_text = await chain.ainvoke({
			"resume_text": resume_text[:100000],
			"model": model or LLM_MODEL_NAME,
		})

		# If chain already parsed JSON (JsonOutputParser), accept dict directly
		import json
		if isinstance(result_text, dict):
			parsed = result_text
		else:
			# Try to parse model JSON output
			parsed = None
			cleaned_text = result_text.strip() if isinstance(result_text, str) else str(result_text)
			try:
				parsed = json.loads(cleaned_text)
			except Exception:
				# Remove markdown code fences if present
				if cleaned_text.startswith("```json"):
					cleaned_text = cleaned_text[7:]
				if cleaned_text.startswith("```"):
					cleaned_text = cleaned_text[3:]
				if cleaned_text.endswith("```"):
					cleaned_text = cleaned_text[:-3]
				# Find the first { and last } to extract JSON
				start = cleaned_text.find("{")
				end = cleaned_text.rfind("}")
				if start != -1 and end != -1 and end > start:
					json_text = cleaned_text[start:end+1]
					try:
						parsed = json.loads(json_text)
					except Exception:
						# Try to fix common JSON issues
						fixed_json = fixCommonJSONIssues(json_text)
						try:
							parsed = json.loads(fixed_json)
						except Exception:
							parsed = None
				else:
					parsed = None
			if parsed is None:
				# Return minimal fallback response if parsing failed
				parsed = createFallbackResponse(seed_user)

		# Merge seed user data if missing
		user = parsed.get("user") or {}
		user = {**seed_user, **user}
		parsed["user"] = user

		# Normalize required defaults across sections to reduce backend validation errors
		def norm_education(e):
			return {
				"educationLevel": e.get("educationLevel") or "Bachelor's Degree",
				"institutionName": e.get("institutionName") or "Not specified",
				"boardUniversity": e.get("boardUniversity") or "Not specified",
				"fieldOfStudy": e.get("fieldOfStudy") or "General Studies",
				"educationStatus": ("Pursuing" if e.get("educationStatus") in ["In Progress", "Pursuing"] else (e.get("educationStatus") or "Completed")),
				"startMonth": e.get("startMonth") or "01",
				"startYear": e.get("startYear") or str(__import__("datetime").datetime.utcnow().year),
				"endMonth": e.get("endMonth") or None,
				"endYear": e.get("endYear") or None,
				"gradeCgpaPercentage": e.get("gradeCgpaPercentage") or "Not specified",
				"links": e.get("links") or None,
			}

		def norm_skill(s):
			lvl = s.get("level")
			return {
				"name": s.get("name") or "Unknown Skill",
				"type": ("hard" if s.get("type") in ["hard", "proficient"] else ("soft" if s.get("type") == "soft" else "hard")),
				"level": lvl if lvl in ["Beginner", "Intermediate", "Advanced"] else None,
				"description": s.get("description") or None,
			}

		def norm_project(p):
			return {
				"title": p.get("title") or "Untitled Project",
				"subTitle": p.get("subTitle") or p.get("title") or "Project",
				"description": p.get("description") or "No description available",
				"duration": p.get("duration") or "Not specified",
				"companyName": p.get("companyName") or None,
				"projectType": p.get("projectType") or None,
				"teamStructure": p.get("teamStructure") or None,
				"location": p.get("location") or None,
				"links": p.get("links") or None,
			}

		def norm_course(c):
			return {
				"courseName": c.get("courseName") or "Unnamed Course",
				"academyName": c.get("academyName") or "Not specified",
				"modeOfCourse": c.get("modeOfCourse") or "Online",
				"courseStatus": ("Pursuing" if c.get("courseStatus") in ["In Progress", "Pursuing"] else (c.get("courseStatus") or "Completed")),
				"startMonth": c.get("startMonth") or "01",
				"endMonth": c.get("endMonth") or "12",
				"startYear": c.get("startYear") or str(__import__("datetime").datetime.utcnow().year),
				"endYear": c.get("endYear") or str(__import__("datetime").datetime.utcnow().year),
				"links": c.get("links") if isinstance(c.get("links"), list) else None,
			}

		if isinstance(parsed.get("education"), list):
			parsed["education"] = [norm_education(e or {}) for e in parsed["education"]]
		if isinstance(parsed.get("skills"), list):
			parsed["skills"] = [norm_skill(s or {}) for s in parsed["skills"]]
		if isinstance(parsed.get("projects"), list):
			parsed["projects"] = [norm_project(p or {}) for p in parsed["projects"]]
		if isinstance(parsed.get("courses"), list):
			parsed["courses"] = [norm_course(c or {}) for c in parsed["courses"]]

		return parsed
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
