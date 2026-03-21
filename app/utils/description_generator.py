import os
import re
import html
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# --- UTILS ---

def clean_html_spacing(html_text: str) -> str:
    """Clean up HTML output for consistent multi-platform rendering."""
    cleaned = re.sub(r'\s*(<ul>)\s*', r'\1', html_text)
    cleaned = re.sub(r'\s*(</ul>)\s*', r'\1', cleaned)
    cleaned = re.sub(r'</li>\s*<li>', '</li><li>', cleaned)
    cleaned = re.sub(r'<br\s*/?>\s*<br\s*/?>[\s<br\s*/?>]*', '<br>', cleaned)
    cleaned = re.sub(r'</ul>\s*(<br\s*/?>){0,2}\s*<b>', '</ul><br><b>', cleaned)
    return cleaned.strip()

# --- TEMPLATES ---

ROLE_PERSONA = """
# ROLE
You are a Senior Strategic Recruitment Specialist and Copywriter. Your mission is to transform raw opportunity data into a compelling, professional, and visually structured narrative that reflects corporate excellence.
"""

CONSTRAINTS = """
# RIGOROUS CONSTRAINTS
1. **ACCURACY**: Strictly adhere to provided skills and experience. Do NOT hallucinate extras.
2. **HTML COMPATIBILITY**: Use <b> for bolding. DO NOT use markdown (**) for bolding.
3. **LIST DENSITY**: Ensure ZERO whitespace or line breaks between <ul>/<li> tags.
4. **SPACING**: Separate logical sections with exactly one <br> tag.
5. **WORD COUNT**: Reach approximately {wordCount} words by elaborating on professional context and impact.
6. **EXCLUSIONS**: Do NOT mention "Vacancy", "Last Date", or meta-notes (e.g., "The image shows...") in the narrative.
"""

FORMATS = {
    "Full-time": """
<b>About the Role:</b> [Brief overview of the impact and scope.]<br>
<b>About the Company:</b> [Brief intro to culture and mission.]<br>
<b>Key Responsibilities:</b> <ul><li>[Specific job duties.]</li></ul><br>
<b>Required Skills & Qualifications:</b> <ul><li>[Core requirements.]</li></ul><br>
<b>Nice to Have:</b> <ul><li>[Optional extras.]</li></ul><br>
<b>Perks & Benefits:</b> <ul><li>[Compensation, health, leave, etc.]</li></ul>
    """,
    "Part-time": """
<b>About the Role:</b> [Overview including flexible hours/commitment.]<br>
<b>Key Responsibilities:</b> <ul><li>[Core tasks.]</li></ul><br>
<b>Qualifications & Skills:</b> <ul><li>[Key requirements.]</li></ul><br>
<b>Perks:</b> <ul><li>[Work-life balance, flexibility.]</li></ul>
    """,
    "Internship (Stipend)": """
<b>Internship Overview:</b> [Focus on learning and mentorship.]<br>
<b>Learning Opportunities:</b> <ul><li>[Skills the intern will develop.]</li></ul><br>
<b>Key Responsibilities:</b> <ul><li>[Daily tasks.]</li></ul><br>
<b>Requirements:</b> <ul><li>[Eligibility and background.]</li></ul>
    """,
    "Internship (Unpaid)": """
<b>Internship Overview:</b> [Focus on networking and growth.]<br>
<b>Learning Opportunities:</b> <ul><li>[Mentorship and skills.]</li></ul><br>
<b>Key Responsibilities:</b> <ul><li>[Daily tasks.]</li></ul><br>
<b>Non-Monetary Benefits:</b> <ul><li>[Certificates, networking, etc.]</li></ul>
    """,
    "Contract-based": """
<b>Project Overview:</b> [Short intro to project deliverables.]<br>
<b>Responsibilities:</b> <ul><li>[List of deliverables.]</li></ul><br>
<b>Requirements:</b> <ul><li>[Tools and specific experience.]</li></ul><br>
<b>Nice to Have:</b> <ul><li>[Prior contract work experience.]</li></ul>
    """,
    "Project (Freelancer)": """
<b>Project Overview:</b> [Goals and technical impact.]<br>
<b>Who We're Looking For:</b> [Independent professionals with specific expertise.]<br>
<b>Required Skills:</b> <ul><li>[Technical requirements.]</li></ul><br>
<b>Timeline:</b> {workDuration}
    """,
    "POP": """
<b>About the Opportunity:</b> [Brief overview of the passed opportunity.]<br>
<b>Key Details:</b> <ul><li>[Location, skills, duration, etc.]</li></ul><br>
<b>Requirements & Preferences:</b> <ul><li>[Skills and eligibility.]</li></ul><br>
<b>Additional Information:</b> [Relevant details like package/notes.]
    """,
    "Default": """
<b>About the Opportunity:</b> [Brief intro to role and purpose.]<br>
<b>Key Responsibilities:</b> <ul><li>[List of duties.]</li></ul><br>
<b>Skills & Qualifications:</b> <ul><li>[Core requirements.]</li></ul><br>
<b>Eligibility:</b> <ul><li>[Criteria for selection.]</li></ul>
    """
}

# --- GENERATION LOGIC ---

def get_llm():
    return ChatGroq(
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

def generate_description(data: dict) -> str:
    word_count = data.get("wordCount", 800) or 800
    company_type = data.get("companyType", "company")
    post_type = data.get("postType") or data.get("opportunityType") or "Default"
    
    # Handle Company Name
    if company_type in ["company", "Adept"]:
        entity_name = (data.get("companyName") or "Not specified").strip()
    else:
        entity_name = (data.get("individualCompanyName") or "Not specified").strip()
        
    title = html.escape(data.get("title") or data.get("opportunityTitle") or "Untitled Role")
    
    # Mapping for specialized formatting
    format_instr = FORMATS.get(post_type, FORMATS["Default"])
    
    prompt_data = {
        "companyName": entity_name,
        "title": title,
        "postType": post_type,
        "location": data.get("location") or "Not specified",
        "workMode": data.get("workMode") or "Not specified",
        "workDuration": data.get("workDuration") or "Not specified",
        "timeCommitment": data.get("timeCommitment") or "Not specified",
        "vacancy": data.get("vacancy") or "1",
        "lastDate": data.get("lastDate") or "Not specified",
        "package": data.get("package") or "Competitive compensation",
        "skills": data.get("skills") or data.get("skillsRequired") or "Not specified",
        "keywords": data.get("keywords") or "Not specified",
        "eligibility": data.get("eligibility") or "Not specified",
        "wordCount": word_count
    }

    prompt = ChatPromptTemplate.from_template(f"""
    {ROLE_PERSONA}

    # CONTEXT & DATA
    - Organization: {{companyName}}
    - Role Title: {{title}} ({{postType}})
    - Parameters: {{location}} ({{workMode}}) | {{timeCommitment}} | {{workDuration}}
    - Requirements: {{skills}} | {{keywords}} | {{eligibility}}
    - Compensation Structure: {{package}}

    # FORMATTING SPECIFICATION
    {format_instr}

    {CONSTRAINTS}
    """)

    chain = LLMChain(llm=get_llm(), prompt=prompt)
    response = chain.run(prompt_data)
    return clean_html_spacing(response)

def generate_pass_opportunity_description(data: dict) -> str:
    """Specialized generator for passed opportunities (POP)."""
    data["postType"] = "POP" # Force POP template
    return generate_description(data)
