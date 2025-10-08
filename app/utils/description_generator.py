import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import html

load_dotenv()

force_elaboration = True

def clean_html_spacing(html_text):
    """
    Clean up HTML output by removing extra line breaks and spaces between <li> elements.
    Ensures tight spacing within <ul> lists and exactly one <br> tag between sections.
    """
    cleaned_html = re.sub(r'\s*(<ul>)\s*', r'\1', html_text)
    cleaned_html = re.sub(r'\s*(</ul>)\s*', r'\1', cleaned_html)
    cleaned_html = re.sub(r'</li>\s*<li>', '</li><li>', cleaned_html)
    cleaned_html = re.sub(r'<br\s*/?>\s*<br\s*/?>[\s<br\s*/?>]*', '<br>', cleaned_html)
    cleaned_html = re.sub(r'</ul>\s*(<br\s*/?>){0,2}\s*<b>', '</ul><br><b>', cleaned_html)
    return cleaned_html

def generate_description(data):
    wordCount = data.get("wordCount", 800) or 800
    company_type = data.get("companyType", "company")
    # Use the correct field name based on companyType
    post_type = (data.get("opportunityType", "") or "") if company_type == "company" else (data.get("postType", "") or "")
    if company_type in ["company", "Adept"]:
        company_name = (data.get("companyName", "") or "Not specified").strip()
    else:  # "individual"
        company_name = None  # No company name for individuals
        individual_name = (data.get("individualCompanyName", "") or "Not specified").strip()
    title = html.escape(data.get("title", "") or "Untitled Role")
    skills = data.get("skills", []) or []
    keywords = data.get("keywords", []) or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    # Combine important words for bolding (remove duplicates)
    important_words = list(set([company_name] + skills + keywords))
    important_words = [word for word in important_words if word]

    # Extract common fields
    location = data.get("location", "") or "Not specified"
    package = data.get("package", "") or "Competitive compensation"
    last_date = data.get("lastDate", "") or "Not specified"
    # Convert vacancy to float and handle invalid inputs
    vacancy_str = data.get("vacancy", "1")
    try:
        vacancy = int(vacancy_str) if str(vacancy_str).replace('.', '').isdigit() else 1
    except ValueError:
        vacancy = 1

    # Fields specific to "For My Company"
    work_duration = data.get("workDuration", "") or "Not specified"
    work_mode = data.get("workMode", "") or "Not specified"
    time_commitment = data.get("timeCommitment", "") or "Not specified"

    # Fields specific to "Individual"
    eligibility = data.get("eligibility", "") or "Not specified"
    
    # Additional enhanced fields for CROP form
    your_name = data.get("yourName", "") or "Not specified"
    your_identity = data.get("yourIdentity", "") or "Not specified"
    education_requirements = data.get("educationRequirements", "") or "Not specified"
    industry_expertise = data.get("industryExpertise", "") or "Not specified"
    # preferredExperience is now sent as formatted string from frontend
    preferred_experience_text = data.get("preferredExperience", "") or "Not specified"
    language_preference = data.get("languagePreference", "") or "Not specified"
    gender_preference = data.get("genderPreference", "") or "Not specified"

    # LLM setup - use environment variable for API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant"
    )

    # Common prompt data
    prompt_data = {
        "companyName": company_name,
        "individualName": individual_name if company_type == "individual" else None,
        "title": title,
        "postType": post_type or "Generic Role",
        "location": location,
        "package": package,
        "lastDate": last_date,
        "vacancy": vacancy,
        "skills": ", ".join(skills) if skills else "Not specified",
        "keywords": ", ".join(keywords) if keywords else "Not specified",
        "workDuration": work_duration,
        "workMode": work_mode,
        "timeCommitment": time_commitment,
        "eligibility": eligibility,
        # Enhanced fields for CROP form
        "yourName": your_name,
        "yourIdentity": your_identity,
        "educationRequirements": education_requirements,
        "industryExpertise": industry_expertise,
        "preferredExperience": preferred_experience_text,
        "languagePreference": language_preference,
        "genderPreference": gender_preference
    }
    
    # Define formats based on companyType and postType
    if company_type in ["company", "Adept"]:
        # "For My Company" formats
        if not post_type:
            intro_instruction = "Generate a generic job description for a role. The tone should be professional and adaptable to any job type."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Brief intro to the role and company.]

<b>Responsibilities:</b>  
<ul>
    <li>[List of duties (if provided, otherwise generic duties).]</li>
</ul>

<b>Skills & Qualifications:</b>  
<ul>
    <li>[List of skills (if provided, otherwise generic skills).]</li>
</ul>
"""
        elif post_type == "Full-time":
            intro_instruction = "Generate a professional full-time job description targeted at attracting qualified candidates. The tone should be formal, aspirational, and highlight long-term career growth, company culture, and stability."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Brief overview.]<br>

<b>About the Company:</b>  
[Brief intro to the company, culture, and mission.]<br>

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of job duties.]</li>
</ul>

<b>Required Skills & Qualifications:</b>  
<ul>
    <li>[List of skills and qualifications.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills or experiences that enhance the candidate's fit.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Perks like health insurance, paid leave, etc.]</li>
</ul>
"""
        elif post_type == "Part-time":
            intro_instruction = "Generate a clear and professional part-time job description. Highlight flexible hours, key responsibilities, and the specific time commitment required. Keep the tone friendly yet informative."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Brief about the role and work hours.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties.]</li>
</ul>

<b>Qualifications & Skills:</b>  
<ul>
    <li>[List key skills.]</li>
</ul>

<b>Perks:</b>  
<ul>
    <li>[Highlight work-life balance, flexibility.]</li>
</ul>
"""
        elif post_type == "Internship (Stipend)":
            intro_instruction = "Create a paid internship post that is inviting to students or fresh graduates. Emphasize learning, mentorship, potential growth opportunities, and the stipend as a financial incentive. Keep the tone encouraging and professional."
            format_instruction = """
Format: 
<b>About the Company:</b>  
[Brief overview.]

<b>Internship Overview:</b>  
[What interns will work on.]

<b>Learning Opportunities:</b>  
<ul>
    <li>[List skills interns will develop.]</li>
</ul>

<b>Requirements:</b>  
<ul>
    <li>[Eligibility, background, or tools.]</li>
</ul>

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties.]</li>
</ul>
"""
        elif post_type == "Internship (Unpaid)":
            intro_instruction = "Create an unpaid internship post that is inviting to students or fresh graduates. Emphasize learning, mentorship, networking opportunities, and other non-monetary benefits to attract candidates. Keep the tone encouraging and professional."
            format_instruction = """
Format: 
<b>About the Company:</b>  
[Brief overview.]
<b>Internship Overview:</b>  
[What interns will work on.]
<b>Learning Opportunities:</b>  
<ul>
    <li>[List skills interns will develop.]</li>
</ul>
<b>Requirements:</b>  
<ul>
    <li>[Eligibility, background, or tools.]</li>
</ul>
<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties.]</li>
</ul>
<b>Non-Monetary Benefits:</b>  
<ul>
    <li>[Highlight mentorship, networking, certificates, etc.]</li>
</ul>
"""
        elif post_type == "Contract-based":
            intro_instruction = "Generate a professional contract opportunity post. Focus on short-term project deliverables, duration, and payment. It is not a job. The tone should appeal to freelancers or short-term collaborators."
            format_instruction = """
Format:
<b>Overview:</b>  
[Short intro to project.]
<b>About the Company:</b>  
[Brief overview.]
<b>Responsibilities:</b>  
<ul>
    <li>[List of deliverables.]</li>
</ul>

<b>Requirements:</b>  
<ul>
    <li>[Tools, skills, experience needed.]</li>
</ul>
<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills, e.g., prior contract work experience.]</li>
</ul>
"""
        elif post_type == "Project (Freelancer)":
            intro_instruction = "Generate a project collaboration post for individual freelancers. This is not a job but an opportunity for freelancers to contribute to a specific project with clear goals and timelines. Focus on skillset needed, project objectives, and payment terms. Keep the tone flexible and appealing to independent professionals."
            format_instruction = """
Format:
<b>Project Overview:</b>  
[Summary of the project, its impact, and goals.]

<b>Who We're Looking For:</b>  
[Type of freelancers, e.g., independent professionals with specific skills.]

<b>Required Skills:</b>  
<ul>
    <li>[List of technical/non-technical skills.]</li>
</ul>

<b>Timeline:</b> {workDuration}  
"""
        elif post_type == "Project (Service Company)":
            intro_instruction = "Generate a project collaboration post for service companies. This is not a job but an opportunity for companies to collaborate on a specific project with clear goals and timelines. Focus on partnership potential, project scale, and required expertise. Keep the tone formal and professional."
            format_instruction = """
Format:
<b>Project Overview:</b>  
[Summary of the project, its impact, and goals.]

<b>Who We're Looking For:</b>  
[Type of service companies, e.g., agencies or firms with specific expertise.]

<b>Required Expertise:</b>  
<ul>
    <li>[List of technical/non-technical expertise required.]</li>
</ul>

<b>Collaboration Scope:</b>  
[Details on how the collaboration will work.]

<b>Timeline:</b> {workDuration}  
"""
        else:   # Individual types, including "entrepreneur"
            # "Individual" formats for "Create New Opportunity"
            intro_instruction = "Generate a generic job description for a role. The tone should be professional and adaptable to any job type."
            format_instruction = """
Format:
<b>About the Opportunity:</b>  
[Brief intro to the role .]

<b>Responsibilities:</b>  
<ul>
    <li>[List of duties (if provided, otherwise generic duties).]</li>
</ul>

<b>Skills & Qualifications:</b>  
<ul>
    <li>[List of skills (if provided, otherwise generic skills).]</li>
</ul>
"""
    else:
        # "Individual" formats for "Create New Opportunity"
        if not post_type:
            intro_instruction = "Generate a generic opportunity description for an individual offering collaboration. The tone should be professional yet approachable, focusing on the individual's goals and the opportunity's purpose."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Brief intro to the opportunity and its purpose.]

<b>About Me:</b>  
[Brief intro to the individual and their goals.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties (if provided, otherwise generic duties).]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List of skills (if provided, otherwise generic skills).]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills or experiences.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package and any additional benefits.]</li>
</ul>

<b>Eligibility:</b>  
<ul>
    <li>[Details about eligibility criteria.]</li>
</ul>
"""
        elif post_type == "Full-time":
            intro_instruction = "Generate a professional full-time opportunity description for an individual seeking a long-term collaborator. The tone should be approachable yet formal, emphasizing the individual's vision, the role's impact, and opportunities for growth."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Details about the role, its impact, and long-term potential.]<br>

<b>About Me:</b><br>
[Brief intro to the individual, their vision, and passion for the project.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of job duties relevant to a full-time role.]</li>
</ul><br>

<b>Required Skills:</b>  
<ul>
    <li>[List of skills and qualifications.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills or experiences, e.g., additional technical skills or domain knowledge.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package, e.g., salary, and any collaboration opportunities.]</li>
</ul>

<b>Eligibility:</b>  
<ul>
    <li>[Details about eligibility criteria.]</li>
</ul>
"""
        elif post_type == "Part-time":
            intro_instruction = "Generate a clear and professional part-time opportunity description for an individual seeking a flexible collaborator. Highlight the role's flexibility, key responsibilities, and the individual's support, with a friendly yet professional tone."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Brief about the role, its flexibility, and expected time commitment.]

<b>About Me:</b>  
[Brief intro to the individual and their goals.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties tailored for a part-time role.]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List key skills.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills or experiences, e.g., prior experience in similar roles.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package, e.g., hourly rate, and benefits like flexible hours.]</li>
</ul>

<b>Eligibility:</b>  
<ul>
    <li>[Details about eligibility criteria.]</li>
</ul>
"""
        elif post_type == "Internship (Stipend)":
            intro_instruction = "Create a paid internship opportunity description for an individual seeking a learner. Emphasize the learning opportunities, mentorship, and stipend, with an encouraging and professional tone focused on growth and development."
            format_instruction = """
Format: 
<b>About the Role:</b>  
[What interns will work on and learn.]

<b>About Me:</b>  
[Brief overview of the individual and their commitment to mentoring.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties for the internship.]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List of skills, e.g., basic knowledge of tools or technologies.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills, e.g., familiarity with specific software.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the stipend and learning opportunities.]</li>
</ul>

"""
        elif post_type == "Internship (Unpaid)":
            intro_instruction = "Create an unpaid internship opportunity description for an individual seeking a learner. Emphasize the learning opportunities, networking benefits, and non-monetary perks, with an encouraging and professional tone focused on growth."
            format_instruction = """
Format: 
<b>About the Role:</b>  
[What interns will work on and learn.]

<b>About Me:</b>  
[Brief overview of the individual and their commitment to mentoring.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties for the internship.]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List of skills, e.g., basic knowledge of tools or technologies.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills, e.g., familiarity with specific software.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Highlight networking opportunities, certificates, mentorship, etc.]</li>
</ul>
"""
        elif post_type == "Contract-based":
            intro_instruction = "Generate a professional contract opportunity description for an individual seeking a short-term collaborator. Focus on the project's deliverables, timeline, and compensation, with a professional tone appealing to freelancers."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Short intro to the project and its goals.]

<b>About Me:</b>  
[Brief overview of the individual and their project.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of deliverables for the contract.]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[Tools, skills, experience needed.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills, e.g., prior contract work experience.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package and any additional benefits.]</li>
</ul>

"""
        elif post_type == "Project (Freelancer)":
            intro_instruction = "Generate a project collaboration opportunity description for an individual seeking freelancers. Focus on the project's goals, required skills, and appeal to independent professionals, with a flexible and approachable tone."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Summary of the project, its impact, and goals.]

<b>About Me:</b>  
[Brief overview of the individual and their vision.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of tasks for the project.]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List of technical/non-technical skills.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills, e.g., experience with similar projects.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package and any additional benefits.]</li>
</ul>

<b>Eligibility:</b>  
<ul>
    <li>[Details about eligibility criteria.]</li>
</ul>
"""
        elif post_type == "Project (Service Company)":
            intro_instruction = "Generate a project collaboration opportunity description for an individual seeking service companies. Focus on the project's scale, partnership potential, and required expertise, with a formal and professional tone."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Summary of the project, its impact, and goals.]

<b>About Me:</b>  
[Brief overview of the individual and their vision.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of responsibilities for the collaboration.]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List of technical/non-technical expertise required.]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional expertise, e.g., experience in similar collaborations.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package and any additional benefits.]</li>
</ul>

"""
        else:
            intro_instruction = "Generate a generic opportunity description for an individual seeking collaboration. The tone should be professional yet approachable, focusing on the individual's goals and the opportunity's purpose."
            format_instruction = """
Format:
<b>About the Role:</b>  
[Brief intro to the opportunity and its purpose.]

<b>About Me:</b>  
[Brief intro to the individual and their goals.]

<b>Key Responsibilities:</b>  
<ul>
    <li>[List of duties (if provided, otherwise generic duties).]</li>
</ul>

<b>Required Skills:</b>  
<ul>
    <li>[List of skills (if provided, otherwise generic skills).]</li>
</ul>

<b>Nice to Have:</b>  
<ul>
    <li>[List of optional skills or experiences.]</li>
</ul>

<b>Perks & Benefits:</b>  
<ul>
    <li>[Details about the package and any additional benefits.]</li>
</ul>

<b>Eligibility:</b>  
<ul>
    <li>[Details about eligibility criteria.]</li>
</ul>
"""

    # Full prompt with dynamic instructions
    prompt = ChatPromptTemplate.from_template(f"""
    {intro_instruction}

    Generate a professional and polished opportunity post using ONLY the following details. DO NOT add skills, experience ranges, or qualifications that are not provided:

    Company Details:
    - Company Name: {{companyName}}
    - Individual Name: {{individualName}}
    
    Opportunity Details:
    - Post Type: {{postType}}
    - Job Title: {{title}}
    - Location: {{location}}
    - Work Mode: {{workMode}}
    - Work Duration: {{workDuration}}
    - Time Commitment: {{timeCommitment}}
    - Vacancy: {{vacancy}}
    - Last Date: {{lastDate}}
    - Package: {{package}}
    
    Requirements (USE EXACTLY AS PROVIDED):
    - Required Skills: {{skills}}
    - Keywords: {{keywords}}
    - Education Requirements: {{educationRequirements}}
    - Preferred Experience: {{preferredExperience}}
    - Industry Expertise: {{industryExpertise}}
    - Language Preference: {{languagePreference}}
    - Gender Preference: {{genderPreference}}
    - Eligibility: {{eligibility}}

    Important words to bold: {', '.join(important_words) if important_words else 'None'}

    {format_instruction}

    CRITICAL INSTRUCTIONS - FOLLOW STRICTLY:
    1. SKILLS SECTION: Use ONLY the skills listed in "Required Skills: {{skills}}". DO NOT add any skills that are not explicitly mentioned.
    2. EXPERIENCE SECTION: Use EXACTLY "{{preferredExperience}}" as provided. DO NOT change the experience range.
    3. EDUCATION: Use "{{educationRequirements}}" as provided. If it says "any", mention "Any educational background" or similar.
    4. DO NOT hallucinate or invent additional requirements, skills, or qualifications.
    5. Use only <b> tags for bold, not **. 
    6. Your response should be suitable for direct copy-pasting into a web page or email client that supports HTML formatting.
    7. For all <ul> lists, ensure there are NO extra line breaks or spaces between <li> elements.
    8. Ensure exactly one <br> tag between sections for clean spacing.
    9. Strictly follow the format provided above. DO NOT add extra fields or sections.
    10. Within paragraph sections, bold the important words listed above using <b> tags. Do NOT bold words within <ul> lists.
    11. Ensure the response is at least {wordCount} words by elaborating on the role context, company culture, and responsibilities - NOT by adding extra skills or requirements.
    12. Make the tone fit the nature of the role (e.g., formal for full-time, friendly for internships).
    13. DO NOT MENTION ANY NOTES at the end of description.
    14. DO NOT mention vacancies and last date to apply in the description body.
    15. If Gender Preference is "Any", DO NOT mention it in the description.
    16. Keywords ({{keywords}}) should be naturally integrated into the description, not listed separately.
    """)

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(prompt_data)
    cleaned_response = clean_html_spacing(response)
    return cleaned_response

def generate_pass_opportunity_description(data):
    wordCount = data.get("wordCount", 800) or 800
    company_type = data.get("companyType", "company")
    company_name = (data.get("companyName", "") or "Individual").strip()
    opportunity_title = html.escape(data.get("opportunityTitle", "") or "Untitled Opportunity")
    opportunity_type = data.get("opportunityType", "") or "Not specified"
    skills = data.get("skillsRequired", []) or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    # Determine if image was submitted by checking extractedText
    extracted_text = data.get("extractedText", "").strip()
    if extracted_text and extracted_text != "No additional context from image":
        # Use extracted text as primary source
        important_words = list(set([company_name] + skills + [word.strip() for word in extracted_text.split() if word.strip()]))
        prompt_data = {
            "companyName": company_name,
            "opportunityTitle": opportunity_title,
            "opportunityType": opportunity_type,
            "location": "Not specified",  # Default unless inferred
            "workMode": "Not specified",  # Default unless inferred
            "timeCommitment": "Not specified",  # Added default
            "skillsRequired": "Not specified",  # Default unless inferred
            "important_words": ", ".join(important_words) if important_words else "None",
            "extractedText": extracted_text,
            "wordCount": wordCount
        }
    else:
        # Use field data when no image is submitted
        location = data.get("location", "") or "Not specified"
        work_mode = data.get("workMode", "") or "Not specified"
        time_commitment = data.get("timeCommitment", "") or "Not specified"
        important_words = list(set([company_name] + skills))
        prompt_data = {
            "companyName": company_name,
            "opportunityTitle": opportunity_title,
            "opportunityType": opportunity_type,
            "location": location,
            "workMode": work_mode,
            "timeCommitment": time_commitment,
            "skillsRequired": ", ".join(skills) if skills else "Not specified",
            "important_words": ", ".join(important_words) if important_words else "None",
            "extractedText": "No additional context from image",
            "wordCount": wordCount
        }

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant"
    )

    if company_type in ["company", "Adept"]:
        format_instruction = """
Format:
<b>About the Company:</b><br>
[Brief intro to the company, culture, and mission.]<br><br>

<b>About the Opportunity:</b><br>
[Brief overview based on extracted text or form data, with bolded important words.]<br><br>

<b>Key Responsibilities:</b><br>
<ul>
    <li>[List of inferred or provided responsibilities.]</li>
</ul><br><br>

<b>Required Skills & Qualifications:</b><br>
<ul>
    <li>[List of inferred or provided skills.]</li>
</ul><br><br>

<b>Nice to Have:</b><br>
<ul>
    <li>[List of optional skills or experiences.]</li>
</ul><br><br>

<b>Perks & Benefits:</b><br>
<ul>
    <li>[Details about compensation or benefits inferred or provided.]</li>
</ul>
"""
    else:
        format_instruction = """
Format:
<b>About the Company:</b>  
[Brief intro to the company, culture, and mission.]<br>

<b>About the Opportunity:</b><br>
[Brief overview based on extracted text or form data, with bolded important words.]<br><br>

<b>Key Responsibilities:</b><br>
<ul>
    <li>[List of inferred or provided responsibilities.]</li>
</ul><br><br>

<b>Required Skills & Qualifications:</b><br>
<ul>
    <li>[List of inferred or provided skills.]</li>
</ul><br><br>

<b>Nice to Have:</b><br>
<ul>
    <li>[List of optional skills or experiences.]</li>
</ul><br><br>

<b>Perks & Benefits:</b><br>
<ul>
    <li>[Details about compensation or benefits inferred or provided.]</li>
</ul><br><br>

<b>Eligibility:</b><br>
<ul>
    <li>[Details about eligibility criteria inferred or provided.]</li>
</ul>
"""

    intro_instruction = """
    You are tasked with generating a professional opportunity description. Use the provided data to create the description:
    - If extracted text is available, analyze it to infer key details such as company name, opportunity title, opportunity type (e.g., Full time, Part time, Contract, Internship, Project), location, skills, and any other relevant information. Prioritize the extracted text for context.
    - If no extracted text is available, rely on the provided form data for all details.
    The tone should adapt to the inferred or provided opportunity type:
    - Formal and structured for Full time or Contract.
    - Encouraging and learning-focused for Internships.
    - Flexible and collaborative for Part time or Projects.
    """

    prompt = ChatPromptTemplate.from_template(f"""
    {intro_instruction}

    Details provided:
    - Extracted Text: {{extractedText}}
    - Company Name: {{companyName}}
    - Opportunity Title: {{opportunityTitle}}
    - Opportunity Type: {{opportunityType}}
    - Location: {{location}}
    - Work Mode: {{workMode}}
    - Time Commitment: {{timeCommitment}}
    - Skills Required: {{skillsRequired}}

    Important words to bold: {{important_words}}

    {format_instruction}

    Instructions:
    - Analyze the {{extractedText}} to infer details if available, otherwise use the provided form data.
    - Fill in any missing details with generic placeholders (e.g., 'Not specified') if neither extracted text nor form data provides them.
    - Ensure exactly one <br> tags between sections for clean spacing.
    - Use only <b> tags, not **, for bolding important words within paragraphs (e.g., <b>Manvian</b>).
    - Do NOT bold words within <ul> lists.
    - Ensure the response is at least {{wordCount}} words. Expand each section thoughtfully with relevant details based on the {{extractedText}} or form data.
    - Infer the opportunity type from {{extractedText}} if not provided, and adjust the tone and section accordingly.
    - Avoid mentioning the instructions or the process of inference in the output.
    - Do NOT mention any notes at the end of the description.
    """)

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(prompt_data)
    cleaned_response = clean_html_spacing(response) 
    return cleaned_response
