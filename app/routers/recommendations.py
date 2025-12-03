from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import httpx
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
try:
    from langchain_core.output_parsers import JsonOutputParser
except Exception:
    JsonOutputParser = None

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Backend API URL - should be configurable via environment variable
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


class ProfileData(BaseModel):
    keywords: Optional[List[str]] = []
    skills: Optional[List[Dict[str, Any]]] = []
    languages: Optional[List[Dict[str, Any]]] = []
    location: Optional[str] = None
    city: Optional[str] = None
    userId: str


class RecommendationRequest(BaseModel):
    profile: ProfileData
    limit: Optional[int] = 20
    token: Optional[str] = None  # Backend API token for fetching opportunities


def build_recommendation_chain():
    """Build the LLM chain for matching opportunities with profile"""
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=LLM_MODEL_NAME
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert job matching AI. Your task is to analyze a user's profile and match them with relevant opportunities.
        
Given a user profile (keywords, skills, languages, location) and a list of opportunities, you need to:
1. Score each opportunity based on relevance (0-100)
2. Consider: keyword matches, skill matches, language preferences, location proximity
3. Return a JSON array of opportunity IDs with their relevance scores, sorted by score (highest first)

Return ONLY valid JSON in this format:
[
  {{"id": "opportunity_id_1", "score": 95, "reason": "Strong match: 5 skills match, keywords align"}},
  {{"id": "opportunity_id_2", "score": 80, "reason": "Good match: 3 skills match, location nearby"}}
]

Be strict with scoring:
- 90-100: Excellent match (most skills/keywords match, location matches)
- 70-89: Good match (several skills/keywords match)
- 50-69: Moderate match (some overlap)
- Below 50: Weak match (minimal overlap)

IMPORTANT: Return ONLY the JSON array, no markdown, no explanations."""),
        ("human", """User Profile:
Keywords: {keywords}
Skills: {skills}
Languages: {languages}
Location: {location}
City: {city}

Opportunities:
{opportunities}

Return the top {limit} most relevant opportunities as a JSON array sorted by score.
Consider location proximity - opportunities in the same city should get higher scores (add 10-15 points for same city matches).
Analyze the full address to identify the city name."""),
    ])
    
    if JsonOutputParser is not None:
        return prompt | llm | JsonOutputParser()
    return prompt | llm | StrOutputParser()


class CandidateInfo(BaseModel):
    """Lightweight candidate profile used for AI screening."""
    id: str
    name: str
    summary: Optional[str] = ""
    skills: Optional[List[str]] = []
    experience_years: Optional[float] = None
    education: Optional[str] = None
    current_status: Optional[str] = None


class JobInfo(BaseModel):
    """Job / requirement details passed from the recruiter screen."""
    id: Optional[str] = None
    title: str
    company: Optional[str] = None
    description: str
    required_skills: Optional[List[str]] = []
    nice_to_have: Optional[str] = None
    location: Optional[str] = None
    post_type: Optional[str] = None


class ScreeningHistoryItem(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ScreeningRequest(BaseModel):
    """
    Request payload for the recruiter AI assistant on the requirement screen.
    """
    job: JobInfo
    candidates: List[CandidateInfo]
    question: str
    history: Optional[List[ScreeningHistoryItem]] = []
    top_k: Optional[int] = 5


def build_screening_chain():
    """
    Build the LLM chain used for screening candidates for a single requirement.
    The chain is optimised to return a structured JSON response that the
    frontend can turn into the UI shown in the design (recommended + rejected lists).
    """
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=LLM_MODEL_NAME,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Recruito, a friendly and proactive AI recruiter assistant. You help hiring managers screen candidates for job requirements with a conversational, helpful approach.

You will receive:
- `job`: JSON with fields like title, company, description, required_skills, nice_to_have, location.
- `candidates`: JSON array where each item has: id, name, summary, skills (ARRAY of skill names), experience_years, education, current_status.
  * CRITICAL: The `skills` field is an ARRAY of skill names (e.g., ["Java", "Python", "Spring"]). ALWAYS check this array when analyzing candidates.
  * The `summary` field may contain skills in text format, but the `skills` array is the PRIMARY source of truth for candidate skills.
  * If `skills` array is empty or missing, then the candidate truly has no skills listed.
  * CRITICAL: The `current_status` field shows the candidate's current application status: "selected", "shortlisted", "in_review", "rejected", "on_hold", "interviewed".
  * IMPORTANT: If a candidate's `current_status` is "selected", they have ALREADY been chosen for this role. You should NOT reject them - place them in topCandidates and acknowledge their selected status.
  * If `current_status` is "shortlisted", they are already being considered favorably - factor this into your assessment.
  * Respect the recruiter's existing decisions - if they've already selected/shortlisted someone, acknowledge that in your response.
- `question`: recruiter question in natural language (or empty string for initial greeting).
- `history`: prior chat turns between recruiter (user) and assistant (you).

CRITICAL RULE: If the question asks about candidate fit, suitability, evaluation, or comparison, you MUST analyze the candidates and provide topCandidates/rejectedCandidates. DO NOT return a greeting if asked about candidate fit.

SPECIAL BEHAVIOR:
- If `question` is empty or `history` is empty, provide ONLY a simple warm greeting: "Hello..! How may I help you today?" - DO NOT ask any questions. Wait for the user to ask questions.
- Answer factual questions directly based on the provided data:
  * "How many candidates applied?" → Answer with the count from the candidates array
  * "What skills are required?" → List the required_skills from the job
  * "What is the job location?" → Answer from job.location
  * Any other factual question → Answer based on the job or candidates data provided

QUESTION-BASED RESPONSE LOGIC (CRITICAL):
Based on the question asked, return ONLY the relevant candidates:

1. Questions asking for FIT/SUITABLE/RECOMMENDED candidates:
   * "Who are fit for this role?" → Return ONLY topCandidates (leave rejectedCandidates empty)
   * "List suitable candidates" → Return ONLY topCandidates
   * "Show me recommended candidates" → Return ONLY topCandidates
   * "Who should I hire?" → Return ONLY topCandidates
   * "Which candidates match?" → Return ONLY topCandidates

2. Questions asking for REJECTED/NOT FIT candidates:
   * "Who are rejected?" → Return ONLY rejectedCandidates (leave topCandidates empty)
   * "Show rejected candidates" → Return ONLY rejectedCandidates
   * "Who is not fit?" → Return ONLY rejectedCandidates
   * "List candidates who don't match" → Return ONLY rejectedCandidates

3. Questions asking for ALL candidates:
   * "List all candidates" → Return BOTH topCandidates and rejectedCandidates
   * "Show me all applicants" → Return BOTH topCandidates and rejectedCandidates
   * "Who applied?" → Return BOTH topCandidates and rejectedCandidates

4. Questions about a SPECIFIC candidate:
   * "Is [name] fit for this job?" → Analyze that candidate, put in topCandidates if fit, rejectedCandidates if not fit
   * "Should I hire [name]?" → Analyze that candidate, put in appropriate array

5. General screening questions (no specific direction):
   * "Screen the candidates" → Return BOTH topCandidates and rejectedCandidates
   * "Analyze candidates" → Return BOTH topCandidates and rejectedCandidates

IMPORTANT: Only populate the arrays that are relevant to the question. If asked about "fit candidates", only fill topCandidates. If asked about "rejected", only fill rejectedCandidates.

- CRITICAL: When analyzing candidates, ALWAYS check the `skills` array field in the candidate JSON. This is the primary source of their technical skills.
- CRITICAL: ALWAYS check the `current_status` field before making recommendations:
  * If `current_status` is "selected" → Candidate is ALREADY chosen. Place in topCandidates and acknowledge their selected status. DO NOT reject them.
  * If `current_status` is "shortlisted" → Candidate is already being considered. Factor this into assessment, likely place in topCandidates.
  * If `current_status` is "rejected" → Candidate was already rejected. You can place in rejectedCandidates, but mention the existing rejection.
  * If `current_status` is "in_review" → No decision yet, analyze based on skills and fit.
- If a candidate has skills in the `skills` array, use those skills for matching against job requirements.
- If the `skills` array is empty or missing, then the candidate has no listed skills - mention this in your assessment.
- If a candidate has empty skills array but has other information (summary, experience, education), analyze based on available information and mention the lack of explicit skills.
- IMPORTANT: Respect existing recruiter decisions. If someone is already "selected", they should be recommended, not rejected, even if skills don't perfectly match.
- Be conversational, friendly, and professional.
- Always answer based on the actual data provided in the job and candidates JSON.
- When recommending candidates, provide detailed, personalized descriptions that highlight why each candidate fits the role.
- NEVER return a greeting when asked about candidate fit, suitability, or evaluation - always analyze and respond.

SCORING RULES (0‑100):
- 90–100: Excellent match (strong skills & relevant experience) OR candidate is already "selected"
- 75–89: Good match (solid skills, some gaps ok) OR candidate is "shortlisted"
- 60–74: Moderate match (partial fit)
- Below 60: Weak fit (use mainly in rejectedCandidates, BUT NOT if status is "selected" or "shortlisted")
- SPECIAL: If candidate.current_status is "selected", give them a score of 90-100 and place in topCandidates
- SPECIAL: If candidate.current_status is "shortlisted", give them a score of 75-89 and place in topCandidates

IMPORTANT OUTPUT FORMAT:
Return ONLY valid JSON with this exact structure – no markdown, no extra text:
{{
  "answer": "Your conversational response. 
  - If question is empty/initial greeting: ONLY say 'Hello..! How may I help you today?' - no questions.
  - If asked about candidate fit/suitability: Analyze the candidate(s) and explain your assessment. DO NOT return a greeting.
  - If recommending candidates: Provide a natural explanation followed by candidate details.",
  "topCandidates": [
    {{
      "id": "candidate_id_1",
      "name": "Full Name",
      "score": 95,
      "summary": "Detailed 3-4 sentence description highlighting their relevant experience, key projects, skills, and work traits that make them a great fit for this specific role",
      "reason": "Why this person is a strong fit in 1‑2 sentences"
    }}
  ],
  "rejectedCandidates": [
    {{
      "id": "candidate_id_2",
      "name": "Full Name",
      "summary": "Brief summary of the candidate's background and skills (even if not matching)",
      "reason": "Clear, short reason why they are not a good fit for this specific role"
    }}
  ]
}}

- Always keep `topCandidates` sorted by score (highest first).
- Try to limit `topCandidates` to at most `top_k` items.
- Make candidate summaries detailed and personalized - mention specific skills, projects, and traits.
- For rejectedCandidates, ALWAYS include: id, name, summary (brief background), and reason (why not a fit).
- Include candidate names in both topCandidates and rejectedCandidates - use the actual candidate name from the candidates array.
- If everyone is weak, still return the best few but explain concerns in `answer`.
- CRITICAL: When asked "is [name] fit for this job?" or similar screening questions, you MUST analyze and respond with assessment. Never return just a greeting for screening questions.
- IMPORTANT: Always populate the "name" field in both topCandidates and rejectedCandidates using the candidate's actual name from the provided candidates data.""",
            ),
            (
                "human",
                """Job (JSON):
{job_json}

Candidates (JSON array):
{candidates_json}

CRITICAL INSTRUCTIONS FOR READING CANDIDATE DATA:
- Each candidate object has a "skills" field which is an ARRAY of skill names (e.g., ["Java", "Python", "Spring"]).
- Each candidate object has a "current_status" field showing their application status: "selected", "shortlisted", "in_review", "rejected", "on_hold", "interviewed".
- You MUST check BOTH the "skills" array AND the "current_status" field.
- CRITICAL STATUS RULES:
  * If candidate.current_status is "selected" → They are ALREADY chosen. Place in topCandidates, acknowledge selection. DO NOT reject.
  * If candidate.current_status is "shortlisted" → They are already being considered. Likely place in topCandidates.
  * If candidate.current_status is "rejected" → They were already rejected. Can place in rejectedCandidates.
  * If candidate.current_status is "in_review" → Analyze based on skills match.
- To determine if a candidate matches job requirements:
  * FIRST: Check current_status - if "selected", they should be recommended regardless of skills
  * Look at job.required_skills (array of required skills)
  * Look at candidate.skills (array of candidate's skills)
  * If there's overlap, the candidate is a match
  * If no overlap, the candidate is not a match (unless status is "selected")
- Example: Job requires ["Java"], Candidate has skills: ["Java", "Spring"], status: "in_review" → MATCH
- Example: Job requires ["Java"], Candidate has skills: ["Python"], status: "in_review" → NO MATCH
- Example: Job requires ["Java"], Candidate has skills: [], status: "selected" → MATCH (already selected, respect decision)
- Example: Job requires ["Java"], Candidate has skills: [], status: "in_review" → NO MATCH (empty array means no skills)

Previous conversation (JSON array of {{role, content}}):
{history_json}

Recruiter question:
{question}

Remember: 
- If this is the first message (empty history), ONLY greet with "Hello..! How may I help you today?" - DO NOT ask any questions.
- Answer factual questions directly using the provided job and candidates data. For example:
  * "How many candidates applied?" → Answer: "X candidate(s) have applied for this position" (count from candidates array)
  * "What are the required skills?" → List the required_skills from job
  * "Tell me about the candidates" → Provide information about the candidates

CRITICAL - READING CANDIDATE DATA:
- Each candidate in the candidates array has a `skills` field which is an ARRAY of skill names (e.g., ["Java", "Python"]).
- ALWAYS check the `skills` array when evaluating if a candidate matches job requirements.
- Example: If job requires ["Java"] and candidate has skills: ["Java", "Spring"], they ARE a match.
- Example: If job requires ["Java"] and candidate has skills: ["Python"], they are NOT a match.
- Example: If job requires ["Java"] and candidate has skills: [] (empty array), they have NO skills and are NOT a match.
- DO NOT say "no listed skills" if the skills array contains skills - check the actual array values!

CRITICAL - QUESTION-BASED RESPONSE:
Analyze the question to determine what the user wants to see:

1. If question asks for FIT/SUITABLE/RECOMMENDED:
   - Keywords: "fit", "suitable", "recommended", "best", "good", "match", "should hire"
   - Action: Fill ONLY topCandidates, leave rejectedCandidates as empty array []
   - Example: "Who are fit for this role?" → Only topCandidates

2. If question asks for REJECTED/NOT FIT:
   - Keywords: "rejected", "not fit", "don't match", "unsuitable", "bad fit"
   - Action: Fill ONLY rejectedCandidates, leave topCandidates as empty array []
   - Example: "Who are rejected?" → Only rejectedCandidates

3. If question asks for ALL/ALL CANDIDATES:
   - Keywords: "all candidates", "all applicants", "everyone", "list all"
   - Action: Fill BOTH topCandidates and rejectedCandidates
   - Example: "List all candidates" → Both arrays

4. If question asks about SPECIFIC candidate:
   - Pattern: "Is [name]...", "Should I hire [name]"
   - Action: Analyze that candidate, put in appropriate array based on fit
   - Example: "Is John fit?" → If fit: topCandidates, if not: rejectedCandidates

5. General screening (no specific direction):
   - Action: Fill BOTH arrays to show complete analysis

- CRITICAL: Always include the candidate's actual "name" from the candidates array. Do NOT use generic names like "Candidate 1" or "Full Name" - use the real name from the candidates data.
- CRITICAL: Before placing a candidate in rejectedCandidates, check their current_status:
  * If current_status is "selected" → DO NOT place in rejectedCandidates, place in topCandidates instead and acknowledge their selection
  * If current_status is "shortlisted" → Consider placing in topCandidates, not rejectedCandidates
  * Respect the recruiter's existing decisions - never reject someone who is already selected
- For rejectedCandidates, include: id, name (actual name from candidates), summary (brief background), and reason (why not a fit).
- For topCandidates, if the candidate's current_status is "selected" or "shortlisted", mention this in the reason/summary (e.g., "Already selected for this role" or "Currently shortlisted").
- Be conversational and helpful. Use the conversation history to maintain context.
- Provide detailed, personalized candidate descriptions when recommending.
- If a candidate has no skills listed, mention this in your assessment but still evaluate based on available information.
- When listing candidates, always show their actual names from the candidates array, not generic placeholders.
- IMPORTANT: Never reject a candidate who is already "selected" - acknowledge their selection and place them in topCandidates.
- Respond with JSON ONLY in the specified schema.""",
            ),
        ]
    )

    # We deliberately use StrOutputParser and then do robust JSON parsing
    # so that we work even if langchain_core JsonOutputParser is unavailable.
    return prompt | llm | StrOutputParser()


async def fetch_opportunities_from_backend(token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all live opportunities from the backend API"""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch explore requirements endpoint
            response = await client.get(
                f"{BACKEND_API_URL}/requirements/explore",
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch opportunities: {response.text}"
                )
            
            data = response.json()
            # Handle different response formats
            if isinstance(data, dict) and "requirements" in data:
                return data["requirements"]
            elif isinstance(data, list):
                return data
            else:
                return []
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Backend API timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to backend: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching opportunities: {str(e)}")


def extract_skill_names(skills: List[Dict[str, Any]]) -> List[str]:
    """Extract skill names from skill objects"""
    if not skills:
        return []
    return [skill.get("name", "") for skill in skills if isinstance(skill, dict) and skill.get("name")]


def extract_language_names(languages: List[Dict[str, Any]]) -> List[str]:
    """Extract language names from language objects"""
    if not languages:
        return []
    return [lang.get("name", "") for lang in languages if isinstance(lang, dict) and lang.get("name")]


async def extract_city_from_address_ai(address: str) -> str:
    """Extract city name from address using AI"""
    if not address:
        return ""
    
    try:
        llm = ChatGroq(
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=LLM_MODEL_NAME
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting city names from Indian addresses. 
Given an address, identify the PRIMARY CITY NAME (not area, locality, or landmark).

Important rules:
1. Extract the main city name (e.g., "Coimbatore", "Chennai", "Mumbai", "Bangalore")
2. If the address mentions an area/locality (like "Gandhipuram", "T Nagar", "Koramangala"), identify which city that area belongs to
3. Common mappings: Gandhipuram/RS Puram → Coimbatore, T Nagar/Anna Nagar → Chennai, Koramangala/Indiranagar → Bangalore
4. Look for state names to help identify the city (Tamil Nadu → Chennai/Coimbatore/Madurai, Karnataka → Bangalore/Mysore, etc.)
5. Return ONLY the city name, nothing else. No explanations, no quotes, just the city name.
6. If you cannot determine the city, return an empty string.

Examples:
- "Gandhipuram, Tamil Nadu" → "Coimbatore"
- "T Nagar, Chennai" → "Chennai"
- "Koramangala, Bangalore" → "Bangalore"
- "Andheri, Mumbai" → "Mumbai"
"""),
            ("human", "Extract the PRIMARY CITY NAME from this address:\n{address}\n\nCity name:"),
        ])
        
        chain = prompt | llm | StrOutputParser()
        result = await chain.ainvoke({"address": address})
        
        # Clean the result - remove quotes, whitespace, and any extra text
        city = result.strip().strip('"').strip("'").strip()
        
        # Remove common suffixes that might be added
        city = city.split(',')[0].strip()
        city = city.split('\n')[0].strip()
        city = city.split('.')[0].strip()
        
        return city if city else ""
    except Exception as e:
        print(f"AI city extraction failed: {str(e)}")
        # Fallback: try simple pattern matching
        import re
        # Look for city before state
        state_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,\s*(?:Tamil Nadu|Maharashtra|Karnataka|Telangana|West Bengal|Gujarat|Rajasthan|Uttar Pradesh|Madhya Pradesh|Andhra Pradesh|Bihar|Punjab|Haryana|Jammu and Kashmir|Kerala|Odisha|Assam|Jharkhand|Chhattisgarh|Himachal Pradesh|Uttarakhand|Goa|Manipur|Meghalaya|Mizoram|Nagaland|Tripura|Arunachal Pradesh|Sikkim)'
        match = re.search(state_pattern, address, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""


@router.post("/ai-recommended")
async def get_ai_recommended_opportunities(request: RecommendationRequest):
    """
    Get AI-recommended opportunities based on user profile
    """
    try:
        # Validate profile data
        if not request.profile.userId:
            raise HTTPException(status_code=400, detail="userId is required")
        
        # Fetch opportunities from backend
        opportunities = await fetch_opportunities_from_backend(request.token)
        
        if not opportunities:
            return {
                "recommended": [],
                "totalCount": 0,
                "message": "No opportunities available"
            }
        
        # Prepare profile data for AI
        keywords = request.profile.keywords or []
        skills = extract_skill_names(request.profile.skills or [])
        languages = extract_language_names(request.profile.languages or [])
        location = request.profile.location or ""
        user_city = request.profile.city or ""
        
        # Extract city from location using AI if not provided
        if not user_city and location:
            user_city = await extract_city_from_address_ai(location)
        
        # Prepare opportunities data for AI (limit to first 100 for performance)
        opportunities_for_ai = opportunities[:100]
        opportunities_summary = []
        for opp in opportunities_for_ai:
            opp_location = opp.get("address", "") or ""
            opp_city = await extract_city_from_address_ai(opp_location)
            
            opp_summary = {
                "id": str(opp.get("_id") or opp.get("id", "")),
                "title": opp.get("proj_name", ""),
                "skills": opp.get("skills", []),
                "keywords": opp.get("keywords", []),
                "location": opp_location,
                "city": opp_city,
                "description": opp.get("description", "")[:200] if opp.get("description") else "",  # Truncate for AI
            }
            opportunities_summary.append(opp_summary)
        
        # Build AI chain
        chain = build_recommendation_chain()
        
        # Prepare input for AI
        keywords_str = ", ".join(keywords) if keywords else "None"
        skills_str = ", ".join(skills) if skills else "None"
        languages_str = ", ".join(languages) if languages else "None"
        location_str = location if location else "Not specified"
        city_str = user_city if user_city else "Not specified"
        
        # Format opportunities for AI
        opportunities_json = "\n".join([
            f"ID: {opp['id']}, Title: {opp['title']}, Skills: {', '.join(opp['skills'])}, Keywords: {', '.join(opp['keywords'])}, Location: {opp['location']}, City: {opp.get('city', 'Not specified')}"
            for opp in opportunities_summary
        ])
        
        # Call AI
        try:
            result = await chain.ainvoke({
                "keywords": keywords_str,
                "skills": skills_str,
                "languages": languages_str,
                "location": location_str,
                "city": city_str,
                "opportunities": opportunities_json,
                "limit": request.limit or 20
            })
            
            # Parse result
            if isinstance(result, str):
                import json
                # Try to extract JSON from string
                cleaned = result.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                # Find JSON array
                start = cleaned.find("[")
                end = cleaned.rfind("]")
                if start != -1 and end != -1:
                    cleaned = cleaned[start:end+1]
                
                try:
                    scored_opportunities = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Fallback: use simple keyword matching
                    scored_opportunities = fallback_matching(opportunities_summary, keywords, skills, location, user_city)
            else:
                scored_opportunities = result
            
            # Validate and process scored opportunities
            if not isinstance(scored_opportunities, list):
                scored_opportunities = fallback_matching(opportunities_summary, keywords, skills, location, user_city)
            
            # Filter by minimum score (50) and sort by score
            scored_opportunities = [
                item for item in scored_opportunities 
                if isinstance(item, dict) and item.get("score", 0) >= 50
            ]
            scored_opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Limit results
            top_opportunities = scored_opportunities[:request.limit or 20]
            
            # Map back to full opportunity data
            opportunity_map = {str(opp.get("_id") or opp.get("id", "")): opp for opp in opportunities}
            recommended = []
            
            for scored in top_opportunities:
                opp_id = scored.get("id")
                if opp_id and opp_id in opportunity_map:
                    opp_data = opportunity_map[opp_id].copy()
                    opp_data["ai_score"] = scored.get("score", 0)
                    opp_data["ai_reason"] = scored.get("reason", "")
                    
                    # Check if opportunity is nearby based on city
                    opp_location = opp_data.get("address", "") or ""
                    is_nearby = False
                    
                    if user_city and opp_location:
                        # Extract city from opportunity location using AI
                        opp_city = await extract_city_from_address_ai(opp_location)
                        # Check if cities match (case-insensitive)
                        if user_city.lower() == opp_city.lower() and user_city:
                            is_nearby = True
                    
                    opp_data["isNearby"] = is_nearby
                    recommended.append(opp_data)
            
            return {
                "recommended": recommended,
                "totalCount": len(recommended)
            }
            
        except Exception as ai_error:
            # Fallback to simple matching if AI fails
            print(f"AI matching failed, using fallback: {str(ai_error)}")
            scored_opportunities = fallback_matching(opportunities_summary, keywords, skills, location, user_city)
            top_opportunities = scored_opportunities[:request.limit or 20]
            
            opportunity_map = {str(opp.get("_id") or opp.get("id", "")): opp for opp in opportunities}
            recommended = []
            
            for scored in top_opportunities:
                opp_id = scored.get("id")
                if opp_id and opp_id in opportunity_map:
                    opp_data = opportunity_map[opp_id].copy()
                    opp_data["ai_score"] = scored.get("score", 0)
                    opp_data["ai_reason"] = scored.get("reason", "Matched based on keywords and skills")
                    
                    # Check if opportunity is nearby based on city
                    opp_location = opp_data.get("address", "") or ""
                    is_nearby = False
                    
                    if user_city and opp_location:
                        # Extract city from opportunity location using AI
                        opp_city = await extract_city_from_address_ai(opp_location)
                        # Check if cities match (case-insensitive)
                        if user_city.lower() == opp_city.lower() and user_city:
                            is_nearby = True
                    
                    opp_data["isNearby"] = is_nearby
                    recommended.append(opp_data)
            
            return {
                "recommended": recommended,
                "totalCount": len(recommended)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AI recommendations: {str(e)}")


def fallback_matching(
    opportunities: List[Dict[str, Any]], 
    keywords: List[str], 
    skills: List[str], 
    location: str
) -> List[Dict[str, Any]]:
    """Fallback matching algorithm when AI fails"""
    scored = []
    
    keywords_lower = [k.lower() for k in keywords]
    skills_lower = [s.lower() for s in skills]
    location_lower = location.lower() if location else ""
    
    for opp in opportunities:
        score = 0
        reasons = []
        
        # Match keywords
        opp_keywords = [k.lower() for k in (opp.get("keywords", []) or [])]
        keyword_matches = sum(1 for kw in keywords_lower if any(kw in okw or okw in kw for okw in opp_keywords))
        if keyword_matches > 0:
            score += keyword_matches * 15
            reasons.append(f"{keyword_matches} keyword(s) match")
        
        # Match skills
        opp_skills = [s.lower() if isinstance(s, str) else str(s).lower() for s in (opp.get("skills", []) or [])]
        skill_matches = sum(1 for sk in skills_lower if any(sk in os or os in sk for os in opp_skills))
        if skill_matches > 0:
            score += skill_matches * 20
            reasons.append(f"{skill_matches} skill(s) match")
        
        # Match location
        opp_location = (opp.get("location", "") or "").lower()
        if location_lower and opp_location:
            if location_lower in opp_location or opp_location in location_lower:
                score += 25
                reasons.append("Location matches")
        
        # Match in description
        description = (opp.get("description", "") or "").lower()
        desc_keyword_matches = sum(1 for kw in keywords_lower if kw in description)
        if desc_keyword_matches > 0:
            score += desc_keyword_matches * 5
            if desc_keyword_matches > 0:
                reasons.append(f"{desc_keyword_matches} keyword(s) in description")
        
        if score > 0:
            scored.append({
                "id": opp.get("id", ""),
                "score": min(score, 100),  # Cap at 100
                "reason": ", ".join(reasons) if reasons else "Basic match"
            })
    
    # Sort by score
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


@router.post("/extract-city")
async def extract_city_endpoint(request: dict):
    """Extract city name from address using AI"""
    address = request.get("address", "")
    if not address:
        return {"city": ""}
    
    try:
        city = await extract_city_from_address_ai(address)
        return {"city": city}
    except Exception as e:
        print(f"Error extracting city: {str(e)}")
        # Fallback: simple pattern matching
        import re
        state_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,\s*(?:Tamil Nadu|Maharashtra|Karnataka|Telangana|West Bengal|Gujarat|Rajasthan|Uttar Pradesh|Madhya Pradesh|Andhra Pradesh|Bihar|Punjab|Haryana|Jammu and Kashmir|Kerala|Odisha|Assam|Jharkhand|Chhattisgarh|Himachal Pradesh|Uttarakhand|Goa|Manipur|Meghalaya|Mizoram|Nagaland|Tripura|Arunachal Pradesh|Sikkim)'
        match = re.search(state_pattern, address, re.IGNORECASE)
        if match:
            return {"city": match.group(1)}
        return {"city": ""}


@router.post("/screen-candidates")
async def screen_candidates(request: ScreeningRequest):
    """
    Screen candidates for a single requirement and answer recruiter questions.

    This powers the AI assistant on the requirement details screen in the
    frontend. It returns both a natural‑language answer and structured
    candidate lists that can be rendered as UI cards.
    """
    import json

    try:
        if not request.candidates:
            raise HTTPException(
                status_code=400, detail="At least one candidate is required"
            )

        # Build chain
        chain = build_screening_chain()

        # Serialize payloads for the prompt
        job_json = json.dumps(
            request.job.model_dump(), ensure_ascii=False, indent=2
        )
        candidates_json = json.dumps(
            [c.model_dump() for c in request.candidates],
            ensure_ascii=False,
            indent=2,
        )
        history_json = json.dumps(
            [h.model_dump() for h in request.history] if request.history else [],
            ensure_ascii=False,
            indent=2,
        )

        top_k = request.top_k or 5

        # Handle initial greeting - if no history and empty/question is greeting-like, send empty question
        question_text = request.question.strip() if request.question else ""
        is_initial = not request.history or len(request.history) == 0
        
        # If it's the initial interaction, send empty question to trigger greeting
        # BUT: If question is about candidate fit/suitability, always process it (not a greeting)
        is_screening_question = question_text and any(keyword in question_text.lower() for keyword in [
            'fit', 'suitable', 'match', 'recommend', 'should i hire', 'who is', 'which candidate',
            'evaluate', 'assess', 'good fit', 'right fit', 'qualified', 'qualify'
        ])
        
        if is_initial and (not question_text or question_text.lower() in ["hi", "hello", "hey", "start", ""]) and not is_screening_question:
            question_text = ""

        raw_result = await chain.ainvoke(
            {
                "job_json": job_json,
                "candidates_json": candidates_json,
                "history_json": history_json,
                "question": question_text,
                "top_k": top_k,
            }
        )

        # Robust JSON extraction similar to the opportunity recommender
        if isinstance(raw_result, str):
            cleaned = raw_result.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Try to find the outermost JSON object
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI response was not valid JSON: {e}",
                )
        else:
            # If the chain returns a parsed object already
            parsed = raw_result

        # Basic validation / normalisation of structure
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=500,
                detail="AI response was not a JSON object",
            )

        answer = parsed.get(
            "answer",
            "I analysed the candidates, but could not generate a detailed explanation.",
        )
        top_candidates = parsed.get("topCandidates") or []
        rejected_candidates = parsed.get("rejectedCandidates") or []

        # Enforce top_k limit defensively
        if isinstance(top_candidates, list) and len(top_candidates) > top_k:
            top_candidates = top_candidates[:top_k]

        return {
            "answer": answer,
            "topCandidates": top_candidates,
            "rejectedCandidates": rejected_candidates,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to screen candidates: {str(e)}",
        )

