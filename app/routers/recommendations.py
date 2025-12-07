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
                """You are Recruito, an AI recruiter that helps evaluate job candidates.

INPUT:
- job: title, description, required_skills array, location
- candidates: array with id, name, skills array, current_status, summary
- question: recruiter's question

RESPONSE RULES:
1. Empty question → Return only: "Hello..! How may I help you today?"
2. Factual questions → Answer directly from data
3. Candidate evaluation → Analyze and return relevant candidate lists

QUESTION TYPES:
- "Who is fit/suitable?" → Return ONLY topCandidates
- "Who is rejected/not fit?" → Return ONLY rejectedCandidates
- "List all candidates?" → Return BOTH arrays
- Specific candidate questions → Evaluate that person only

CANDIDATE ANALYSIS:
- STRICT SKILLS MATCHING: Only recommend if candidate has skills that match job.required_skills
- EMPTY SKILLS RULE: If skills array is empty, candidate is NOT a fit (unless already "selected")
- STATUS RULES:
  * "selected" → ALWAYS in topCandidates (respect existing decision)
  * "shortlisted" → Can be in topCandidates if skills match, otherwise reject
  * "rejected" → Put in rejectedCandidates
  * "in_review" → Must have matching skills to be recommended
- NO MATCH = REJECT: If no skills overlap with job requirements, reject the candidate

OUTPUT FORMAT (JSON only):
CRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations before or after the JSON. Just the raw JSON object.

{{
  "answer": "Clear, concise explanation answering the question",
  "candidates": [
    {{
      "id": "candidate_id",
      "name": "Candidate Name",
      "fit": true,
      "score": 85,
      "summary": "Detailed description of candidate's background and fit",
      "reason": "Why they fit or don't fit in 1-2 sentences"
    }}
  ]
}}

GUIDELINES:
- Return ALL candidates in a single "candidates" array
- Set "fit": true if candidate matches job requirements, false otherwise
- Use actual candidate names from data
- Sort by fit (true first), then by score (highest first)
- Be helpful and direct in answer field
- When NO candidates are a good fit, say: "No candidates are a good fit for this role based on the current requirements."
- Do NOT show static messages like "Based on the provided job requirements..." when no candidates match
- Only set fit=true for candidates who actually have the required skills or are already selected""",
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

STRICT CANDIDATE EVALUATION:
- Each candidate has a `skills` array (e.g., ["Java", "Python"]).
- ONLY recommend candidates whose skills array contains matches with job.required_skills
- RULES:
  * Empty skills array [] = NO MATCH (unless status is "selected")
  * "selected" status = ALWAYS recommend (respect existing decision)
  * "in_review" with no skills = REJECT
  * No skill overlap = REJECT
- Examples:
  * Job requires ["Python"]: candidate skills ["Python", "Django"] = MATCH
  * Job requires ["Python"]: candidate skills ["Java"] = NO MATCH
  * Job requires ["Python"]: candidate skills [] = NO MATCH
  * Job requires ["Python"]: candidate skills [], status "selected" = MATCH (respect decision)

CANDIDATE EVALUATION:
- Return ALL candidates in the "candidates" array
- For each candidate, set "fit": true if they match job requirements, false otherwise
- Determine fit based on:
  * Skills match: candidate.skills overlaps with job.required_skills
  * Status: "selected" candidates always have fit=true
  * Empty skills: fit=false (unless status is "selected")
- Include for each candidate: id, name, fit (boolean), score (0-100), summary, reason
- Use actual candidate names from the data array
- Sort candidates: fit=true first, then by score (highest first)
- Be direct and clear in answer field - no static introductory messages

CRITICAL: Return ONLY the JSON object. Do NOT include any text before or after the JSON. Do NOT wrap it in markdown code blocks. Return pure JSON starting with {{ and ending with }}.""",
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

            # Try to find the outermost JSON object by matching braces
            start = cleaned.find("{")
            if start != -1:
                # Find the matching closing brace
                brace_count = 0
                end = start
                for i in range(start, len(cleaned)):
                    if cleaned[i] == "{":
                        brace_count += 1
                    elif cleaned[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break
                
                if brace_count == 0:
                    # Extract only the JSON object, ignoring any extra text
                    cleaned = cleaned[start : end + 1]
                else:
                    # Fallback: use last } if brace matching fails
                    end = cleaned.rfind("}")
                    if end != -1:
                        cleaned = cleaned[start : end + 1]

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                # Log the problematic response for debugging (first 500 chars)
                error_preview = cleaned[:500] if len(cleaned) > 500 else cleaned
                print(f"JSON parsing error. Response preview: {error_preview}")
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
        ai_candidates = parsed.get("candidates") or []

        # ------------------------------------------------------------------
        # ENFORCE BUSINESS RULES ON CANDIDATE FIT STATUS
        # ------------------------------------------------------------------
        # We re-check skills vs required_skills and current_status to ensure
        # accurate fit classification, regardless of what the AI returned.
        # ------------------------------------------------------------------

        # Build quick lookup for original candidate payloads
        candidate_map = {c.id: c for c in request.candidates}

        def get_candidate_info(c_json: dict) -> Optional[CandidateInfo]:
            cid = str(c_json.get("id") or "").strip()
            return candidate_map.get(cid)

        # Normalise required skills from job
        job_required_skills = [
            s.strip().lower()
            for s in (request.job.required_skills or [])
            if isinstance(s, str) and s.strip()
        ]

        def compute_fit(c_info: CandidateInfo) -> tuple[bool, bool, list[str]]:
            """
            Returns:
              (is_fit, has_any_skills, matching_skills)
            """
            # Candidate skills (already include keywords from frontend)
            cand_skills = [
                s.strip().lower()
                for s in (c_info.skills or [])
                if isinstance(s, str) and s.strip()
            ]
            has_any_skills = len(cand_skills) > 0

            # Status-based overrides
            status = (c_info.current_status or "").lower()
            if status == "selected":
                # Always considered a fit – recruiter already chose them
                return True, has_any_skills, []

            # Strict skills matching for all non-selected candidates
            if not job_required_skills:
                # If job has no explicit required skills defined, treat everyone
                # as potentially fit based on status/other data
                return has_any_skills, has_any_skills, []

            matching = [
                s
                for s in cand_skills
                if any(
                    rs in s or s in rs  # substring match to be a bit flexible
                    for rs in job_required_skills
                )
            ]

            is_fit = len(matching) > 0
            return is_fit, has_any_skills, matching

        # Process all candidates from AI response and enforce our business rules
        processed_candidates: list[dict] = []

        for c_json in ai_candidates:
            if not isinstance(c_json, dict):
                continue
            
            c_info = get_candidate_info(c_json)
            if not c_info:
                continue

            # Re-compute fit based on our strict rules
            is_fit, has_any_skills, matching = compute_fit(c_info)

            # Update the fit status in the candidate object
            c_json["fit"] = is_fit

            # Clean up misleading reasons like "No match found"
            reason = str(c_json.get("reason") or "").strip()
            if not reason or "no match" in reason.lower() or "no required skills" in reason.lower():
                if is_fit:
                    if matching:
                        reason = (
                            "Matches required skills: "
                            + ", ".join(sorted(set(matching))).title()
                        )
                    elif c_info.current_status and c_info.current_status.lower() == "selected":
                        reason = "Already selected for this role."
                    else:
                        reason = "Good fit based on skills and profile."
                else:
                    if not has_any_skills:
                        reason = "No relevant skills listed in the candidate's profile for this role."
                    else:
                        reason = (
                            "Has skills, but none match the job's required skills: "
                            + ", ".join(sorted(set(job_required_skills))).title()
                        )
                c_json["reason"] = reason

            # Ensure score is present (default to 0 if missing)
            if "score" not in c_json:
                c_json["score"] = 90 if is_fit else 30

            processed_candidates.append(c_json)

        # Sort: fit=true first, then by score (highest first)
        processed_candidates.sort(
            key=lambda x: (not x.get("fit", False), -x.get("score", 0))
        )

        return {
            "answer": answer,
            "candidates": processed_candidates,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to screen candidates: {str(e)}",
        )

