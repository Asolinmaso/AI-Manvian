from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import re
from ..utils.description_generator import generate_description, generate_pass_opportunity_description, clean_html_spacing
from ..utils.extraction import extract_text_from_file

router = APIRouter(prefix="/description", tags=["description"])


class DescriptionRequest(BaseModel):
    wordCount: Optional[int] = 800
    companyType: Optional[str] = "company"
    companyName: Optional[str] = None
    opportunityTitle: Optional[str] = None
    opportunityType: Optional[str] = None
    postType: Optional[str] = None
    location: Optional[str] = None
    workMode: Optional[str] = None
    numberOfOpenings: Optional[int] = None
    lastDate: Optional[str] = None
    skillsRequired: Optional[str] = None
    timeCommitment: Optional[str] = None
    salaryMin: Optional[float] = None
    salaryMax: Optional[float] = None
    salaryOption: Optional[str] = None
    title: Optional[str] = None
    address: Optional[str] = None
    vacancy: Optional[int] = None
    skills: Optional[str] = None
    keywords: Optional[str] = None
    eligibility: Optional[str] = None
    package: Optional[str] = None
    workDuration: Optional[str] = None
    individualCompanyName: Optional[str] = None
    extractedText: Optional[str] = None
    # Additional fields for enhanced CROP form
    yourName: Optional[str] = None
    yourIdentity: Optional[str] = None
    educationRequirements: Optional[str] = None
    industryExpertise: Optional[str] = None
    preferredExperience: Optional[str] = None  # Changed to string (formatted from frontend)
    languagePreference: Optional[str] = None
    genderPreference: Optional[str] = None


@router.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from uploaded image or document using unified extraction utility"""
    try:
        extracted_text = await extract_text_from_file(file)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the provided file.")
        
        return {"text": extracted_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/generate")
async def generate_description_endpoint(data: DescriptionRequest):
    """Generate opportunity description from form data"""
    try:
        # Convert Pydantic model to dict
        data_dict = data.dict()
        
        # Validate mandatory fields based on company type
        if data.companyType == "company" or "Adept" in str(data_dict):
            mandatory_fields = {
                "companyName": "Company Name",
                "opportunityTitle": "Opportunity Title", 
                "opportunityType": "Opportunity Type",
                "location": "Location",
                "workMode": "Work Mode",
                "numberOfOpenings": "Number of Openings",
                "lastDate": "Last Date to Apply",
                "skillsRequired": "Skills Required",
                "timeCommitment": "Time Commitment",
                "salaryMin": "Minimum Salary",
                "salaryMax": "Maximum Salary"
            }
        else:
            mandatory_fields = {
                "postType": "Post Type",
                "location": "Location", 
                "address": "Address",
                "title": "Title",
                "lastDate": "Last Date",
                "vacancy": "Vacancy",
                "skills": "Skills"
            }
            
            # Skip package validation for unpaid internships or "Prefer Not to Disclose"
            salary_option = data.salaryOption or ""
            if (data.postType == "Internship (Unpaid)" or 
                not salary_option or 
                salary_option.lower() == "prefer not to disclose"):
                pass  # Skip package validation
            else:
                mandatory_fields["package"] = "Package"

        # Validate mandatory fields
        for field, field_name in mandatory_fields.items():
            value = getattr(data, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                # Use default values instead of failing validation
                continue
                
            # Validate numeric fields
            if field in ["numberOfOpenings", "vacancy"] and value is not None:
                if not isinstance(value, (int, float)) or value <= 0:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Required field '{field_name}' must be a positive number"
                    )
            elif field in ["salaryMin", "salaryMax"] and value is not None:
                if not isinstance(value, (int, float)) or value < 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Required field '{field_name}' must be a non-negative number"
                    )

        # Additional validations for company type
        if data.companyType == "company" or "Adept" in str(data_dict):
            if data.salaryMin and data.salaryMax and data.salaryMin > data.salaryMax:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum salary must be greater than or equal to minimum salary"
                )
            
            # Process skills as list
            if data.skillsRequired and isinstance(data.skillsRequired, str):
                data_dict["skillsRequired"] = [s.strip() for s in data.skillsRequired.split(",") if s.strip()]
        else:
            # Process individual form data
            if data.skills and isinstance(data.skills, str):
                data_dict["skills"] = [s.strip() for s in data.skills.split(",") if s.strip()]
            if data.keywords and isinstance(data.keywords, str):
                data_dict["keywords"] = [k.strip() for k in data.keywords.split(",") if k.strip()]
            else:
                data_dict["keywords"] = []

        # Validate salary options
        valid_salary_options = ["Negotiable", "Prefer Not to Disclose", ""]
        if data.salaryOption and data.salaryOption not in valid_salary_options:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid salary option: {data.salaryOption}. Must be one of {valid_salary_options[:-1]} or empty."
            )

        # Generate description
        response = generate_description(data_dict)
        cleaned_response = clean_html_spacing(response)
        
        return {"description": cleaned_response}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")


@router.post("/generate-pass")
async def generate_pass_description_endpoint(data: dict):
    """Generate description for passed opportunity"""
    try:
        data_dict = data.copy()

        # Process skills if provided
        skills_required = data.get("skillsRequired")
        if skills_required and isinstance(skills_required, str) and skills_required.strip():
            try:
                data_dict["skillsRequired"] = [s.strip() for s in skills_required.split(",") if s and s.strip()]
            except Exception as e:
                data_dict["skillsRequired"] = []
        else:
            data_dict["skillsRequired"] = []
        response = generate_pass_opportunity_description(data_dict)
        cleaned_response = clean_html_spacing(response)
        
        return {"description": cleaned_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")


@router.post("/pass-opportunity")
async def pass_opportunity(data: DescriptionRequest):
    """Pass opportunity endpoint"""
    return {"message": "Opportunity passed successfully!"}
