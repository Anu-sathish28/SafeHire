import joblib
import re


# ============================================================
# SAFEHIRE TEST MODEL
# ============================================================
#
# PURPOSE:
# Test one job/internship at a time before connecting the
# finalized logic to Flask and the frontend.
#
# IMPORTANT:
# - User input can be written naturally.
# - User does NOT need to follow a fixed format.
# - SafeHire extracts useful information internally.
#
# ML representation:
#
#     Structured_Text + Natural_Text
#
# This MUST match train.py.
#
# ML Prediction and ML Confidence are shown here only for
# testing. They will be hidden from the final user interface.
#
# ============================================================


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "models/fake_job_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# 2. ENTER ONE JOB / INTERNSHIP
# ============================================================

new_job = """Software Developer Job

Company: TechNova Solutions
Location: Bengaluru

We are urgently hiring Software Developers.
Candidates can earn ₹12 LPA without any previous experience.

Your selection is guaranteed.
To confirm your selection, you must pay a registration fee of ₹2500
before the interview.

The recruiter will contact you through WhatsApp.
Send your Aadhaar card, PAN card, and other documents through WhatsApp.

No official company careers page is provided.
The recruiter says the payment must be completed today to secure the position.
"""

# ============================================================
# 3. BASIC TEXT PREPARATION
# ============================================================

text = new_job.lower()

text = re.sub(
    r"\s+",
    " ",
    text
).strip()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def contains_any(text, phrases):

    return any(
        phrase.lower() in text
        for phrase in phrases
    )


def has_any_email_domain(text, domains):

    return any(
        domain.lower() in text
        for domain in domains
    )


def first_match(patterns, text):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return match

    return None


def regex_match(patterns, text):

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
        for pattern in patterns
    )


# ============================================================
# 5. COMMON SAFEHIRE FIELDS
# ============================================================

safehire_fields = {

    "Job_Title": "N/A",
    "Company_Name": "N/A",
    "Domain": "N/A",
    "Job_Type": "N/A",
    "Job_Post": new_job,
    "Internship_Post": "N/A",
    "Internship_Type": "N/A",
    "Location": "N/A",
    "Salary": "N/A",
    "Stipend": "N/A",
    "Experience": "N/A",
    "Qualification": "N/A",
    "Website": "N/A",
    "Careers_Page": "N/A",
    "Recruitment_Platforms": "N/A",
    "Application_Method": "N/A",
    "Contact_Provided": "N/A",
    "Internship_Fee": "N/A",
    "Fee_Type": "N/A"
}


# ============================================================
# 6. EXTRACT JOB TITLE
# ============================================================

job_title_match = first_match(
    [
        r"job\s+title\s*:\s*(.+?)(?:\n|$)",
        r"position\s*:\s*(.+?)(?:\n|$)",
        r"role\s*:\s*(.+?)(?:\n|$)"
    ],
    new_job
)

if job_title_match:

    safehire_fields["Job_Title"] = (
        job_title_match.group(1).strip()
    )

else:

    lines = [
        line.strip()
        for line in new_job.splitlines()
        if line.strip()
    ]

    if lines:

        first_line = lines[0]

        if not re.match(
            r"^(company|company name|location|salary|stipend|"
            r"experience|qualification|website|careers|"
            r"email|contact|employer)\s*:",
            first_line,
            re.IGNORECASE
        ):

            safehire_fields["Job_Title"] = first_line


# ============================================================
# 7. EXTRACT COMPANY NAME
# ============================================================

company_match = first_match(
    [
        r"company\s*:\s*(.+?)(?:\n|$)",
        r"company\s+name\s*:\s*(.+?)(?:\n|$)",
        r"employer\s*:\s*(.+?)(?:\n|$)"
    ],
    new_job
)

if company_match:

    safehire_fields["Company_Name"] = (
        company_match.group(1).strip()
    )


# ============================================================
# 8. EXTRACT LOCATION
# ============================================================

location_match = first_match(
    [
        r"location\s*:\s*(.+?)(?:\n|$)",
        r"based\s+in\s+(.+?)(?:\n|$)",
        r"located\s+in\s+(.+?)(?:\n|$)"
    ],
    new_job
)

if location_match:

    safehire_fields["Location"] = (
        location_match.group(1).strip()
    )


# ============================================================
# 9. EXTRACT JOB TYPE
# ============================================================

if contains_any(
    text,
    [
        "full-time",
        "full time",
        "fulltime"
    ]
):

    safehire_fields["Job_Type"] = "Full-time"

elif contains_any(
    text,
    [
        "part-time",
        "part time",
        "parttime"
    ]
):

    safehire_fields["Job_Type"] = "Part-time"

elif contains_any(
    text,
    [
        "contract position",
        "contract job",
        "contract role",
        "contract work"
    ]
):

    safehire_fields["Job_Type"] = "Contract"


# ============================================================
# 10. EXTRACT INTERNSHIP INFORMATION
# ============================================================

is_internship = contains_any(
    text,
    [
        "internship",
        "intern",
        "internship opportunity",
        "internship program",
        "intern role"
    ]
)

if is_internship:

    safehire_fields["Internship_Post"] = new_job

    if contains_any(
        text,
        [
            "paid internship",
            "paid intern"
        ]
    ):

        safehire_fields["Internship_Type"] = "Paid"

    elif contains_any(
        text,
        [
            "unpaid internship",
            "unpaid intern"
        ]
    ):

        safehire_fields["Internship_Type"] = "Unpaid"


# ============================================================
# 11. EXTRACT DOMAIN
# ============================================================

if contains_any(
    text,
    [
        "software",
        "software engineering",
        "information technology",
        "information technology sector",
        "it company",
        "technology company",
        "tech company",
        "computer science",
        "web development",
        "app development",
        "data science",
        "machine learning",
        "artificial intelligence",
        "cybersecurity",
        "cloud computing"
    ]
):

    safehire_fields["Domain"] = (
        "Information Technology"
    )


# ============================================================
# 12. EXTRACT QUALIFICATION
# ============================================================

qualification_patterns = [

    r"degree\s+in\s+[^.\n]+",

    r"bachelor'?s\s+degree[^.\n]*",

    r"master'?s\s+degree[^.\n]*",

    r"computer\s+science\s+degree",

    r"engineering\s+degree",

    r"b\.?\s*e\.?\s+(?:degree)?",

    r"b\.?\s*tech\s+(?:degree)?",

    r"m\.?\s*e\.?\s+(?:degree)?",

    r"m\.?\s*tech\s+(?:degree)?",

    r"graduate[^.\n]*",

    r"undergraduate[^.\n]*"
]

qualification_match = first_match(
    qualification_patterns,
    new_job
)

if qualification_match:

    safehire_fields["Qualification"] = (
        qualification_match.group(0).strip()
    )


# ============================================================
# 13. EXTRACT EXPERIENCE
# ============================================================

experience_match = first_match(
    [

        r"\b\d+\s*(?:-|to)\s*\d+\s+years?\s+of\s+experience\b",

        r"\b\d+\+?\s+years?\s+of\s+experience\b",

        r"\bfreshers?\b",

        r"\bfresher\b",

        r"\bentry[- ]level\b",

        r"\bno\s+experience\s+required\b",

        r"\bwithout\s+experience\b"

    ],
    new_job
)

if experience_match:

    safehire_fields["Experience"] = (
        experience_match.group(0).strip()
    )


# ============================================================
# 14. EXTRACT SALARY
# ============================================================

salary_match = first_match(
    [

        r"(?:salary|ctc|package)\s*[:\-]?\s*₹?\s*[\d,.]+"
        r"(?:\s*[-to]+\s*₹?\s*[\d,.]+)?",

        r"₹\s*[\d,.]+\s*"
        r"(?:per\s+month|monthly|per\s+year|annually)",

        r"\b\d+\s*(?:lpa|lakhs?\s+per\s+annum)\b"

    ],
    new_job
)

if salary_match:

    safehire_fields["Salary"] = (
        salary_match.group(0).strip()
    )


# ============================================================
# 15. EXTRACT STIPEND
# ============================================================

stipend_match = first_match(
    [

        r"stipend\s*[:\-]?\s*₹?\s*[\d,.]+"
        r"(?:\s*[-to]+\s*₹?\s*[\d,.]+)?",

        r"₹\s*[\d,.]+\s*"
        r"(?:stipend|per\s+month\s+stipend)"

    ],
    new_job
)

if stipend_match:

    safehire_fields["Stipend"] = (
        stipend_match.group(0).strip()
    )


# ============================================================
# 16. EXTRACT WEBSITE / URL
# ============================================================

url_match = first_match(
    [
        r"https?://[^\s]+",
        r"www\.[^\s]+"
    ],
    new_job
)

if url_match:

    safehire_fields["Website"] = (
        url_match.group(0).strip()
    )


# ============================================================
# 17. OFFICIAL WEBSITE DETECTION
# ============================================================

official_website_patterns = [

    r"\bofficial\s+(?:company\s+)?website\b",

    r"\bofficial\s+(?:company\s+)?site\b",

    r"\bcompany'?s\s+official\s+website\b",

    r"\bofficial\s+website\s+of\s+the\s+company\b",

    r"\bofficial\s+company\s+site\b",

    r"\bcompany\s+website\b"

]

official_website_negative_patterns = [

    r"\bno\s+official\s+website\b",

    r"\bofficial\s+website\s+not\s+provided\b",

    r"\bwebsite\s+not\s+provided\b",

    r"\bcompany\s+website\s+is\s+not\s+provided\b",

    r"\bcompany\s+has\s+no\s+website\b"

]

official_website_detected = regex_match(
    official_website_patterns,
    text
)

official_website_negative = regex_match(
    official_website_negative_patterns,
    text
)

if official_website_detected and not official_website_negative:

    safehire_fields["Website"] = (
        "Official company website mentioned"
    )


# ============================================================
# 18. OFFICIAL CAREERS PAGE DETECTION
# ============================================================

official_careers_patterns = [

    r"\bofficial\s+(?:[a-z0-9&.'-]+\s+){0,5}"
    r"careers\s+(?:website|page|portal|site)\b",

    r"\bcompany'?s\s+official\s+careers\s+"
    r"(?:website|page|portal|site)\b",

    r"\bofficial\s+company\s+careers\s+"
    r"(?:website|page|portal|site)\b",

    r"\bofficial\s+careers\s+"
    r"(?:website|page|portal|site)\b"

]

official_careers_negative_patterns = [

    r"\bno\s+(?:official\s+)?careers\s+"
    r"(?:website|page|portal|site)\b",

    r"\bcareers\s+(?:website|page|portal|site)\s+"
    r"not\s+provided\b",

    r"\bofficial\s+careers\s+(?:website|page|portal|site)\s+"
    r"not\s+provided\b",

    r"\bcompany\s+careers\s+(?:website|page|portal|site)\s+"
    r"not\s+provided\b"

]

official_careers_detected = regex_match(
    official_careers_patterns,
    text
)

official_careers_negative = regex_match(
    official_careers_negative_patterns,
    text
)

if official_careers_detected and not official_careers_negative:

    safehire_fields["Careers_Page"] = (
        "Official careers page mentioned"
    )


# ============================================================
# 19. EXTRACT APPLICATION METHOD
# ============================================================

official_application_patterns = [

    r"\bapply\s+(?:through|via|on)\s+"
    r"(?:the\s+)?(?:[a-z0-9&.'-]+\s+){0,5}"
    r"official\s+(?:company\s+)?"
    r"(?:careers\s+)?(?:website|page|portal|site)\b",

    r"\bapplications?\s+(?:are\s+)?submitted\s+"
    r"(?:through|via|on)\s+(?:the\s+)?"
    r"(?:[a-z0-9&.'-]+\s+){0,5}"
    r"official\s+(?:company\s+)?"
    r"(?:careers\s+)?(?:website|page|portal|site)\b",

    r"\bapply\s+(?:through|via|on)\s+"
    r"the\s+company'?s\s+official\s+"
    r"(?:careers\s+)?(?:website|page|portal|site)\b"

]

if regex_match(
    official_application_patterns,
    text
):

    safehire_fields["Application_Method"] = (
        "Official company application"
    )

elif contains_any(
    text,
    [
        "apply through linkedin",
        "apply on linkedin",
        "linkedin application"
    ]
):

    safehire_fields["Application_Method"] = "LinkedIn"

elif contains_any(
    text,
    [
        "apply through naukri",
        "apply on naukri"
    ]
):

    safehire_fields["Application_Method"] = "Naukri"

elif contains_any(
    text,
    [
        "apply through internshala",
        "apply on internshala"
    ]
):

    safehire_fields["Application_Method"] = "Internshala"


# ============================================================
# 20. EXTRACT RECRUITMENT PLATFORM
# ============================================================

if contains_any(
    text,
    [
        "linkedin",
        "naukri",
        "indeed",
        "internshala",
        "foundit",
        "glassdoor"
    ]
):

    safehire_fields["Recruitment_Platforms"] = (
        "External recruitment platform mentioned"
    )


# ============================================================
# 21. EXTRACT CONTACT INFORMATION
# ============================================================

email_match = first_match(
    [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ],
    new_job
)

phone_match = first_match(
    [
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"
    ],
    new_job
)

if email_match:

    safehire_fields["Contact_Provided"] = (
        email_match.group(0)
    )

elif phone_match:

    safehire_fields["Contact_Provided"] = (
        phone_match.group(0)
    )

elif contains_any(
    text,
    [
        "whatsapp",
        "telegram",
        "recruiter",
        "hr contacted",
        "contact the recruiter"
    ]
):

    safehire_fields["Contact_Provided"] = (
        "Recruitment contact mentioned"
    )


# ============================================================
# 22. EXTRACT INTERNSHIP / TRAINING FEE
# ============================================================

internship_fee_match = first_match(
    [

        r"(?:internship|training|participation|program)"
        r"\s+(?:fee|charge|payment)\b",

        r"(?:fee|charge|payment)\s+(?:for|towards)"
        r"\s+(?:the\s+)?(?:internship|training|program)"

    ],
    new_job
)

if internship_fee_match:

    safehire_fields["Internship_Fee"] = (
        internship_fee_match.group(0).strip()
    )


# ============================================================
# 23. EXTRACT FEE TYPE
# ============================================================

no_fee_patterns = [

    r"\bno\s+(?:registration|application|processing|"
    r"verification|joining)\s+fee\b",

    r"\bno\s+fee\s+(?:is\s+)?required\b",

    r"\bno\s+payment\s+(?:is\s+)?required\b",

    r"\bwithout\s+(?:any\s+)?(?:registration|application|"
    r"processing|verification|joining)\s+fee\b",

    r"\bfree\s+(?:of\s+charge|application|registration)\b"

]

has_no_fee = any(
    re.search(
        pattern,
        text,
        re.IGNORECASE
    )
    for pattern in no_fee_patterns
)

if has_no_fee:

    safehire_fields["Fee_Type"] = "No fee"

elif contains_any(
    text,
    [
        "registration fee",
        "registration charge"
    ]
):

    safehire_fields["Fee_Type"] = "Registration fee"

elif contains_any(
    text,
    [
        "application fee",
        "application charge"
    ]
):

    safehire_fields["Fee_Type"] = "Application fee"

elif contains_any(
    text,
    [
        "processing fee",
        "processing charge"
    ]
):

    safehire_fields["Fee_Type"] = "Processing fee"

elif contains_any(
    text,
    [
        "verification fee",
        "verification charge"
    ]
):

    safehire_fields["Fee_Type"] = "Verification fee"

elif contains_any(
    text,
    [
        "joining fee",
        "joining charge"
    ]
):

    safehire_fields["Fee_Type"] = "Joining fee"

elif contains_any(
    text,
    [
        "security deposit",
        "refundable deposit"
    ]
):

    safehire_fields["Fee_Type"] = "Security deposit"


# ============================================================
# 24. CREATE STRUCTURED ML TEXT
# ============================================================

ml_field_order = [

    "Job_Title",
    "Company_Name",
    "Domain",
    "Job_Type",
    "Job_Post",
    "Internship_Post",
    "Internship_Type",
    "Location",
    "Salary",
    "Stipend",
    "Experience",
    "Qualification",
    "Website",
    "Careers_Page",
    "Recruitment_Platforms",
    "Application_Method",
    "Contact_Provided",
    "Internship_Fee",
    "Fee_Type"

]

structured_parts = []

for field in ml_field_order:

    value = str(
        safehire_fields[field]
    ).strip()

    if value:

        structured_parts.append(
            value
        )

structured_text = " ".join(
    structured_parts
)


# ============================================================
# 25. CREATE NATURAL-LANGUAGE REPRESENTATION
# ============================================================

natural_parts = []

job_title = str(
    safehire_fields["Job_Title"]
).strip()

company = str(
    safehire_fields["Company_Name"]
).strip()

location = str(
    safehire_fields["Location"]
).strip()


if job_title and job_title != "N/A":

    if company and company != "N/A":

        natural_parts.append(
            f"{company} is offering a "
            f"{job_title} opportunity."
        )

    else:

        natural_parts.append(
            f"There is a "
            f"{job_title} opportunity."
        )


domain = str(
    safehire_fields["Domain"]
).strip()

if domain and domain != "N/A":

    natural_parts.append(
        f"The opportunity is related to "
        f"{domain}."
    )


job_type = str(
    safehire_fields["Job_Type"]
).strip()

if job_type and job_type != "N/A":

    natural_parts.append(
        f"It is a "
        f"{job_type} position."
    )


if location and location != "N/A":

    natural_parts.append(
        f"The location is "
        f"{location}."
    )


job_post = str(
    safehire_fields["Job_Post"]
).strip()

if job_post and job_post != "N/A":

    natural_parts.append(
        job_post
    )


internship_post = str(
    safehire_fields["Internship_Post"]
).strip()

internship_type = str(
    safehire_fields["Internship_Type"]
).strip()


if internship_post and internship_post != "N/A":

    natural_parts.append(
        internship_post
    )


if internship_type and internship_type != "N/A":

    natural_parts.append(
        f"The internship type is "
        f"{internship_type}."
    )


if location and location != "N/A":

    pass


salary = str(
    safehire_fields["Salary"]
).strip()

if salary and salary != "N/A":

    natural_parts.append(
        f"The salary is "
        f"{salary}."
    )


stipend = str(
    safehire_fields["Stipend"]
).strip()

if stipend and stipend != "N/A":

    natural_parts.append(
        f"The stipend is "
        f"{stipend}."
    )


experience = str(
    safehire_fields["Experience"]
).strip()

if experience and experience != "N/A":

    natural_parts.append(
        f"The experience requirement is "
        f"{experience}."
    )


qualification = str(
    safehire_fields["Qualification"]
).strip()

if qualification and qualification != "N/A":

    natural_parts.append(
        f"The qualification requirement is "
        f"{qualification}."
    )


website = str(
    safehire_fields["Website"]
).strip()

if website and website != "N/A":

    natural_parts.append(
        f"Website information: "
        f"{website}."
    )


careers = str(
    safehire_fields["Careers_Page"]
).strip()

if careers and careers != "N/A":

    natural_parts.append(
        f"Careers page information: "
        f"{careers}."
    )


platforms = str(
    safehire_fields["Recruitment_Platforms"]
).strip()

if platforms and platforms != "N/A":

    natural_parts.append(
        f"Recruitment platform: "
        f"{platforms}."
    )


application_method = str(
    safehire_fields["Application_Method"]
).strip()

if application_method and application_method != "N/A":

    natural_parts.append(
        f"Application method: "
        f"{application_method}."
    )


contact = str(
    safehire_fields["Contact_Provided"]
).strip()

if contact and contact != "N/A":

    natural_parts.append(
        f"Contact information: "
        f"{contact}."
    )


internship_fee = str(
    safehire_fields["Internship_Fee"]
).strip()

if internship_fee and internship_fee != "N/A":

    natural_parts.append(
        f"Internship fee information: "
        f"{internship_fee}."
    )


fee_type = str(
    safehire_fields["Fee_Type"]
).strip()

if fee_type and fee_type != "N/A":

    natural_parts.append(
        f"Fee type: "
        f"{fee_type}."
    )


natural_text = " ".join(
    natural_parts
)


# ============================================================
# 26. COMBINE REPRESENTATIONS
# ============================================================

ml_text = (
    structured_text
    + " "
    + natural_text
)

ml_text = re.sub(
    r"\s+",
    " ",
    ml_text
).strip()


# ============================================================
# 27. DISPLAY INTERNAL REPRESENTATION
# ============================================================

print("\n")
print("=" * 70)
print("              INTERNAL SAFEHIRE REPRESENTATION")
print("=" * 70)

for field in ml_field_order:

    print(
        f"{field:22}: "
        f"{safehire_fields[field]}"
    )

print("\nStructured Text:")
print(
    structured_text
)

print("\nNatural Text:")
print(
    natural_text
)

print("\nFinal ML Text:")
print(
    ml_text
)

print("=" * 70)


# ============================================================
# 28. ML PREDICTION
# ============================================================

job_vector = vectorizer.transform(
    [ml_text]
)

prediction = model.predict(
    job_vector
)[0]

probability = model.predict_proba(
    job_vector
)[0]

confidence = max(
    probability
) * 100


# ============================================================
# 29. INDICATOR LISTS
# ============================================================

red_flags = []

positive_indicators = []

caution_indicators = []


# ============================================================
# RED FLAG 1 - PERSONAL / UNOFFICIAL EMAIL
# ============================================================

personal_email_domains = [

    "@gmail.com",
    "@yahoo.com",
    "@outlook.com",
    "@hotmail.com",
    "@rediffmail.com"

]

if (
    has_any_email_domain(
        text,
        personal_email_domains
    )
    and not contains_any(
        text,
        [
            "no personal email",
            "official company email",
            "company email address"
        ]
    )
):

    red_flags.append(
        "A personal or unofficial email address is used for recruitment"
    )


# ============================================================
# RED FLAG 2 - UNSOLICITED CONTACT
# ============================================================

if contains_any(
    text,
    [

        "unsolicited email",
        "unsolicited contact",
        "unsolicited message",
        "unsolicited job offer",
        "unexpected job offer",
        "unexpected recruitment message",
        "random job offer",
        "random recruitment message"

    ]
):

    red_flags.append(
        "The opportunity was received through unsolicited contact"
    )


# ============================================================
# RED FLAG 3 - NO OFFICIAL WEBSITE
# ============================================================

if regex_match(
    [

        r"\bwebsite\s+not\s+provided\b",
        r"\bno\s+official\s+website\b",
        r"\bofficial\s+website\s+not\s+provided\b",
        r"\bofficial\s+company\s+website\s+not\s+provided\b",
        r"\bcompany\s+website\s+is\s+not\s+provided\b",
        r"\bcompany\s+has\s+no\s+website\b"

    ],
    text
):

    red_flags.append(
        "No official company website was provided"
    )


# ============================================================
# RED FLAG 4 - NO OFFICIAL CAREERS PAGE
# ============================================================

if regex_match(
    [

        r"\bcareers\s+page\s+not\s+provided\b",
        r"\bcareers\s+website\s+not\s+provided\b",
        r"\bno\s+careers\s+page\b",
        r"\bno\s+official\s+careers\s+page\b",
        r"\bofficial\s+careers\s+page\s+not\s+provided\b",
        r"\bcompany\s+careers\s+page\s+not\s+provided\b"

    ],
    text
):

    red_flags.append(
        "No official careers page was provided"
    )


# ============================================================
# RED FLAG 5 - FEES
# ============================================================

fee_patterns = [

    r"\bregistration\s+(?:and\s+)?"
    r"(?:training\s+)?fee\b",

    r"\bapplication\s+fee\b",

    r"\bprocessing\s+fee\b",

    r"\bverification\s+fee\b",

    r"\bdocument\s+verification\s+(?:fee|charge)\b",

    r"\bregistration\s+charge\b",

    r"\bapplication\s+charge\b",

    r"\bprocessing\s+charge\b",

    r"\bverification\s+charge\b",

    r"\bjoining\s+fee\b",

    r"\bselection\s+fee\b",

    r"\brecruitment\s+fee\b",

    r"\bsecurity\s+deposit\b",

    r"\b(?:pay|payment)\s+.*\bregistration\b",

    r"\b(?:pay|payment)\s+.*\bverification\b",

    r"\b(?:pay|payment)\s+.*\bprocessing\b"

]

no_fee_phrases = [

    "no registration fee",
    "no application fee",
    "no processing fee",
    "no verification fee",
    "no joining fee",
    "no recruitment fee",
    "no fee is required",
    "no fee required",
    "without any fee",
    "without a fee",
    "no payment required",
    "no payment is required"

]

fee_detected = any(
    re.search(
        pattern,
        text,
        re.IGNORECASE
    )
    for pattern in fee_patterns
)

no_fee = contains_any(
    text,
    no_fee_phrases
)

if fee_detected and not no_fee:

    red_flags.append(
        "A registration, application, processing, or verification fee is required"
    )


# ============================================================
# RED FLAG 6 - PAYMENT BEFORE INTERVIEW / SELECTION / JOINING
# ============================================================

payment_before_phrases = [

    "payment before joining",
    "pay before joining",
    "payment required before joining",
    "pay money before joining",

    "payment before interview",
    "pay before interview",
    "payment required before interview",
    "pay money before interview",

    "payment before selection",
    "pay before selection",
    "payment required before selection",
    "pay money before selection",

    "payment before onboarding",
    "pay before onboarding",
    "payment required before onboarding",

    "pay to confirm selection",
    "payment to confirm selection",

    "selection confirmed after payment",
    "joining confirmed after payment",
    "selected after payment",

    "selection will be confirmed only after payment",

    "pay to process application",
    "payment to process application",

    "pay to complete application",
    "payment required to process application",

    "payment must be made before the internship",

    "payment must be made before the internship begins",

    "payment required before the internship",

    "pay before the internship begins",

    "payment before the internship begins",

    "payment before starting the internship",

    "payment before starting the job",

    "pay before starting the internship",

    "pay before starting the job",

    "payment before starting work",

    "pay before starting work"

]

no_payment_phrases = [

    "no payment required before joining",
    "no payment is required before joining",

    "no payment required before interview",
    "no payment is required before interview",

    "no payment required before selection",
    "no payment is required before selection",

    "no payment required before onboarding",
    "no payment is required before onboarding",

    "no payment required",
    "no payment is required",

    "no fee is required"

]

if (
    contains_any(
        text,
        payment_before_phrases
    )
    and not contains_any(
        text,
        no_payment_phrases
    )
):

    red_flags.append(
        "Payment is requested before interview, selection, joining, or onboarding"
    )


# ============================================================
# RED FLAG 7 - WHATSAPP / TELEGRAM
# ============================================================

messaging_patterns = [

    r"\bcontact\s+.*\bthrough\s+whatsapp\b",
    r"\bcontact\s+.*\bon\s+whatsapp\b",
    r"\breach\s+out\s+.*\bon\s+whatsapp\b",
    r"\bcontact\s+.*\bvia\s+whatsapp\b",
    r"\bthrough\s+whatsapp\b",
    r"\bvia\s+whatsapp\b",
    r"\bon\s+whatsapp\b",
    r"\bwhatsapp\s+for\s+further\s+processing\b",
    r"\bsend\s+.*\bdocuments\b.*\bwhatsapp\b",
    r"\bsubmit\s+.*\bdocuments\b.*\bwhatsapp\b",

    r"\bcontact\s+.*\bthrough\s+telegram\b",
    r"\bcontact\s+.*\bon\s+telegram\b",
    r"\bthrough\s+telegram\b",
    r"\bvia\s+telegram\b",
    r"\bsend\s+.*\bdocuments\b.*\btelegram\b",
    r"\bsubmit\s+.*\bdocuments\b.*\btelegram\b",

    r"\bpersonal\s+messaging\s+platform\b",
    r"\bpersonal\s+messaging\s+app\b",
    r"\bpersonal\s+phone\s+number\b.*\brecruitment\b"

]

no_messaging_phrases = [

    "no whatsapp",
    "not through whatsapp",
    "not via whatsapp",
    "official whatsapp",
    "no telegram",
    "not through telegram",
    "not via telegram"

]

messaging_detected = any(
    re.search(
        pattern,
        text,
        re.IGNORECASE
    )
    for pattern in messaging_patterns
)

no_messaging = contains_any(
    text,
    no_messaging_phrases
)

if messaging_detected and not no_messaging:

    red_flags.append(
        "Recruitment or document submission is being conducted through a personal messaging platform"
    )


# ============================================================
# RED FLAG 8 - IMPERSONATION
# ============================================================

impersonation_patterns = [

    "impersonated",
    "impersonation",
    "pretending to be",
    "fake company identity",
    "posing as",
    "pretending to represent",
    "claims to represent",
    "claiming to represent",
    "someone claiming to represent",
    "person claiming to represent",
    "pretends to represent"

]

if (
    contains_any(
        text,
        impersonation_patterns
    )
    and not contains_any(
        text,
        [
            "not impersonated",
            "no impersonation"
        ]
    )
):

    red_flags.append(
        "The opportunity may be impersonating a legitimate organization"
    )


# ============================================================
# CAUTION 1 - INTERNSHIP / TRAINING FEE
# ============================================================

internship_fee_phrases = [

    "internship fee",
    "training fee",
    "internship participation fee",
    "participation fee",
    "program fee",
    "internship charge",
    "training charge",
    "internship payment",
    "program participation fee"

]

no_internship_fee_phrases = [

    "no internship fee",
    "no training fee",
    "no participation fee",
    "no program fee",
    "no internship charge"

]

if (
    contains_any(
        text,
        internship_fee_phrases
    )
    and not contains_any(
        text,
        no_internship_fee_phrases
    )
):

    caution_indicators.append(
        "An internship, training, or program participation fee is mentioned; verify the program and payment terms carefully"
    )


# ============================================================
# CAUTION 2 - EXTERNAL RECRUITMENT AGENCY
# ============================================================

if contains_any(
    text,
    [

        "external recruitment agency",
        "external recruiting agency",
        "third-party recruitment agency",
        "third party recruitment agency",
        "recruitment agency"

    ]
):

    caution_indicators.append(
        "Recruitment is handled by an external agency and should be verified"
    )


# ============================================================
# CAUTION 3 - EXTERNAL JOB PLATFORM
# ============================================================

if contains_any(
    text,
    [

        "external job portal",
        "external recruitment portal",
        "external recruitment platform",
        "external job platform",
        "third-party job portal",
        "third party job portal",
        "third-party recruitment platform",
        "third party recruitment platform",
        "external application platform"

    ]
):

    caution_indicators.append(
        "Application is submitted through an external recruitment platform"
    )


# ============================================================
# CAUTION 4 - LIMITED INFORMATION
# ============================================================

if contains_any(
    text,
    [

        "company information is limited",
        "limited company information",
        "limited information about the company",
        "limited information available about the position",
        "very little information about the company"

    ]
):

    caution_indicators.append(
        "Company or position information is limited and should be verified"
    )


# ============================================================
# CAUTION 5 - DIRECT RECRUITER CONTACT
# ============================================================

if contains_any(
    text,
    [

        "recruiter contacted the candidate directly",
        "recruiter contacted the candidate",
        "recruiter contacted me directly",
        "hr contacted the candidate directly",
        "hr contacted me directly",
        "recruiter reached out directly"

    ]
):

    caution_indicators.append(
        "The recruiter contacted the candidate directly and should be verified"
    )


# ============================================================
# CAUTION 6 - UNCLEAR SALARY
# ============================================================

if contains_any(
    text,
    [

        "salary details are not clearly specified",
        "salary not clearly specified",
        "salary is not clearly specified",
        "salary details are unclear",
        "salary is unclear",
        "salary information is not provided"

    ]
):

    caution_indicators.append(
        "Salary details are not clearly specified"
    )


# ============================================================
# CAUTION 7 - URGENCY
# ============================================================

if contains_any(
    text,
    [

        "urgent hiring",
        "urgent registration",
        "urgent joining",
        "immediate joining",
        "immediate joining required",
        "join immediately",
        "apply immediately",
        "apply quickly",
        "register today",
        "act now",
        "limited openings",
        "limited seats",
        "limited vacancies",
        "only a few openings",
        "only a few seats"

    ]
):

    caution_indicators.append(
        "Urgency is used and should be verified before proceeding"
    )


# ============================================================
# CAUTION 8 - UNREALISTIC / GUARANTEED CLAIMS
# ============================================================

if contains_any(
    text,
    [

        "guaranteed job",
        "guaranteed employment",
        "guaranteed placement",
        "guaranteed salary",
        "guaranteed selection",
        "guaranteed internship",
        "guaranteed internship certificate",
        "no interview",
        "no technical interview",
        "high salary with no experience",
        "earn a high salary",
        "unusually high salary",
        "unrealistic salary",
        "easy job with high salary"

    ]
):

    caution_indicators.append(
        "The opportunity contains an unusually guaranteed or unrealistic employment claim"
    )


# ============================================================
# POSITIVE 1 - OFFICIAL COMPANY WEBSITE
# ============================================================

positive_official_website_patterns = [

    r"\bofficial\s+(?:[a-z0-9&.'-]+\s+){0,5}website\b",

    r"\bofficial\s+(?:[a-z0-9&.'-]+\s+){0,5}site\b",

    r"\b(?:company|employer)'?s\s+official\s+"
    r"(?:[a-z0-9&.'-]+\s+){0,5}website\b",

    r"\bofficial\s+company\s+website\b",

    r"\bcompany\s+website\b"

]

positive_official_website_detected = any(
    re.search(
        pattern,
        text,
        re.IGNORECASE
    )
    for pattern in positive_official_website_patterns
)

positive_official_website_negative = contains_any(
    text,
    [
        "official website not provided",
        "no official website",
        "website not provided",
        "company website is not provided",
        "company has no website"
    ]
)

if (
    positive_official_website_detected
    and not positive_official_website_negative
):

    positive_indicators.append(
        "An official company website is mentioned"
    )


# ============================================================
# POSITIVE 2 - OFFICIAL CAREERS PAGE
# ============================================================

positive_official_careers_patterns = [

    r"\bofficial\s+(?:[a-z0-9&.'-]+\s+){0,5}"
    r"careers\s+(?:website|page|portal|site)\b",

    r"\b(?:[a-z0-9&.'-]+\s+){1,5}"
    r"official\s+careers\s+(?:website|page|portal|site)\b",

    r"\bcompany'?s\s+official\s+careers\s+"
    r"(?:website|page|portal|site)\b",

    r"\bofficial\s+company\s+careers\s+"
    r"(?:website|page|portal|site)\b",

    r"\bcompany\s+careers\s+"
    r"(?:website|page|portal|site)\b"

]

positive_official_careers_detected = any(
    re.search(
        pattern,
        text,
        re.IGNORECASE
    )
    for pattern in positive_official_careers_patterns
)

positive_official_careers_negative = contains_any(
    text,
    [
        "careers page not provided",
        "careers page is not provided",
        "no careers page",

        "official careers page not provided",
        "official careers page is not provided",
        "no official careers page",

        "company careers page not provided",
        "company careers page is not provided",

        "no official company careers page",
        "no official company careers page is provided",

        "careers website not provided",
        "careers website is not provided",

        "no official careers website",
        "no official company careers website"
    ]
)

if (
    positive_official_careers_detected
    and not positive_official_careers_negative
):

    positive_indicators.append(
        "An official careers page is mentioned"
    )


# ============================================================
# POSITIVE 3 - OFFICIAL APPLICATION CHANNEL
# ============================================================

official_application_positive_patterns = [

    r"\bapply\s+(?:through|via|on|at)\s+"
    r"(?:the\s+)?official\s+"
    r"(?:[a-z0-9&.'-]+\s+){0,5}"
    r"(?:careers\s+)?(?:website|page|portal|site)\b",

    r"\bapply\s+(?:through|via|on|at)\s+"
    r"(?:the\s+)?"
    r"(?:[a-z0-9&.'-]+\s+){1,5}"
    r"official\s+"
    r"(?:careers\s+)?(?:website|page|portal|site)\b",

    r"\bsubmit\s+(?:your\s+)?application\s+"
    r"(?:through|via|on|at)\s+"
    r"(?:the\s+)?official\s+"
    r"(?:[a-z0-9&.'-]+\s+){0,5}"
    r"(?:careers\s+)?(?:website|page|portal|site)\b",

    r"\bapplications?\s+(?:are\s+)?submitted\s+"
    r"(?:through|via|on|at)\s+"
    r"(?:the\s+)?official\s+"
    r"(?:[a-z0-9&.'-]+\s+){0,5}"
    r"(?:careers\s+)?(?:website|page|portal|site)\b",

    r"\buse\s+(?:the\s+)?official\s+"
    r"(?:[a-z0-9&.'-]+\s+){0,5}"
    r"(?:careers\s+)?(?:website|page|portal|site)\s+"
    r"to\s+apply\b"

]

official_application_detected = any(
    re.search(
        pattern,
        text,
        re.IGNORECASE
    )
    for pattern in official_application_positive_patterns
)

if official_application_detected:

    positive_indicators.append(
        "Application is directed through an official company channel"
    )


# ============================================================
# POSITIVE 4 - NO REGISTRATION FEE
# ============================================================

if contains_any(
    text,
    [
        "no registration fee",
        "no registration fee is required",
        "there is no registration fee"
    ]
):

    positive_indicators.append(
        "No registration fee is mentioned"
    )


# ============================================================
# POSITIVE 5 - NO APPLICATION FEE
# ============================================================

if contains_any(
    text,
    [
        "no application fee",
        "no application fee is required",
        "there is no application fee"
    ]
):

    positive_indicators.append(
        "No application fee is mentioned"
    )


# ============================================================
# POSITIVE 6 - NO PAYMENT
# ============================================================

if contains_any(
    text,
    [
        "no payment required before joining",
        "no payment is required before joining",
        "no payment required",
        "no payment is required",
        "no payment before joining",
        "no payment before selection",
        "no payment before interview",
        "no payment before onboarding"
    ]
):

    positive_indicators.append(
        "No payment is required during the stated recruitment process"
    )


# ============================================================
# 30. FINAL SAFEHIRE DECISION
# ============================================================

if len(red_flags) >= 2:

    final_result = "Fake"

elif (
    len(red_flags) == 1
    and prediction == "Fake"
):

    final_result = "Fake"

elif len(red_flags) >= 1:

    final_result = "Fake"

elif len(caution_indicators) >= 2:

    final_result = "Suspicious"

elif prediction == "Fake":

    # The ML model indicates potential risk, but
    # there are not enough rule-based indicators
    # to classify the opportunity as Fake.
    final_result = "Suspicious"

    caution_indicators.append(
        "ML analysis indicates potential risk, but no strong fraud indicator was identified; further verification is recommended"
    )

elif (
    prediction == "Genuine"
    and len(red_flags) == 0
):

    final_result = "Genuine"

else:

    final_result = "Suspicious"

    if not caution_indicators:

        caution_indicators.append(
            "The available information is insufficient to confidently verify the opportunity; further verification is recommended"
        )

# ============================================================
# 31. SAFEHIRE CONFIDENCE
# ============================================================
#
# This is DIFFERENT from ML Confidence.
#
# SafeHire Confidence combines:
# - ML confidence
# - Positive evidence
# - Red flags
# - Caution indicators
# - Final SafeHire result
#
# Suspicious intentionally stays moderate.
#
# ============================================================

def calculate_safehire_confidence(
    ml_confidence,
    final_result,
    positive_indicators,
    red_flags,
    caution_indicators
):

    positive_count = len(
        positive_indicators
    )

    red_count = len(
        red_flags
    )

    caution_count = len(
        caution_indicators
    )

    # --------------------------------------------------------
    # GENUINE
    # --------------------------------------------------------

    if final_result == "Genuine":

        score = (
            ml_confidence * 0.60
        )

        score += min(
            positive_count * 12,
            48
        )

        score -= min(
            caution_count * 4,
            12
        )

        score -= min(
            red_count * 15,
            45
        )

    # --------------------------------------------------------
    # FAKE
    # --------------------------------------------------------

    elif final_result == "Fake":

        score = (
            ml_confidence * 0.60
        )

        score += min(
            red_count * 20,
            40
        )

        score -= min(
            positive_count * 3,
            9
        )

        score -= min(
            caution_count * 2,
            6
        )

    # --------------------------------------------------------
    # SUSPICIOUS
    # --------------------------------------------------------

    else:

        score = 55

        score += min(
            red_count * 4,
            4
        )

        score += min(
            caution_count * 4,
            4
        )

        score += min(
            positive_count * 2,
            2
        )

        score = min(
            score,
            60
        )

    score = max(
        0,
        min(
            score,
            100
        )
    )

    return round(
        score,
        2
    )


safehire_confidence = calculate_safehire_confidence(
    confidence,
    final_result,
    positive_indicators,
    red_flags,
    caution_indicators
)


# ============================================================
# 32. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 70)
print("                         SAFEHIRE RESULT")
print("=" * 70)


# ------------------------------------------------------------
# TESTING ONLY
# ------------------------------------------------------------

print("\nML Prediction :")

print(
    prediction
)

print(
    "ML Confidence :",
    round(
        confidence,
        2
    ),
    "%"
)


# ------------------------------------------------------------
# FINAL USER-STYLE RESULT
# ------------------------------------------------------------

print(
    "SafeHire Result:",
    final_result
)

print(
    "SafeHire Confidence:",
    safehire_confidence,
    "%"
)


# ============================================================
# POSITIVE INDICATORS
# ============================================================

print("\nPositive Indicators:")

if positive_indicators:

    for item in positive_indicators:

        print(
            "✓",
            item
        )

else:

    print(
        "- None identified"
    )


# ============================================================
# RED FLAGS
# ============================================================

print("\nRed Flags:")

if red_flags:

    for item in red_flags:

        print(
            "✗",
            item
        )

else:

    print(
        "- None identified"
    )


# ============================================================
# CAUTION INDICATORS
# ============================================================

print("\nCaution Indicators:")

if caution_indicators:

    for item in caution_indicators:

        print(
            "⚠",
            item
        )

else:

    print(
        "- None identified"
    )


# ============================================================
# END
# ============================================================

print(
    "\n" + "=" * 70
)