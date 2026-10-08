from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import re
import io
import zipfile
import requests
import joblib

from bs4 import BeautifulSoup


# ============================================================
# OPTIONAL / FILE PROCESSING IMPORTS
# ============================================================

try:
    from PIL import (
        Image,
        ImageOps,
        ImageEnhance,
        ImageFilter
    )
except ImportError:
    Image = None
    ImageOps = None
    ImageEnhance = None
    ImageFilter = None


try:
    import pytesseract
except ImportError:
    pytesseract = None


try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


try:
    from docx import Document
except ImportError:
    Document = None


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "fake_job_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD ML MODEL AND TF-IDF
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    print(
        "ML model and TF-IDF vectorizer loaded successfully."
    )

except Exception as e:

    print(
        "ERROR loading ML model or TF-IDF vectorizer:"
    )

    print(e)

    model = None
    vectorizer = None


# ============================================================
# ML FIELDS
# MUST MATCH TRAINING PIPELINE
# ============================================================

ML_FIELDS = [

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


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    if isinstance(value, float):

        if value != value:
            return ""

    return str(value).strip()


def normalize_for_search(value):

    return re.sub(
        r"\s+",
        " ",
        normalize_text(value)
    ).strip().lower()


def clean_text(value):

    value = normalize_text(
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def contains_any(
    text,
    patterns
):

    text = normalize_for_search(
        text
    )

    for pattern in patterns:

        if pattern in text:
            return True

    return False


def first_match(
    text,
    patterns
):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    return ""


# ============================================================
# SENTENCE / NEGATION HELPERS
# ============================================================

def get_local_context(
    text,
    start,
    end,
    window=120
):

    start_index = max(
        0,
        start - window
    )

    end_index = min(
        len(text),
        end + window
    )

    return text[
        start_index:end_index
    ].lower()


def is_negated_fee_context(
    text,
    start,
    end
):

    context = get_local_context(
        text,
        start,
        end,
        150
    )

    negation_patterns = [

        r"\bno\s+(?:registration|application|processing|verification|recruitment|joining|onboarding)\s+fee\b",

        r"\bno\s+(?:registration|application|processing|verification|recruitment|joining|onboarding)\s+fee\s+(?:is\s+)?required\b",

        r"\b(?:registration|application|processing|verification|recruitment|joining|onboarding)\s+fee\s+(?:is\s+)?not\s+required\b",

        r"\bwithout\s+(?:any\s+)?(?:registration|application|processing|verification|recruitment|joining|onboarding)\s+fee\b",

        r"\bfree\s+of\s+(?:registration|application|processing|verification|recruitment|joining|onboarding)\s+fee\b",

        r"\bno\s+fees?\s+(?:are\s+)?required\b",

        r"\bapplicants?\s+(?:are\s+)?not\s+required\s+to\s+pay\b",

        r"\bno\s+payment\s+(?:is\s+)?required\b",

        r"\bpayment\s+(?:is\s+)?not\s+required\b"

    ]

    for pattern in negation_patterns:

        if re.search(
            pattern,
            context,
            re.IGNORECASE
        ):
            return True

    return False


def is_negated_payment_context(
    text,
    start,
    end
):

    context = get_local_context(
        text,
        start,
        end,
        180
    )

    negation_patterns = [

        r"\bno\s+payment\s+(?:is\s+)?required\b",

        r"\bpayment\s+(?:is\s+)?not\s+required\b",

        r"\bno\s+payment\s+is\s+needed\b",

        r"\bpayment\s+is\s+not\s+needed\b",

        r"\bno\s+fee\s+is\s+required\b",

        r"\bno\s+fees?\s+(?:are\s+)?required\b",

        r"\bapplicants?\s+(?:are\s+)?not\s+required\s+to\s+pay\b",

        r"\bwithout\s+(?:any\s+)?payment\b"

    ]

    for pattern in negation_patterns:

        if re.search(
            pattern,
            context,
            re.IGNORECASE
        ):
            return True

    return False


# ============================================================
# URL / EMAIL / PHONE HELPERS
# ============================================================

def extract_urls(text):

    return re.findall(
        r"https?://[^\s<>\"]+",
        text,
        re.IGNORECASE
    )


def clean_url(url):

    return url.rstrip(
        ".,;:!?)]}"
    )


def extract_emails(text):

    return re.findall(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        text,
        re.IGNORECASE
    )


def extract_phone_numbers(text):

    return re.findall(
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        text
    )


def is_personal_email(email):

    personal_domains = {

        "gmail.com",
        "yahoo.com",
        "yahoo.in",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "rediffmail.com",
        "icloud.com",
        "protonmail.com",
        "proton.me"

    }

    email = email.lower().strip()

    if "@" not in email:
        return False

    domain = email.split("@")[-1]

    return domain in personal_domains


# ============================================================
# DOMAIN / COMPANY HELPERS
# ============================================================

def get_domain_from_url(url):

    try:

        match = re.search(
            r"https?://(?:www\.)?([^/]+)",
            url,
            re.IGNORECASE
        )

        if match:

            return match.group(1).lower()

    except Exception:
        pass

    return ""


def detect_domain_from_urls(urls):

    if not urls:
        return ""

    for url in urls:

        domain = get_domain_from_url(
            url
        )

        if domain:
            return domain

    return ""


def company_name_from_domain(domain):

    if not domain:
        return ""

    domain = domain.lower()

    known_companies = {

        "microsoft.com": "Microsoft",
        "google.com": "Google",
        "amazon.com": "Amazon",
        "amazon.jobs": "Amazon",
        "apple.com": "Apple",
        "ibm.com": "IBM",
        "infosys.com": "Infosys",
        "tcs.com": "TCS",
        "wipro.com": "Wipro",
        "accenture.com": "Accenture",
        "deloitte.com": "Deloitte",
        "capgemini.com": "Capgemini",
        "cognizant.com": "Cognizant",
        "oracle.com": "Oracle",
        "meta.com": "Meta",
        "intel.com": "Intel",
        "nvidia.com": "NVIDIA",
        "salesforce.com": "Salesforce"

    }

    if domain in known_companies:

        return known_companies[
            domain
        ]

    base = domain.split(".")[0]

    if base in {

        "careers",
        "jobs",
        "job",
        "work",
        "career"

    }:

        return ""

    base = base.replace(
        "-",
        " "
    ).replace(
        "_",
        " "
    )

    if base:

        return base.title()

    return ""


def company_name_from_email(
    email
):

    email = normalize_text(
        email
    ).lower()

    if "@" not in email:
        return ""

    domain = email.split(
        "@",
        1
    )[1]

    return company_name_from_domain(
        domain
    )


# ============================================================
# URL EXTRACTION
# ============================================================

def fetch_url_text(url):

    try:

        url = clean_url(
            url
        )

        headers = {

            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),

            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),

            "Accept-Language":
                "en-US,en;q=0.9"

        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe"
            ]
        ):

            tag.decompose()

        title = ""

        if soup.title:

            title = soup.title.get_text(
                " ",
                strip=True
            )

        visible_text = soup.get_text(
            separator=" ",
            strip=True
        )

        combined = (
            title
            + "\n"
            + visible_text
        ).strip()

        return combined

    except Exception as e:

        print(
            "URL extraction error:",
            e
        )

        return ""


# ============================================================
# IMAGE / OCR EXTRACTION
# ============================================================

def preprocess_image(
    image
):

    try:

        if ImageOps is not None:

            try:

                image = ImageOps.exif_transpose(
                    image
                )

            except Exception:
                pass

        image = image.convert(
            "RGB"
        )

        image = ImageOps.grayscale(
            image
        )

        width, height = image.size

        target_width = 1800

        if width < target_width:

            scale = (
                target_width /
                float(width)
            )

            image = image.resize(
                (
                    int(width * scale),
                    int(height * scale)
                ),
                Image.Resampling.LANCZOS
            )

        image = ImageEnhance.Contrast(
            image
        ).enhance(2.0)

        if ImageFilter is not None:

            image = image.filter(
                ImageFilter.SHARPEN
            )

        return image

    except Exception:

        return image


def extract_image_text(
    file_bytes
):

    if (
        Image is None
        or pytesseract is None
    ):

        return ""

    try:

        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )

        image = preprocess_image(
            image
        )

        ocr_results = []

        for psm in [
            6,
            11,
            12
        ]:

            try:

                text = pytesseract.image_to_string(
                    image,
                    config=f"--psm {psm}"
                )

                text = text.strip()

                if text:

                    ocr_results.append(
                        text
                    )

            except Exception as e:

                print(
                    "OCR pass error:",
                    e
                )

        if not ocr_results:

            return ""

        ocr_results.sort(
            key=len,
            reverse=True
        )

        return ocr_results[0]

    except Exception as e:

        print(
            "OCR error:",
            e
        )

        return ""


def ocr_pdf_page(
    page
):

    if (
        fitz is None
        or Image is None
        or pytesseract is None
    ):

        return ""

    try:

        matrix = fitz.Matrix(
            2.0,
            2.0
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image = Image.open(
            io.BytesIO(
                pix.tobytes(
                    "png"
                )
            )
        )

        image = preprocess_image(
            image
        )

        results = []

        for psm in [
            6,
            11
        ]:

            try:

                text = pytesseract.image_to_string(
                    image,
                    config=f"--psm {psm}"
                )

                if text.strip():

                    results.append(
                        text.strip()
                    )

            except Exception:
                pass

        if not results:

            return ""

        results.sort(
            key=len,
            reverse=True
        )

        return results[0]

    except Exception as e:

        print(
            "PDF page OCR error:",
            e
        )

        return ""


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(
    file_bytes
):

    if fitz is None:

        return ""

    extracted_pages = []

    try:

        pdf = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        for page_number, page in enumerate(
            pdf
        ):

            page_text = page.get_text(
                "text"
            ).strip()

            if len(page_text) < 80:

                ocr_text = ocr_pdf_page(
                    page
                )

                if ocr_text:

                    if page_text:

                        combined = (
                            page_text
                            + "\n"
                            + ocr_text
                        )

                        extracted_pages.append(
                            combined
                        )

                    else:

                        extracted_pages.append(
                            ocr_text
                        )

                elif page_text:

                    extracted_pages.append(
                        page_text
                    )

            else:

                extracted_pages.append(
                    page_text
                )

        pdf.close()

        return "\n".join(
            extracted_pages
        ).strip()

    except Exception as e:

        print(
            "PDF extraction error:",
            e
        )

        return ""


# ============================================================
# DOCX EMBEDDED IMAGE OCR
# ============================================================

def extract_docx_embedded_images(
    file_bytes
):

    if (
        Image is None
        or pytesseract is None
    ):

        return ""

    results = []

    try:

        with zipfile.ZipFile(
            io.BytesIO(
                file_bytes
            )
        ) as archive:

            media_files = [

                name

                for name in archive.namelist()

                if name.startswith(
                    "word/media/"
                )

            ]

            for media_name in media_files:

                try:

                    image_bytes = archive.read(
                        media_name
                    )

                    image_text = extract_image_text(
                        image_bytes
                    )

                    if image_text:

                        results.append(
                            image_text
                        )

                except Exception as e:

                    print(
                        "DOCX embedded image OCR error:",
                        e
                    )

    except Exception as e:

        print(
            "DOCX embedded image extraction error:",
            e
        )

    return "\n".join(
        results
    ).strip()


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_text(
    file_bytes
):

    if Document is None:

        return ""

    content = []

    try:

        document = Document(
            io.BytesIO(
                file_bytes
            )
        )

        # ----------------------------------------------------
        # PARAGRAPHS
        # ----------------------------------------------------

        for paragraph in document.paragraphs:

            paragraph_text = clean_text(
                paragraph.text
            )

            if paragraph_text:

                content.append(
                    paragraph_text
                )

        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        for table in document.tables:

            for row in table.rows:

                row_values = []

                for cell in row.cells:

                    cell_text = clean_text(
                        cell.text
                    )

                    if cell_text:

                        row_values.append(
                            cell_text
                        )

                if row_values:

                    content.append(
                        " | ".join(
                            row_values
                        )
                    )

    except Exception as e:

        print(
            "DOCX text extraction error:",
            e
        )

    # --------------------------------------------------------
    # EMBEDDED IMAGES
    # --------------------------------------------------------

    image_text = (
        extract_docx_embedded_images(
            file_bytes
        )
    )

    if image_text:

        content.append(
            image_text
        )

    return "\n".join(
        content
    ).strip()


# ============================================================
# GENERIC FILE EXTRACTION
# ============================================================

def extract_file_text(
    file_storage
):

    filename = (
        file_storage.filename or ""
    ).lower()

    file_bytes = file_storage.read()

    if not file_bytes:

        return ""

    # IMAGE
    if filename.endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff"
        )
    ):

        return extract_image_text(
            file_bytes
        )

    # PDF
    if filename.endswith(
        ".pdf"
    ):

        return extract_pdf_text(
            file_bytes
        )

    # DOCX
    if filename.endswith(
        ".docx"
    ):

        return extract_docx_text(
            file_bytes
        )

    # TEXT / CSV
    if filename.endswith(
        (
            ".txt",
            ".csv"
        )
    ):

        try:

            return file_bytes.decode(
                "utf-8",
                errors="ignore"
            )

        except Exception:

            return ""

    return ""


# ============================================================
# FIELD EXTRACTION HELPERS
#
# IMPROVED:
# Supports both:
#   Company: TechNova
#   Company | TechNova
#   Company - TechNova
# ============================================================

def extract_labeled_value(
    text,
    labels
):

    for label in labels:

        pattern = (
            rf"^\s*{re.escape(label)}"
            rf"\s*(?::|\-|\|)\s*(.+)$"
        )

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.MULTILINE
        )

        if match:

            value = match.group(
                1
            ).strip()

            value = value.split(
                "\n"
            )[0].strip()

            return value

    return ""


# ============================================================
# TITLE EXTRACTION
# ============================================================

def extract_title(
    text
):

    title = extract_labeled_value(
        text,
        [
            "Job Title",
            "Position",
            "Role",
            "Designation",
            "Internship Title",
            "Internship Role"
        ]
    )

    if title:

        return title

    lines = [

        clean_text(line)

        for line in text.splitlines()

        if clean_text(line)

    ]

    for line in lines[:10]:

        lower_line = line.lower()

        if lower_line in {
            "we are hiring",
            "we're hiring",
            "job opportunity",
            "career opportunity",
            "apply now",
            "urgent hiring"
        }:

            continue

        if (
            "intern" in lower_line
            or "developer" in lower_line
            or "engineer" in lower_line
            or "analyst" in lower_line
            or "designer" in lower_line
            or "manager" in lower_line
            or "associate" in lower_line
            or "software" in lower_line
        ):

            if len(line) <= 150:

                return line

    for line in lines[:8]:

        if len(line) <= 120:

            return line

    return ""


# ============================================================
# COMPANY EXTRACTION
# ============================================================

def extract_company(
    text,
    urls,
    emails
):

    # 1. Explicit company label

    company = extract_labeled_value(
        text,
        [
            "Company",
            "Company Name",
            "Organization",
            "Employer"
        ]
    )

    if company:

        return company

    # 2. Corporate email domain

    for email in emails:

        if not is_personal_email(
            email
        ):

            inferred = (
                company_name_from_email(
                    email
                )
            )

            if inferred:

                return inferred

    # 3. Website domain

    for url in urls:

        domain = get_domain_from_url(
            url
        )

        inferred = (
            company_name_from_domain(
                domain
            )
        )

        if inferred:

            return inferred

    # 4. Poster-style heading

    lines = [

        clean_text(line)

        for line in text.splitlines()

        if clean_text(line)

    ]

    for line in lines[:8]:

        lower_line = line.lower()

        if any(
            phrase in lower_line
            for phrase in [
                "we are hiring",
                "we're hiring",
                "apply now",
                "job opportunity",
                "career opportunity"
            ]
        ):

            continue

        if (
            "@" in line
            or "http://" in lower_line
            or "https://" in lower_line
        ):

            continue

        if len(line) <= 80:

            if lower_line not in {
                "location",
                "eligibility",
                "qualification",
                "duration",
                "stipend",
                "salary",
                "responsibilities",
                "requirements",
                "skills",
                "apply"
            }:

                if (
                    "intern" not in lower_line
                    and "developer" not in lower_line
                    and "engineer" not in lower_line
                    and "hiring" not in lower_line
                ):

                    return line

    return ""


# ============================================================
# SAFEHIRE FIELD EXTRACTION
# ============================================================

def extract_safehire_fields(
    text
):

    text = normalize_text(
        text
    )

    lower_text = text.lower()

    raw_urls = extract_urls(
        text
    )

    urls = [

        clean_url(url)

        for url in raw_urls

    ]

    emails = extract_emails(
        text
    )

    phones = extract_phone_numbers(
        text
    )

    fields = {

        "Job_Title": "",
        "Company_Name": "",
        "Domain": "",
        "Job_Type": "",
        "Job_Post": "",
        "Internship_Post": "",
        "Internship_Type": "",
        "Location": "",
        "Salary": "",
        "Stipend": "",
        "Experience": "",
        "Qualification": "",
        "Website": "",
        "Careers_Page": "",
        "Recruitment_Platforms": "",
        "Application_Method": "",
        "Contact_Provided": "",
        "Internship_Fee": "",
        "Fee_Type": ""

    }


    # JOB TITLE

    fields["Job_Title"] = extract_title(
        text
    )


    # COMPANY

    fields["Company_Name"] = extract_company(
        text,
        urls,
        emails
    )


    # LOCATION

    fields["Location"] = extract_labeled_value(
        text,
        [
            "Location",
            "Place",
            "Work Location",
            "Job Location"
        ]
    )


    # SALARY

    fields["Salary"] = extract_labeled_value(
        text,
        [
            "Salary",
            "CTC",
            "Package",
            "Compensation",
            "Pay"
        ]
    )


    # STIPEND

    fields["Stipend"] = extract_labeled_value(
        text,
        [
            "Stipend",
            "Monthly Stipend",
            "Internship Stipend"
        ]
    )


    # EXPERIENCE

    fields["Experience"] = extract_labeled_value(
        text,
        [
            "Experience",
            "Work Experience"
        ]
    )


    # QUALIFICATION

    fields["Qualification"] = extract_labeled_value(
        text,
        [
            "Qualification",
            "Education",
            "Eligibility",
            "Requirements"
        ]
    )


    # WEBSITE

    if urls:

        fields["Website"] = urls[0]


    # DOMAIN

    fields["Domain"] = detect_domain_from_urls(
        urls
    )


    # COMPANY FROM DOMAIN

    if not fields["Company_Name"]:

        inferred_company = (
            company_name_from_domain(
                fields["Domain"]
            )
        )

        if inferred_company:

            fields["Company_Name"] = (
                inferred_company
            )


    # CAREERS PAGE

    careers_url = ""

    for url in urls:

        if re.search(
            r"(career|careers|jobs|job|"
            r"university|internship|students)",
            url,
            re.IGNORECASE
        ):

            careers_url = url

            break


    if careers_url:

        fields["Careers_Page"] = (
            careers_url
        )

    elif re.search(
        r"\bofficial\s+careers?\s+page\b",
        lower_text
    ):

        fields["Careers_Page"] = (
            "Official careers page mentioned"
        )

    elif re.search(
        r"\bcompany.?s\s+official\s+careers?\s+page\b",
        lower_text
    ):

        fields["Careers_Page"] = (
            "Official careers page mentioned"
        )

    elif re.search(
        r"\bcareers?\s+page\b",
        lower_text
    ):

        fields["Careers_Page"] = (
            "Careers page mentioned"
        )


    # JOB / INTERNSHIP TYPE

    if "internship" in lower_text:

        fields["Internship_Post"] = text

        fields["Job_Type"] = (
            "Internship"
        )

        # IMPROVEMENT:
        # A stipend itself means the internship is paid.

        if (
            re.search(
                r"\bpaid\s+internship\b",
                lower_text
            )
            or fields["Stipend"]
            or re.search(
                r"\bstipend\b",
                lower_text
            )
        ):

            fields["Internship_Type"] = (
                "Paid"
            )

        elif re.search(
            r"\bunpaid\s+internship\b",
            lower_text
        ):

            fields["Internship_Type"] = (
                "Unpaid"
            )

    else:

        fields["Job_Post"] = text

        if re.search(
            r"\bfull[-\s]?time\b",
            lower_text
        ):

            fields["Job_Type"] = (
                "Full-time"
            )

        elif re.search(
            r"\bpart[-\s]?time\b",
            lower_text
        ):

            fields["Job_Type"] = (
                "Part-time"
            )

        elif re.search(
            r"\bcontract\b",
            lower_text
        ):

            fields["Job_Type"] = (
                "Contract"
            )


    # CONTACT

    contact_parts = []

    if emails:

        contact_parts.extend(
            emails
        )

    if phones:

        contact_parts.extend(
            phones
        )

    fields["Contact_Provided"] = (
        ", ".join(
            contact_parts
        )
    )


    # RECRUITMENT PLATFORMS

    platforms = []

    platform_map = {

        "linkedin": "LinkedIn",
        "naukri": "Naukri",
        "indeed": "Indeed",
        "internshala": "Internshala",
        "foundit": "Foundit",
        "monster": "Monster",
        "whatsapp": "WhatsApp",
        "telegram": "Telegram"

    }

    for keyword, name in platform_map.items():

        if keyword in lower_text:

            platforms.append(
                name
            )

    fields["Recruitment_Platforms"] = (
        ", ".join(
            dict.fromkeys(
                platforms
            )
        )
    )


    # APPLICATION METHOD

    if (
        fields.get(
            "Careers_Page"
        )
        or re.search(
            r"\bapply\s+(?:through|via|on)\s+"
            r"(?:the\s+)?(?:company.?s\s+)?"
            r"official\s+(?:careers?|website)",
            lower_text
        )
        or re.search(
            r"\bofficial\s+company\s+website\b",
            lower_text
        )
        or re.search(
            r"\bofficial\s+application\b",
            lower_text
        )
    ):

        fields["Application_Method"] = (
            "Official company channel"
        )

    elif "whatsapp" in lower_text:

        fields["Application_Method"] = (
            "WhatsApp"
        )

    elif "telegram" in lower_text:

        fields["Application_Method"] = (
            "Telegram"
        )

    elif emails:

        fields["Application_Method"] = (
            "Email"
        )

    elif urls:

        fields["Application_Method"] = (
            "Website"
        )


    # ========================================================
    # INTERNSHIP / APPLICATION FEE
    # ========================================================

    fee_patterns = [

        r"\bregistration\s+fee\b",
        r"\bapplication\s+fee\b",
        r"\bprocessing\s+fee\b",
        r"\bverification\s+fee\b",
        r"\brecruitment\s+fee\b",
        r"\bjoining\s+fee\b",
        r"\bonboarding\s+fee\b",
        r"\btraining\s+fee\b",
        r"\bprogram\s+fee\b",
        r"\binternship\s+fee\b"

    ]

    actual_fee_found = False

    for pattern in fee_patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            if not is_negated_fee_context(
                text,
                match.start(),
                match.end()
            ):

                fields["Internship_Fee"] = (
                    match.group(0)
                )

                actual_fee_found = True

                break

        if actual_fee_found:

            break


    # FEE TYPE

    if actual_fee_found:

        fee_type_patterns = [

            (
                r"\bregistration\s+fee\b",
                "Registration Fee"
            ),

            (
                r"\bapplication\s+fee\b",
                "Application Fee"
            ),

            (
                r"\bprocessing\s+fee\b",
                "Processing Fee"
            ),

            (
                r"\bverification\s+fee\b",
                "Verification Fee"
            ),

            (
                r"\brecruitment\s+fee\b",
                "Recruitment Fee"
            ),

            (
                r"\bjoining\s+fee\b",
                "Joining Fee"
            ),

            (
                r"\bonboarding\s+fee\b",
                "Onboarding Fee"
            ),

            (
                r"\btraining\s+fee\b",
                "Training Fee"
            ),

            (
                r"\bprogram\s+fee\b",
                "Program Fee"
            ),

            (
                r"\binternship\s+fee\b",
                "Internship Fee"
            )

        ]

        for pattern, fee_name in fee_type_patterns:

            if re.search(
                pattern,
                fields["Internship_Fee"],
                re.IGNORECASE
            ):

                fields["Fee_Type"] = (
                    fee_name
                )

                break


    return fields


# ============================================================
# STRUCTURED TEXT
# MUST MATCH TRAINING STYLE
# ============================================================

def create_structured_text(
    fields
):

    parts = []

    for field in ML_FIELDS:

        value = normalize_text(
            fields.get(
                field,
                ""
            )
        )

        if value:

            parts.append(
                f"{field}: {value}"
            )

    return " | ".join(
        parts
    )


# ============================================================
# NATURAL TEXT
# ============================================================

def create_natural_text(
    fields
):

    parts = []

    if fields.get(
        "Job_Title"
    ):

        parts.append(
            f"The job title is "
            f"{fields['Job_Title']}."
        )

    if fields.get(
        "Company_Name"
    ):

        parts.append(
            f"The company is "
            f"{fields['Company_Name']}."
        )

    if fields.get(
        "Domain"
    ):

        parts.append(
            f"The domain is "
            f"{fields['Domain']}."
        )

    if fields.get(
        "Job_Type"
    ):

        parts.append(
            f"The job type is "
            f"{fields['Job_Type']}."
        )

    if fields.get(
        "Job_Post"
    ):

        parts.append(
            f"Job details: "
            f"{fields['Job_Post']}."
        )

    if fields.get(
        "Internship_Post"
    ):

        parts.append(
            f"Internship details: "
            f"{fields['Internship_Post']}."
        )

    if fields.get(
        "Internship_Type"
    ):

        parts.append(
            f"Internship type: "
            f"{fields['Internship_Type']}."
        )

    if fields.get(
        "Location"
    ):

        parts.append(
            f"The location is "
            f"{fields['Location']}."
        )

    if fields.get(
        "Salary"
    ):

        parts.append(
            f"The salary is "
            f"{fields['Salary']}."
        )

    if fields.get(
        "Stipend"
    ):

        parts.append(
            f"The stipend is "
            f"{fields['Stipend']}."
        )

    if fields.get(
        "Experience"
    ):

        parts.append(
            f"The required experience is "
            f"{fields['Experience']}."
        )

    if fields.get(
        "Qualification"
    ):

        parts.append(
            f"The qualification is "
            f"{fields['Qualification']}."
        )

    if fields.get(
        "Website"
    ):

        parts.append(
            f"The website is "
            f"{fields['Website']}."
        )

    if fields.get(
        "Careers_Page"
    ):

        parts.append(
            f"The careers page is "
            f"{fields['Careers_Page']}."
        )

    if fields.get(
        "Recruitment_Platforms"
    ):

        parts.append(
            f"Recruitment platforms include "
            f"{fields['Recruitment_Platforms']}."
        )

    if fields.get(
        "Application_Method"
    ):

        parts.append(
            f"The application method is "
            f"{fields['Application_Method']}."
        )

    if fields.get(
        "Contact_Provided"
    ):

        parts.append(
            f"Contact information provided: "
            f"{fields['Contact_Provided']}."
        )

    if fields.get(
        "Internship_Fee"
    ):

        parts.append(
            f"Internship fee information: "
            f"{fields['Internship_Fee']}."
        )

    if fields.get(
        "Fee_Type"
    ):

        parts.append(
            f"Fee type: "
            f"{fields['Fee_Type']}."
        )

    return " ".join(
        parts
    )


# ============================================================
# INDICATOR ANALYSIS
# ============================================================

def analyze_indicators(
    text,
    fields
):

    text = normalize_text(
        text
    )

    lower_text = text.lower()

    red_flags = []
    caution_indicators = []
    positive_indicators = []


    # ========================================================
    # POSITIVE EVIDENCE
    # ========================================================

    if fields.get(
        "Website"
    ):

        positive_indicators.append(
            "An official website or web address is provided"
        )


    if (
        fields.get(
            "Careers_Page"
        )
        or re.search(
            r"\bofficial\s+careers?\s+page\b",
            lower_text
        )
        or re.search(
            r"\bcompany.?s\s+official\s+careers?\s+page\b",
            lower_text
        )
    ):

        positive_indicators.append(
            "An official careers page is mentioned"
        )


    if (
        fields.get(
            "Application_Method"
        )
        == "Official company channel"
        or re.search(
            r"(official\s+careers?\s+page|"
            r"official\s+company\s+website|"
            r"official\s+application|"
            r"company.?s\s+official\s+careers?)",
            lower_text
        )
    ):

        positive_indicators.append(
            "Application is directed through an official company channel"
        )


    recognized_platforms = [

        "linkedin",
        "naukri",
        "indeed",
        "internshala",
        "foundit"

    ]

    if any(
        platform in lower_text
        for platform in recognized_platforms
    ):

        positive_indicators.append(
            "A recognized recruitment platform is mentioned"
        )


    no_payment_patterns = [

        r"\bno\s+registration\s+fee\b",
        r"\bno\s+application\s+fee\b",
        r"\bno\s+processing\s+fee\b",
        r"\bno\s+verification\s+fee\b",
        r"\bno\s+recruitment\s+fee\b",
        r"\bno\s+joining\s+fee\b",
        r"\bno\s+onboarding\s+fee\b",
        r"\bno\s+payment\s+(?:is\s+)?required\b",
        r"\bno\s+payment\s+is\s+needed\b",
        r"\bapplicants?\s+(?:are\s+)?not\s+required\s+to\s+pay\b",
        r"\bno\s+fee\s+(?:is\s+)?required\b",
        r"\bno\s+fees?\s+(?:are\s+)?required\b",
        r"\bwithout\s+(?:any\s+)?fee\b"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in no_payment_patterns
    ):

        positive_indicators.append(
            "The opportunity explicitly states that applicants are not required to make a payment"
        )


    compensation_patterns = [

        r"\bsalary\b",
        r"\bstipend\b",
        r"\bcompensation\b",
        r"\bpaid\s+internship\b",
        r"\bper\s+month\b",
        r"\bper\s+year\b",
        r"\bper\s+annum\b",
        r"\blpa\b",
        r"₹\s*[\d,]+"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in compensation_patterns
    ):

        positive_indicators.append(
            "Salary or compensation information is stated"
        )


    if (
        "internship" in lower_text
        and (
            "paid internship" in lower_text
            or "stipend" in lower_text
        )
    ):

        positive_indicators.append(
            "The internship provides compensation to the applicant"
        )


    # ========================================================
    # RED FLAGS
    # ========================================================

    emails = extract_emails(
        text
    )


    # PERSONAL / UNOFFICIAL EMAIL

    personal_email_found = False

    for email in emails:

        if is_personal_email(
            email
        ):

            personal_email_found = True

            break

    if personal_email_found:

        red_flags.append(
            "A personal or unofficial email address is used for recruitment"
        )


    # EXPLICIT APPLICANT FEES

    applicant_fee_patterns = [

        r"\bregistration\s+fee\b",
        r"\bapplication\s+fee\b",
        r"\bprocessing\s+fee\b",
        r"\bverification\s+fee\b",
        r"\brecruitment\s+fee\b",
        r"\bjoining\s+fee\b",
        r"\bonboarding\s+fee\b"

    ]

    applicant_fee_found = False

    for pattern in applicant_fee_patterns:

        for match in re.finditer(
            pattern,
            lower_text
        ):

            if not is_negated_fee_context(
                lower_text,
                match.start(),
                match.end()
            ):

                applicant_fee_found = True

                break

        if applicant_fee_found:

            break


    if applicant_fee_found:

        red_flags.append(
            "The opportunity asks the applicant for a registration, application, processing, verification, recruitment, joining, or similar fee"
        )


    # PAYMENT BEFORE INTERVIEW / SELECTION / JOINING

    payment_before_patterns = [

        r"\bpayment\b.{0,100}\bbefore\b.{0,100}\binterview\b",

        r"\bpay\b.{0,100}\bbefore\b.{0,100}\binterview\b",

        r"\bpayment\b.{0,100}\bbefore\b.{0,100}\bselection\b",

        r"\bpay\b.{0,100}\bbefore\b.{0,100}\bselection\b",

        r"\bpayment\b.{0,100}\bbefore\b.{0,100}\bjoining\b",

        r"\bpay\b.{0,100}\bbefore\b.{0,100}\bjoining\b",

        r"\bpayment\b.{0,100}\bbefore\b.{0,100}\bonboarding\b",

        r"\bpay\b.{0,100}\bbefore\b.{0,100}\bonboarding\b",

        r"\bmust\s+pay\b.{0,100}\bto\s+secure\b",

        r"\bpay\b.{0,100}\bto\s+secure\s+your\s+position\b"

    ]

    payment_before_found = False

    for pattern in payment_before_patterns:

        for match in re.finditer(
            pattern,
            lower_text,
            re.IGNORECASE
        ):

            if not is_negated_payment_context(
                lower_text,
                match.start(),
                match.end()
            ):

                payment_before_found = True

                break

        if payment_before_found:

            break


    if payment_before_found:

        red_flags.append(
            "Payment is requested before interview, selection, joining, or onboarding"
        )


    # IMPERSONATION

    impersonation_patterns = [

        r"\bimpersonat",

        r"\bpretending\s+to\s+be\b",

        r"\bposing\s+as\b",

        r"\bfake\s+representative\b",

        r"\bfake\s+recruiter\b",

        r"\blookalike\s+domain\b",

        r"\bnot\s+affiliated\s+with\b"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in impersonation_patterns
    ):

        red_flags.append(
            "The opportunity contains signs of possible company or recruiter impersonation"
        )


    # SENSITIVE INFORMATION

    sensitive_info_patterns = [

        r"\baadhaar\b",
        r"\baadhar\b",
        r"\bpan\s+card\b",
        r"\bpan\s+details\b",
        r"\bpan\s+number\b",
        r"\bbank\s+account\b",
        r"\bbank\s+details\b",
        r"\bbank\s+information\b",
        r"\baccount\s+number\b",
        r"\bdebit\s+card\b",
        r"\bcredit\s+card\b",
        r"\bcard\s+details\b",
        r"\bnet\s+banking\b",
        r"\bupi\b",
        r"\bfinancial\s+information\b",
        r"\bfinancial\s+details\b"

    ]

    sensitive_information_found = any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in sensitive_info_patterns
    )

    if sensitive_information_found:

        red_flags.append(
            "The opportunity requests sensitive personal or financial information"
        )


    # ========================================================
    # CAUTION INDICATORS
    # ========================================================

    if (
        "whatsapp" in lower_text
        or "telegram" in lower_text
    ):

        caution_indicators.append(
            "Recruitment or communication uses a messaging platform and should be independently verified"
        )


    # UNSOLICITED CONTACT

    unsolicited_patterns = [

        r"\bwe\s+found\s+your\s+profile\b",

        r"\bwe\s+contacted\s+you\b",

        r"\bwe\s+reached\s+out\s+to\s+you\b",

        r"\bour\s+recruiter\s+contacted\s+you\b",

        r"\ba\s+recruiter\s+contacted\s+you\b",

        r"\byou\s+were\s+contacted\s+by\s+our\s+recruiter\b",

        r"\bunsolicited\s+job\s+offer\b",

        r"\bunsolicited\s+recruitment\b"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in unsolicited_patterns
    ):

        caution_indicators.append(
            "The opportunity appears to involve unsolicited recruitment contact; verify the source independently"
        )


    # RECRUITMENT AGENCY

    if re.search(
        r"\brecruitment\s+agency\b"
        r"|\bplacement\s+agency\b"
        r"|\bstaffing\s+agency\b",
        lower_text
    ):

        caution_indicators.append(
            "A recruitment or placement agency is involved; verify the agency and employer relationship"
        )


    # LIMITED COMPANY INFORMATION

    if not fields.get(
        "Company_Name"
    ):

        caution_indicators.append(
            "Company information is limited; further verification is recommended"
        )


    # INSUFFICIENT VERIFICATION INFORMATION

    if (
        not fields.get(
            "Website"
        )
        and not fields.get(
            "Careers_Page"
        )
        and not any(
            platform in lower_text
            for platform in [
                "linkedin",
                "naukri",
                "indeed",
                "internshala",
                "foundit"
            ]
        )
    ):

        caution_indicators.append(
            "The available information is insufficient to confidently verify the opportunity; further verification is recommended"
        )


    # PROGRAM / TRAINING / INTERNSHIP FEE

    program_fee_patterns = [

        r"\bprogram\s+fee\b",
        r"\btraining\s+fee\b",
        r"\bcourse\s+fee\b",
        r"\bparticipation\s+fee\b",
        r"\binternship\s+fee\b"

    ]

    program_fee_found = any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in program_fee_patterns
    )

    if (
        program_fee_found
        and not applicant_fee_found
    ):

        caution_indicators.append(
            "An internship, training, or program participation fee is mentioned; verify the program and payment terms carefully"
        )


    # URGENCY

    urgency_patterns = [

        r"\bact\s+now\b",
        r"\burgent\b",
        r"\btoday\s+only\b",
        r"\bimmediately\b",
        r"\blimited\s+slots\b",
        r"\bpay\s+today\b",
        r"\bwithin\s+\d+\s+hours\b",
        r"\bwithin\s+\d+\s+minutes\b"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in urgency_patterns
    ):

        caution_indicators.append(
            "The opportunity uses urgency or pressure; verify the offer before taking action"
        )


    # GUARANTEED / UNREALISTIC CLAIMS

    unrealistic_patterns = [

        r"\bguaranteed\s+job\b",
        r"\b100%\s+job\s+guarantee\b",
        r"\bguaranteed\s+placement\b",
        r"\bguaranteed\s+salary\b",
        r"\bno\s+interview\b",
        r"\bselected\s+immediately\b"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in unrealistic_patterns
    ):

        caution_indicators.append(
            "The opportunity contains unusually strong or guaranteed recruitment claims; verify independently"
        )


    # HIGH COMPENSATION

    high_compensation_patterns = [

        r"\b(?:salary|stipend|compensation)"
        r"\s*[:\-]?\s*₹\s*([\d,]+)",

        r"\b(?:salary|stipend|compensation)"
        r".{0,40}?"
        r"₹\s*([\d,]+)",

        r"₹\s*([\d,]+)"
        r".{0,40}?"
        r"\b(?:salary|stipend|compensation)\b"

    ]

    compensation_amount = None

    for pattern in high_compensation_patterns:

        match = re.search(
            pattern,
            lower_text,
            re.IGNORECASE
        )

        if match:

            try:

                compensation_amount = float(
                    match.group(
                        1
                    ).replace(
                        ",",
                        ""
                    )
                )

                break

            except ValueError:

                pass


    if compensation_amount is not None:

        if compensation_amount >= 1000000:

            caution_indicators.append(
                "The salary or compensation claim appears unusually high; verify the opportunity independently"
            )


    # UNREALISTIC SALARY CLAIMS

    salary_claim_patterns = [

        r"\bhuge\s+salary\b",
        r"\bvery\s+high\s+salary\b",
        r"\bhigh\s+salary\s+guaranteed\b",
        r"\bunlimited\s+salary\b",
        r"\bearn\s+lakhs\s+immediately\b"

    ]

    if any(
        re.search(
            pattern,
            lower_text
        )
        for pattern in salary_claim_patterns
    ):

        caution_indicators.append(
            "The compensation claim appears unusually strong or unrealistic; verify the opportunity independently"
        )


    # EXPLICIT NO WEBSITE / NO CAREERS PAGE

    if re.search(
        r"\bno\s+official\s+website\b",
        lower_text
    ):

        caution_indicators.append(
            "The opportunity explicitly states that no official website is available"
        )


    if re.search(
        r"\bno\s+official\s+careers?\s+page\b",
        lower_text
    ):

        caution_indicators.append(
            "The opportunity explicitly states that no official careers page is available"
        )


    # REMOVE DUPLICATES

    positive_indicators = list(
        dict.fromkeys(
            positive_indicators
        )
    )

    red_flags = list(
        dict.fromkeys(
            red_flags
        )
    )

    caution_indicators = list(
        dict.fromkeys(
            caution_indicators
        )
    )


    return (
        positive_indicators,
        red_flags,
        caution_indicators
    )


# ============================================================
# SAFEHIRE FINAL DECISION
#
# HYBRID DECISION LOGIC
#
# ML prediction is one signal.
# Rule-based fraud indicators are stronger safety signals.
# Positive evidence helps resolve ML false positives.
# ============================================================

def make_safehire_decision(
    prediction,
    red_flags,
    caution_indicators,
    positive_indicators
):

    red_count = len(
        red_flags
    )

    caution_count = len(
        caution_indicators
    )

    positive_count = len(
        positive_indicators
    )


    red_text = " ".join(
        item.lower()
        for item in red_flags
    )


    # --------------------------------------------------------
    # STRONG RED FLAGS
    # --------------------------------------------------------

    payment_related = any(
        word in red_text
        for word in [
            "fee",
            "payment"
        ]
    )

    impersonation_present = (
        "impersonat" in red_text
        or "fake recruiter" in red_text
        or "fake representative" in red_text
        or "lookalike domain" in red_text
    )

    sensitive_information_present = (
        "sensitive personal or financial"
        in red_text
    )


    # --------------------------------------------------------
    # TWO OR MORE RED FLAGS
    #
    # Multiple independent red flags are sufficient for Fake.
    # --------------------------------------------------------

    if red_count >= 2:

        return "Fake"


    # --------------------------------------------------------
    # ONE STRONG RED FLAG
    #
    # Payment / impersonation / sensitive information
    # is treated more seriously than a personal email alone.
    # --------------------------------------------------------

    if red_count == 1:

        if (
            payment_related
            or impersonation_present
            or sensitive_information_present
        ):

            return "Fake"

        return "Suspicious"


    # --------------------------------------------------------
    # NO RED FLAGS
    #
    # First evaluate the rule-based evidence.
    # --------------------------------------------------------

    if red_count == 0:

        # Clean opportunity with strong positive evidence.
        #
        # This is important for technical internships,
        # DOCX tables and normal job PDFs where the ML model
        # can occasionally produce a false positive.

        strong_positive_evidence = (
            positive_count >= 3
            and caution_count == 0
        )


        # ML says Genuine:
        # keep Genuine when there are no serious cautions.

        if prediction == "Genuine":

            if caution_count >= 2:

                return "Suspicious"

            return "Genuine"


        # ML says Fake but the opportunity contains several
        # strong positive indicators and no caution/red flags.
        #
        # Treat this as a model false-positive candidate rather
        # than immediately showing Fake.

        if prediction == "Fake":

            if strong_positive_evidence:

                return "Genuine"

            if caution_count == 0:

                return "Suspicious"

            return "Suspicious"


        # Unknown / unexpected ML label

        if (
            strong_positive_evidence
            and caution_count == 0
        ):

            return "Genuine"


    # --------------------------------------------------------
    # CAUTION SIGNALS
    # --------------------------------------------------------

    if caution_count >= 2:

        return "Suspicious"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "Suspicious"


# ============================================================
# SAFEHIRE CONFIDENCE
#
# Confidence represents the combined SafeHire assessment,
# not the internal ML probability.
# ============================================================

def calculate_safehire_confidence(
    result,
    ml_confidence,
    positive_indicators,
    red_flags,
    caution_indicators,
    ml_prediction
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


    ml_base = (
        ml_confidence * 0.60
    )


    # --------------------------------------------------------
    # GENUINE
    # --------------------------------------------------------

    if result == "Genuine":

        confidence = (

            ml_base

            + min(
                positive_count * 12,
                48
            )

            - min(
                caution_count * 4,
                12
            )

            - min(
                red_count * 15,
                45
            )

        )


        # If ML disagrees with a clean positive assessment,
        # do not give an excessively high confidence.
        if (
            ml_prediction == "Fake"
            and red_count == 0
        ):

            confidence = min(
                confidence,
                88
            )


        return round(
            max(
                0,
                min(
                    confidence,
                    99
                )
            ),
            2
        )


    # --------------------------------------------------------
    # FAKE
    # --------------------------------------------------------

    if result == "Fake":

        confidence = (

            ml_base

            + min(
                red_count * 20,
                40
            )

            - min(
                positive_count * 3,
                9
            )

            - min(
                caution_count * 2,
                6
            )

        )

        return round(
            max(
                0,
                min(
                    confidence,
                    99
                )
            ),
            2
        )


    # --------------------------------------------------------
    # SUSPICIOUS
    # --------------------------------------------------------

    confidence = (

        55

        + min(
            red_count * 4,
            8
        )

        + min(
            caution_count * 3,
            9
        )

        + min(
            positive_count * 1,
            5
        )

    )

    return round(
        max(
            50,
            min(
                confidence,
                70
            )
        ),
        2
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def create_recommendation(
    result
):

    if result == "Fake":

        return (
            "Do not make the requested payment or share "
            "sensitive personal or financial information."
        )

    if result == "Suspicious":

        return (
            "Proceed with caution. Avoid payments or "
            "sensitive document sharing until the unclear "
            "recruitment details are resolved."
        )

    return (
        "The opportunity appears legitimate based on "
        "the information provided. You can continue "
        "through the stated recruitment process while "
        "following normal safety precautions."
    )


# ============================================================
# ML PREDICTION
# ============================================================

def run_ml_prediction(
    structured_text,
    natural_text
):

    if (
        model is None
        or vectorizer is None
    ):

        return (
            "Unknown",
            0.0
        )


    ml_text = (
        structured_text
        + " "
        + natural_text
    )


    try:

        X = vectorizer.transform(
            [ml_text]
        )

        prediction = model.predict(
            X
        )[0]

        prediction_text = str(
            prediction
        )


        # MAP MODEL LABELS

        if prediction_text.lower() in [

            "1",
            "fake",
            "fraud",
            "false"

        ]:

            prediction_text = "Fake"

        elif prediction_text.lower() in [

            "0",
            "genuine",
            "real",
            "true"

        ]:

            prediction_text = "Genuine"


        confidence = 0.0


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    X
                )[0]
            )

            confidence = float(
                max(
                    probabilities
                ) * 100
            )


        return (
            prediction_text,
            confidence
        )


    except Exception as e:

        print(
            "ML prediction error:",
            e
        )

        return (
            "Unknown",
            0.0
        )


# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        input_text = ""

        source_type = "text"


        # ====================================================
        # JSON REQUEST
        # ====================================================

        if request.is_json:

            data = request.get_json(
                silent=True
            ) or {}


            input_text = normalize_text(
                data.get(
                    "job_text"
                )
                or data.get(
                    "text"
                )
                or data.get(
                    "description"
                )
            )


            url = normalize_text(
                data.get(
                    "url"
                )
            )


            if url:

                source_type = "url"

                url_text = fetch_url_text(
                    url
                )

                if url_text:

                    input_text = (
                        input_text
                        + "\n"
                        + url_text
                    ).strip()


        # ====================================================
        # FORM REQUEST
        # ====================================================

        else:

            input_text = normalize_text(
                request.form.get(
                    "job_text"
                )
                or request.form.get(
                    "text"
                )
            )


            # URL

            url = normalize_text(
                request.form.get(
                    "url"
                )
            )


            if url:

                source_type = "url"

                url_text = fetch_url_text(
                    url
                )

                if url_text:

                    input_text = (
                        input_text
                        + "\n"
                        + url_text
                    ).strip()


            # FILE

            uploaded_file = (
                request.files.get(
                    "file"
                )
            )


            if uploaded_file:

                source_type = "file"

                file_text = extract_file_text(
                    uploaded_file
                )


                if file_text:

                    input_text = (
                        input_text
                        + "\n"
                        + file_text
                    ).strip()


        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        if not input_text.strip():

            return jsonify(
                {
                    "error": (
                        "No job or internship "
                        "information was provided."
                    )
                }
            ), 400


        # ====================================================
        # EXTRACT SAFEHIRE FIELDS
        # ====================================================

        fields = extract_safehire_fields(
            input_text
        )


        # ====================================================
        # CREATE MODEL TEXT
        # ====================================================

        structured_text = (
            create_structured_text(
                fields
            )
        )

        natural_text = (
            create_natural_text(
                fields
            )
        )


        # ====================================================
        # ML PREDICTION
        # ====================================================

        (
            ml_prediction,
            ml_confidence
        ) = run_ml_prediction(
            structured_text,
            natural_text
        )


        # ====================================================
        # INDICATOR ANALYSIS
        # ====================================================

        (
            positive_indicators,
            red_flags,
            caution_indicators

        ) = analyze_indicators(
            input_text,
            fields
        )


        # ====================================================
        # FINAL SAFEHIRE DECISION
        # ====================================================

        safehire_result = (
            make_safehire_decision(
                ml_prediction,
                red_flags,
                caution_indicators,
                positive_indicators
            )
        )


        # ====================================================
        # SAFEHIRE CONFIDENCE
        # ====================================================

        safehire_confidence = (
            calculate_safehire_confidence(
                safehire_result,
                ml_confidence,
                positive_indicators,
                red_flags,
                caution_indicators,
                ml_prediction
            )
        )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        recommendation = (
            create_recommendation(
                safehire_result
            )
        )


        # ====================================================
        # REASONS
        # ====================================================

        reasons = []


        for item in red_flags:

            reasons.append(
                item
            )


        for item in caution_indicators:

            reasons.append(
                item
            )


        if (
            not red_flags
            and not caution_indicators
        ):

            reasons.append(
                "No strong fraud indicators were identified from the provided information."
            )


        # ====================================================
        # RESPONSE
        #
        # ML prediction remains internal.
        # ====================================================

        response = {

            "safehire_result":
                safehire_result,

            "confidence":
                safehire_confidence,

            "reasons":
                reasons,

            "recommendation":
                recommendation,

            "positive_indicators":
                positive_indicators,

            "red_flags":
                red_flags,

            "caution_indicators":
                caution_indicators,

            "safehire_fields":
                fields,

            "structured_text":
                structured_text,

            "natural_text":
                natural_text,

            "ml_text":
                structured_text
                + " "
                + natural_text,

            "extracted_text":
                input_text,

            # INTERNAL ONLY

            "ml_prediction":
                ml_prediction,

            "ml_confidence":
                ml_confidence,

            "source_type":
                source_type

        }


        return jsonify(
            response
        ), 200


    except Exception as e:

        print(
            "Prediction endpoint error:",
            e
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify(
        {
            "status":
                "SafeHire backend running",

            "endpoint":
                "/predict"
        }
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 60
    )

    print(
        "SAFEHIRE BACKEND STARTING"
    )

    print(
        "=" * 60
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "Endpoint: http://127.0.0.1:5000/predict"
    )

    print(
        "=" * 60
    )

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )