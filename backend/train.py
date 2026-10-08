import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
 

# ============================================================
# 1. READ FINAL 600-RECORD DATASET
# ============================================================

data = pd.read_csv(
    "dataset/Safehire_Dataset.csv",
    encoding="latin1"
)

data = data.fillna("")

print("Dataset loaded successfully!")
print("Total records:", len(data))

print("\nLabel distribution:")
print(data["Label"].value_counts())


# ============================================================
# 2. DEFINE THE 19 FIELDS USED BY SAFEHIRE
# ============================================================

ml_fields = [
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
# 3. CREATE STRUCTURED REPRESENTATION
# ============================================================

def create_structured_text(row):

    parts = []

    for field in ml_fields:
        value = str(row[field]).strip()

        if value:
            parts.append(value)

    return " ".join(parts)


data["Structured_Text"] = data.apply(
    create_structured_text,
    axis=1
)


# ============================================================
# 4. CREATE NATURAL-LANGUAGE REPRESENTATION
#
# This converts the existing dataset information into a form
# closer to how a real user may describe an opportunity.
#
# IMPORTANT:
# This does NOT create new fake records.
# It only rephrases information already present in each row.
# ============================================================

def create_natural_text(row):

    parts = []

    # --------------------------------------------------------
    # JOB / INTERNSHIP TITLE
    # --------------------------------------------------------

    job_title = str(row["Job_Title"]).strip()
    company = str(row["Company_Name"]).strip()
    location = str(row["Location"]).strip()

    if job_title and job_title != "N/A":
        if company and company != "N/A":
            parts.append(
                f"{company} is offering a {job_title} opportunity."
            )
        else:
            parts.append(
                f"There is a {job_title} opportunity."
            )

    # --------------------------------------------------------
    # DOMAIN
    # --------------------------------------------------------

    domain = str(row["Domain"]).strip()

    if domain and domain != "N/A":
        parts.append(
            f"The opportunity is related to {domain}."
        )

    # --------------------------------------------------------
    # JOB TYPE
    # --------------------------------------------------------

    job_type = str(row["Job_Type"]).strip()

    if job_type and job_type != "N/A":
        parts.append(
            f"It is a {job_type} position."
        )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if location and location != "N/A":
        parts.append(
            f"The location is {location}."
        )

    # --------------------------------------------------------
    # JOB POST
    # --------------------------------------------------------

    job_post = str(row["Job_Post"]).strip()

    if job_post and job_post != "N/A":
        parts.append(job_post)

    # --------------------------------------------------------
    # INTERNSHIP INFORMATION
    # --------------------------------------------------------

    internship_post = str(
        row["Internship_Post"]
    ).strip()

    internship_type = str(
        row["Internship_Type"]
    ).strip()

    if internship_post and internship_post != "N/A":

        parts.append(
            internship_post
        )

    if internship_type and internship_type != "N/A":

        parts.append(
            f"The internship type is {internship_type}."
        )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if location and location != "N/A":
        pass

    # --------------------------------------------------------
    # SALARY
    # --------------------------------------------------------

    salary = str(row["Salary"]).strip()

    if salary and salary != "N/A":
        parts.append(
            f"The salary is {salary}."
        )

    # --------------------------------------------------------
    # STIPEND
    # --------------------------------------------------------

    stipend = str(row["Stipend"]).strip()

    if stipend and stipend != "N/A":
        parts.append(
            f"The stipend is {stipend}."
        )

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    experience = str(row["Experience"]).strip()

    if experience and experience != "N/A":
        parts.append(
            f"The experience requirement is {experience}."
        )

    # --------------------------------------------------------
    # QUALIFICATION
    # --------------------------------------------------------

    qualification = str(
        row["Qualification"]
    ).strip()

    if qualification and qualification != "N/A":
        parts.append(
            f"The qualification requirement is {qualification}."
        )

    # --------------------------------------------------------
    # WEBSITE
    # --------------------------------------------------------

    website = str(row["Website"]).strip()

    if website and website != "N/A":
        parts.append(
            f"Website information: {website}."
        )

    # --------------------------------------------------------
    # CAREERS PAGE
    # --------------------------------------------------------

    careers = str(
        row["Careers_Page"]
    ).strip()

    if careers and careers != "N/A":
        parts.append(
            f"Careers page information: {careers}."
        )

    # --------------------------------------------------------
    # RECRUITMENT PLATFORM
    # --------------------------------------------------------

    platforms = str(
        row["Recruitment_Platforms"]
    ).strip()

    if platforms and platforms != "N/A":
        parts.append(
            f"Recruitment platform: {platforms}."
        )

    # --------------------------------------------------------
    # APPLICATION METHOD
    # --------------------------------------------------------

    application_method = str(
        row["Application_Method"]
    ).strip()

    if application_method and application_method != "N/A":
        parts.append(
            f"Application method: {application_method}."
        )

    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    contact = str(
        row["Contact_Provided"]
    ).strip()

    if contact and contact != "N/A":
        parts.append(
            f"Contact information: {contact}."
        )

    # --------------------------------------------------------
    # INTERNSHIP FEE
    # --------------------------------------------------------

    internship_fee = str(
        row["Internship_Fee"]
    ).strip()

    if internship_fee and internship_fee != "N/A":
        parts.append(
            f"Internship fee information: {internship_fee}."
        )

    # --------------------------------------------------------
    # FEE TYPE
    # --------------------------------------------------------

    fee_type = str(
        row["Fee_Type"]
    ).strip()

    if fee_type and fee_type != "N/A":
        parts.append(
            f"Fee type: {fee_type}."
        )

    return " ".join(parts)


data["Natural_Text"] = data.apply(
    create_natural_text,
    axis=1
)


# ============================================================
# 5. INPUT AND OUTPUT
# ============================================================

X = data[
    [
        "Structured_Text",
        "Natural_Text"
    ]
]

y = data["Label"]


# ============================================================
# 6. SPLIT ORIGINAL RECORDS FIRST
#
# IMPORTANT:
# We split the 600 original records BEFORE creating the
# training representations.
#
# This prevents the same original record from appearing
# in both training and testing representations.
# ============================================================

train_indices, test_indices = train_test_split(
    data.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_data = data.loc[train_indices].copy()
test_data = data.loc[test_indices].copy()

y_train = train_data["Label"]
y_test = test_data["Label"]


print("\nTraining original records:", len(train_data))
print("Testing original records:", len(test_data))


# ============================================================
# 7. COMBINE STRUCTURED + NATURAL TEXT FOR TRAINING
#
# Each ORIGINAL training record contributes both:
#
# 1. Structured representation
# 2. Natural-language representation
#
# They are joined into one ML document.
#
# We do NOT duplicate rows into the test set.
# ============================================================

X_train = (
    train_data["Structured_Text"].astype(str)
    + " "
    + train_data["Natural_Text"].astype(str)
)

X_test = (
    test_data["Structured_Text"].astype(str)
    + " "
    + test_data["Natural_Text"].astype(str)
)


# ============================================================
# 8. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=1,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("\nTF-IDF completed.")

print(
    "Number of TF-IDF features:",
    len(vectorizer.get_feature_names_out())
)


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model trained successfully!")
print("Classes:", model.classes_)


# ============================================================
# 10. TEST THE MODEL
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/fake_job_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")
print("TF-IDF vectorizer saved successfully!")


# ============================================================
# 13. TRAINING INFORMATION
# ============================================================

print("\n==============================")
print("TRAINING SUMMARY")
print("==============================")

print(
    "Original dataset records:",
    len(data)
)

print(
    "Training records:",
    len(train_data)
)

print(
    "Testing records:",
    len(test_data)
)

print(
    "Training representation:",
    "Structured + Natural Language"
)

print(
    "Data leakage prevention:",
    "Original records split before training"
)

print("==============================")