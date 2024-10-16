import streamlit as st
import pickle
import re
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def get_ats_score(classifier, input_features):
    # First, check if classifier has 'predict_proba' method
    if hasattr(classifier, 'predict_proba'):
        probabilities = classifier.predict_proba(input_features)
        ats_score = np.max(probabilities) * 100  # Use highest probability as ATS score
    else:
        # If 'predict_proba' is not available, fallback to decision function or set a default
        if hasattr(classifier, 'decision_function'):
            decision_scores = classifier.decision_function(input_features)
            ats_score = (np.max(decision_scores) / np.sum(np.abs(decision_scores))) * 1000 * -1
        else:
            ats_score = 50  # Default ATS score if no confidence measure is available
    return ats_score

# Web app
def main():
    st.title("Resume Screening App with ATS Score")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Get ATS score
        ats_score = get_ats_score(clf, input_features)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)
        st.write(f"ATS Score: {ats_score:.2f}%")  # Display ATS score

# python main
if __name__ == "__main__":
    main()
