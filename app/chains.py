import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        
        # Include resume data directly as a dictionary
        self.resume_data = {
            "TechnicalSkills": {
                "Languages": ["Python", "Java", "SQL", ".JS", "HTML-5", "CSS-3"],
                "Technologies": ["Docker", "Kubernetes", "Matlab", "Randoop", "Evosuite", "Maven", "Wireshark", "NumPy", "Pandas", "MySQL"],
                "Others": ["Azure", "Jira", "Github", "Agile", "Web-Scraping", "MS365", "UML", "Software Design", "DSA", "TCP/IP"]
            },
            "Experience": [
                {
                    "Company": "Acosta, BestBuy",
                    "Position": "HP Product Associate",
                    "Duration": "April 2023 – Present",
                    "Location": "Brampton, ON, Canada",
                    "Technologies": ["Salesforce", "Alteryx", "Power BI", "Excel"],
                    "Responsibilities": [
                        "Managed customer relationships and analyzed customer data using Salesforce CRM, maintaining an NPS of 87%.",
                        "Analyzed data with Excel and Tableau, improving data management efficiency by 15%.",
                        "Achieved monthly sales of $200,000, acquiring 200+ memberships, contributing to a 27% quarterly sales increase.",
                        "Handled data entry and managed 1,000+ tagged items within TREX system, ensuring inventory accuracy and efficient customer service."
                    ]
                },
                {
                    "Company": "CGI",
                    "Position": "Software Engineering Intern (Automation QA)",
                    "Duration": "January 2022 – Dec 2022",
                    "Location": "Toronto, ON, Canada",
                    "Technologies": ["Python", "Selenium", "SoapUI", "Postman", "SQL", "Power BI", "Alteryx"],
                    "Responsibilities": [
                        "Created 500+ automated test scripts with Robot Framework, reducing manual testing by 30%.",
                        "Enhanced reporting with Power BI and Alteryx, cutting report generation time by 40%.",
                        "Performed API testing using SoapUI and Postman.",
                        "Improved debugging by logging bugs in XML, decreasing defect turnaround by 20%.",
                        "Developed dashboards integrated with JAMA, Jira, and SQL servers for QA defect resolution.",
                        "Conducted smoke, performance, and regression testing using HP ALM, improving system stability by 25%."
                    ]
                },
                {
                    "Company": "York University (UIT)",
                    "Position": "Computing Support Assistant",
                    "Duration": "September 2021 – December 2021",
                    "Location": "Toronto, ON, Canada",
                    "Technologies": ["Jira"],
                    "Responsibilities": [
                        "Provided technical support through Jira, resolving 15-20 tickets daily.",
                        "Improved ticket resolution rate by 15%, achieving 90% within 24 hours."
                    ]
                }
            ],
            "Projects": [
                {
                    "Name": "LinkedIn Clone",
                    "Technologies": ["React", "Redux", "Firebase", "Git"],
                    "Description": "Developed a LinkedIn clone with Firebase authentication for user login and post-sharing functionality."
                },
                {
                    "Name": "Java GUI Parking System Application",
                    "Technologies": ["React.js", "Node.js", "JavaScript", "Oracle", "Agile"],
                    "Description": "Built front-end with React and JavaScript, integrated with Node.js back-end and Oracle database, achieving 90% code coverage using Randoop and Evosuite."
                }
            ],
            "Education": {
                "Institution": "York University (Lassonde School of Engineering)",
                "Degree": "Bachelor of Engineering (B.Eng) in Computer Engineering with Specialized Honors",
                "Duration": "January 2020 - May 2024",
                "GPA": "3.3/4",
                "Location": "Toronto, ON, Canada"
            },
            "Certifications": [
                {
                    "Name": "Microsoft AZ-900 (Azure Fundamentals)",
                    "Link": "Link"
                },
                {
                    "Name": "Bergeron Entrepreneur for Science and Technology (BEST)",
                    "Link": "Link"
                },
                {
                    "Name": "Microsoft Excel Crash Course for Business Analysts",
                    "Link": "Link"
                }
            ],
            "PersonalDetails": {
                "Name": "Rajat",
                "Phone": "+1(647)-879-3424",
                "Email": "rajat.rajat0210@gmail.com",
                "LinkedIn": "https://www.linkedin.com/in/rajat-rajat12/",
                "Location": "Brampton, ON, Canada"
            }
        }

    def answer_question(self, user_question):
        # Resume in JSON format as a string
        resume_json = json.dumps(self.resume_data, indent=2)

        # Prompt template for general Q&A
        prompt_qa = PromptTemplate.from_template(
            """
            ### RESUME DETAILS (JSON FORMAT):
            {resume_data}

            ### INSTRUCTION:
            A user has asked the following question about Rajat:
            {user_question}
            
            Based on the resume details above, answer the user's question in a conversational and concise tone.
            Only provide the answer, no preamble or JSON output.
            """
        )

        # Combine prompt with LLM
        chain_qa = prompt_qa | self.llm
        response = chain_qa.invoke({"resume_data": resume_json, "user_question": user_question})
        return response.content

if __name__ == "__main__":
    # Example usage
    chain = Chain()
    user_question = "What projects has Rajat worked on?"
    print(chain.answer_question(user_question))
