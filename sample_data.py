"""
sample_data.py
--------------
Pre-configured realistic sample resumes and matching job descriptions.
Allows users and interviewers to test the application instantly with one click.
"""

SAMPLE_PRESETS = {
    "Full-Stack Software Engineer (Junior/New Grad)": {
        "resume": """ALEX CHEN
San Francisco, CA | alex.chen@email.com | (555) 019-2834 | linkedin.com/in/alexchen | github.com/alexchen

EDUCATION
University of California, Berkeley
B.S. in Computer Science | GPA: 3.82 | Graduated May 2024
Relevant Coursework: Data Structures & Algorithms, Database Systems, Web Development, Software Engineering

SKILLS
- Languages: Python, JavaScript, TypeScript, SQL, HTML5, CSS3, Go
- Frameworks & Libraries: React, Next.js, Node.js, FastAPI, Flask, Tailwind CSS, Express
- Databases & Tools: PostgreSQL, MongoDB, Redis, Docker, Git, GitHub Actions, Postman
- Cloud & Concepts: AWS (S3, EC2), RESTful APIs, CI/CD, Microservices, Object-Oriented Programming

PROJECTS & EXPERIENCE
Software Engineering Intern | DevPulse Technologies (June 2023 - August 2023)
- Architected and deployed 4 RESTful microservices using Python and FastAPI, handling 15,000+ daily API requests.
- Reduced database query latency by 42% by indexing PostgreSQL tables and implementing Redis caching.
- Collaborated in an agile scrum team of 6 engineers, participating in bi-weekly sprints, PR code reviews, and standups.
- Configured automated GitHub Actions CI/CD pipelines to run unit tests and linting before staging deployments.

Campus Event Portal (Full-Stack Web App) | Personal Project (Jan 2024 - April 2024)
- Developed a full-stack event discovery portal using React, TypeScript, Tailwind CSS, and Node.js with 1,200 active student users.
- Built secure authentication using JWT tokens and OAuth 2.0 with Google Sign-In.
- Containerized frontend and backend services using Docker and orchestrated deployment onto AWS EC2 with an Nginx reverse proxy.
- Implemented real-time RSVP notifications using WebSockets and PostgreSQL pub/sub triggers.
""",
        "job": """Role: Junior Full-Stack Software Engineer
Company: CloudScale Dynamics
Location: San Francisco, CA (Hybrid)

About the Role:
We are seeking a motivated Junior Full-Stack Software Engineer to join our core product team. You will build and scale customer-facing web applications, contribute to high-performance microservices, and collaborate closely with product managers and designers.

Responsibilities:
- Build clean, responsive user interfaces using React, TypeScript, and modern CSS frameworks.
- Design, build, and maintain scalable RESTful APIs using Python (FastAPI or Flask) or Node.js.
- Work with relational databases like PostgreSQL and NoSQL data stores like Redis for caching.
- Write clean, maintainable, well-tested code with unit and integration tests.
- Participate in Agile/Scrum ceremonies, code reviews, and sprint planning.
- Deploy and manage containerized services using Docker and AWS cloud infrastructure.

Required Qualifications:
- Bachelor's degree in Computer Science, Software Engineering, or equivalent practical experience.
- Strong proficiency in Python or TypeScript/JavaScript.
- Experience building full-stack web applications with React and REST APIs.
- Familiarity with PostgreSQL, SQL database schema design, and query optimization.
- Hands-on experience with Git, version control, and GitHub workflows.
- Understanding of Docker containerization and CI/CD pipelines.

Nice to Have:
- Experience with Kubernetes or AWS cloud services (EC2, S3, RDS).
- Familiarity with GraphQL, Next.js, or microservices architecture.
- Demonstrated side projects or contributions to open-source software.
""",
    },
    "Data Science & Machine Learning Intern": {
        "resume": """PRIYA PATEL
New York, NY | priya.patel@email.com | (555) 432-8765 | linkedin.com/in/priyapatel-ds | github.com/priyapatel

EDUCATION
New York University (NYU)
B.S. in Data Science & Applied Mathematics | Minor in Computer Science | Expected Dec 2024
Honors: Dean's List (all semesters) | GPA: 3.88

TECHNICAL SKILLS
- Programming: Python, R, SQL, Bash
- ML & Data Libraries: Scikit-learn, PyTorch, Pandas, NumPy, SciPy, Matplotlib, Seaborn, XGBoost
- NLP & LLMs: HuggingFace Transformers, NLTK, Spacy, TF-IDF, Vector Embeddings
- Big Data & Tools: Apache Spark (PySpark), PostgreSQL, Jupyter Notebooks, Git, Docker, Streamlit

EXPERIENCE & RESEARCH
Data Science Research Assistant | NYU Center for Data Science (Jan 2023 - Present)
- Engineered end-to-end predictive machine learning pipelines in Python using Scikit-Learn and XGBoost on 2.4M customer records.
- Improved classification F1-score from 0.73 to 0.86 through extensive feature engineering, PCA, and hyperparameter tuning.
- Built interactive Streamlit dashboards enabling researchers to visualize statistical distributions and regression models in real-time.
- Performed statistical hypothesis testing (A/B testing, ANOVA, t-tests) to validate experimental drug efficacy results.

Customer Sentiment Analysis Platform | Academic Capstone (Sep 2023 - Dec 2023)
- Implemented natural language processing (NLP) model fine-tuning BERT via HuggingFace for aspect-based sentiment classification.
- Scraped and preprocessed 85,000 customer reviews using Python and regex, removing noise and handling class imbalance.
- Containerized the inference pipeline with Docker and achieved sub-50ms latency per request.
""",
        "job": """Role: Machine Learning / Data Science Intern
Company: QuantMetrics AI
Location: New York, NY

Overview:
QuantMetrics AI is looking for an enthusiastic Machine Learning Intern to assist our applied research team. You will work on real-world datasets, build predictive models, and help deploy AI solutions to production.

Key Responsibilities:
- Clean, analyze, and preprocess structured and unstructured data using Python (Pandas, NumPy).
- Train and evaluate statistical and machine learning models using Scikit-Learn, XGBoost, and PyTorch.
- Work with Natural Language Processing (NLP) techniques, transformers, and text embeddings.
- Formulate and execute A/B testing frameworks and perform rigorous statistical analysis.
- Collaborate with software engineers to package models into production-ready data pipelines and APIs.
- Communicate findings and model metrics through clear data visualizations (Matplotlib, Tableau, or Streamlit).

Requirements:
- Enrolled in or recent graduate of a BS/MS program in Computer Science, Data Science, Statistics, or related STEM field.
- Proficiency in Python and SQL for data analysis and querying.
- Solid theoretical understanding of machine learning algorithms (regression, classification, clustering, ensemble methods).
- Hands-on experience with Scikit-learn, PyTorch or TensorFlow, and Pandas.
- Strong foundation in probability, statistics, and linear algebra.
- Experience using Git for version control.

Preferred:
- Experience with Cloud platforms (AWS SageMaker, GCP BigQuery).
- Experience with MLOps tools (Docker, MLflow) or large language models (LLMs).
""",
    },
    "Frontend Engineer (React / UI)": {
        "resume": """JORDAN LEE
Austin, TX | jordan.lee@email.com | (555) 987-6543 | linkedin.com/in/jordanlee-dev | github.com/jordanlee

SUMMARY
Passionate Frontend Software Engineer with 2+ years of experience building accessible, responsive, and performant web applications using React, TypeScript, and modern CSS.

TECHNICAL SKILLS
- Languages: JavaScript (ES6+), TypeScript, HTML5, CSS3/SCSS
- Frontend Technologies: React, Next.js, Redux Toolkit, Tailwind CSS, Material UI, Webpack, Vite
- Testing & Quality: Jest, React Testing Library, Cypress, Lighthouse, Web Vitals, WCAG Accessibility
- Tools & Workflows: Git, Figma, Storybook, REST APIs, GraphQL, npm/yarn

PROFESSIONAL EXPERIENCE
Frontend Developer | PixelCraft Studio (Aug 2022 - Present)
- Developed responsive web applications for 12+ client brands using React, TypeScript, and Tailwind CSS.
- Boosted Google Lighthouse Performance score from 64 to 98 across core pages by optimizing bundle size and lazy loading.
- Created a reusable component design system in Storybook, decreasing UI development turnaround time by 30%.
- Implemented WCAG 2.1 AA accessibility standards across customer portals, ensuring keyboard navigation and screen-reader support.
- Integrated RESTful APIs and GraphQL queries with client-side caching to deliver snappy, offline-resilient experiences.
""",
        "job": """Role: Frontend Software Engineer
Company: Elevate Digital Solutions
Location: Austin, TX (Hybrid)

Job Description:
We are looking for a creative and detail-oriented Frontend Engineer to craft delightful, fast user experiences. You will collaborate with product designers in Figma and backend engineers to bring new features to life.

Requirements:
- 2+ years of hands-on experience building web applications with React and TypeScript.
- Strong mastery of HTML5, CSS3, responsive design, and CSS frameworks (Tailwind CSS or styled-components).
- Experience integrating REST APIs and state management solutions (Redux, Zustand, or React Query).
- Understanding of web performance optimization (Core Web Vitals, code splitting, asset optimization).
- Familiarity with UI testing using Jest and React Testing Library.
- Passion for clean UI/UX, typography, design systems, and web accessibility standards (WCAG).
- Fluency in Git and modern frontend build tools (Vite, Next.js).
""",
    },
}
