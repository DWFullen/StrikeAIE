# Certification Roadmap, Capstone Project, and 6-Month Execution Plan

Over the next six months, the target certifications are Microsoft Azure AI Fundamentals (AI-900), Azure AI Engineer Associate (AI-102), and Azure Data Scientist Associate (DP-100), acquired in that sequence and each directly tied to a performance tier: AI-900 anchors the Succeeds tier, which is demonstrated by a documented project architecture and a one-pager explaining the design choices; AI-102 anchors the Exceeds tier, which is demonstrated by fully deployed infrastructure and CI/CD pipelines that follow GitFlow best practices; and DP-100 anchors the Exemplary tier, which is demonstrated by a working, publicly showcaseable web application with AI integrated into its core experience; the capstone that encapsulates this knowledge is a secure, full-stack Azure AI application that includes retrieval-augmented generation, prompt/tool orchestration, model evaluation, telemetry, and CI/CD deployment, with months 1–2 focused on AI-900 fundamentals and architecture design, months 3–4 focused on AI-102 service integration and infrastructure/pipeline readiness, and months 5–6 focused on DP-100 model lifecycle and delivering the fully functional AI-powered application.

## 6-Month Timeline

- **Month 1:** Complete AI-900 fundamentals study path and define capstone scope.
- **Month 2:** Pass AI-900 exam, produce project architecture diagrams, and write one-pager explaining design choices.
- **Month 3:** Complete AI-102 training and provision infrastructure (networking, storage, AI services) via IaC.
- **Month 4:** Pass AI-102 exam, wire up CI/CD pipelines following GitFlow, and verify end-to-end deployment.
- **Month 5:** Complete DP-100 training and integrate AI features into the running web application.
- **Month 6:** Pass DP-100 exam, finalize the AI-powered web app, and publish a polished showcase package.

## Performance Tiers

### Succeeds — *Microsoft Certified: Azure AI Fundamentals (AI-900)*
- Complete the AI-900 training path and pass the AI-900 exam.
- Produce a project architecture document covering components, data flows, security boundaries, and Azure service selections.
- Write a one-pager that explains and justifies each major architectural decision (technology choices, trade-offs considered, and alternatives ruled out).
- Validate the architecture against baseline security controls (secrets handling, access control, and input validation).

### Exceeds — *Microsoft Certified: Azure AI Engineer Associate (AI-102)*
- Pass both the AI-900 and AI-102 exams within six months.
- Deploy all infrastructure defined in the Succeeds-tier architecture using Infrastructure-as-Code (e.g., Bicep or Terraform), with no manual portal steps.
- Establish CI/CD pipelines that follow GitFlow (feature branches, pull request reviews, protected main/release branches, and semantic versioning) and automate build, test, and deployment stages.
- Demonstrate a passing pipeline run end-to-end: code pushed to a feature branch, PR merged, and the change automatically promoted through dev → staging → production environments.

### Exemplary — *Microsoft Certified: Azure Data Scientist Associate (DP-100)*
- Pass all three certifications (AI-900, AI-102, DP-100) on first attempt and complete ahead of the six-month target.
- Deliver a working, publicly accessible web application that has AI meaningfully integrated into the user experience (e.g., natural-language queries, generative responses, intelligent recommendations, or AI-powered search).
- The application must run on the infrastructure and pipelines from the Exceeds tier, demonstrating the full end-to-end stack from commit to live deployment.
- Produce a showcase package (live demo, architecture walkthrough, and a brief explanation of how AI is used and why it adds value).
