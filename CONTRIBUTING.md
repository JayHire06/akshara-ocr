# Contributing to Akshara OCR

Welcome to the Akshara OCR project. To maintain order and a clear history, we enforce a strict branching strategy. The Project Lead AI agent (Zayed) oversees all integration into the main codebase.

## Branch Strategy

The core repository follows a Feature Branch Workflow for all 11 Agents.

### 1. Creating Feature Branches
Each agent or contributor must work in their assigned scope using an isolated branch derived from `main`.

**Current Agent Branches:**
- `feature/preprocessor-jay`: Image Processing Pipeline
- `feature/model-tanmay`: Deep Learning Core OCR Architectures
- `feature/nlp-hindi`: NLP Post-processing & Hindu dictionaries
- `feature/nlp-bengali`: NLP Post-processing (Bengali phonetic rules)
- `feature/nlp-tamil`: NLP Post-processing (Tamil phonetic rules)
- `feature/frontend-design`: React SPA UI & Design System Components
- `feature/backend-api`: FastAPI routing, validation, & Celery configuration
- `feature/security-auth`: JWT Authorization & User file isolation tracking
- `feature/infra-cicd`: Docker setup, AWS provisioning, and GitHub CI actions
- `feature/data-pipeline`: Synthetic generation and dataset version control (`DVC`)
- `feature/testing-qa`: Global testing utilities, benchmarking, and fixtures

### 2. Committing Constraints Walkthrough
When you are ready to push work back to the repository:

1. **Commit your changes locally** on your branch (`feature/...`).
2. **Push to Origin:** `git push origin <your-feature-branch>`
3. **Open a Pull Request:** Navigate to GitHub and open a PR against the `main` branch.

### 3. Branch Protection Policy
The `main` branch is protected.
- **No Direct Commits:** You cannot push directly to `main`.
- **Review Requirement:** Every Pull Request requires **1 Approval** before it can be merged.
- **Approver Role:** The Project Lead AI Agent **(Zayed)** acts as the sole approver. PRs will be evaluated for Interface Contract Compliance (see `docs/interfaces.py`), Type Hinting completeness, and test integrations.

*Note: For the current sprint phase, contributors may operate directly on `main` to move quickly. However, this policy will be strictly implemented for all future sprints.*
