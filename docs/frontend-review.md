# Akshara OCR - Frontend Design Review

**Reviewer:** Prince (UI/UX Design AI)
**Developer:** John (Frontend Engineering AI)
**Reviewed Date:** 2026-03-03

Here is a review of the current frontend implementation running at `http://127.0.0.1:5173/` against the established design tokens and handoff document.

## 1. Styling & Brand Consistency

*   **Color Palette:**
    *   ✅ **Primary Accent:** The primary accent color correctly matches the design token (`#c6f135`).
    *   ❌ **Background/Surface Color:** The background color is slightly off. It appears as `#131313` (`rgb(19,19,19)`) instead of the requested deep slate `#0f1115`.
*   **Typography:**
    *   ✅ **Heading/Display Font:** Correctly uses **Outfit**, providing a distinctive, modern look.
    *   ❌ **Body Font:** The body text is currently rendered in **Inter** instead of the intended **Manrope** font specified in the design handoff.
*   **Spacing & Interactions:**
    *   ✅ The implementation successfully follows an 8px grid system (e.g., 8px/16px/24px padding), and components exhibit smooth 0.2s hover transitions.

## 2. Screen-by-Screen Evaluation

*   **Landing Screen:**
    *   Features the logo, CTA, and feature cards.
    *   *Note:* The tagline ("Preserve your heritage...") differs slightly from the requested "Extract Text from Any Indian Script Instantly".
*   **Upload Screen:**
    *   Includes a large drag-and-drop zone and document type selectors.
    *   🔴 **Critical Issue:** The language grid is stuck on "Loading languages...", likely due to the backend service at `http://127.0.0.1:8000/languages` being unresponsive or failing. This blocks the main workflow.
    *   *Requirement Gap:* Only 2 document types (Printed, Handwritten) are implemented, whereas 4 were required in the specs.
*   **History & Settings Screens:**
    *   Both screens are implemented. Settings includes preference management.
    *   History is currently empty and shows a loading state.
*   **Error Screen / Routing:**
    *   Currently, invalid routes (like `/invalid`) redirect to the landing page instead of showing the dedicated user-friendly error screen.

## 3. Responsive Design

*   The application is highly functional at mobile viewports (e.g., 375px), adapting the layout well and maintaining the vertical flow of the upload and settings sections.
*   Navigation buttons in the header become slightly cramped on very small screens and could use a hamburger menu or refined spacing.

## Recommended Action Items for John

1.  **Fix Critical Blocker:** Investigate the `/languages` API endpoint to ensure the frontend can fetch and display the language grid. The core OCR flow is blocked without it.
2.  **Typography Update:** Switch the body `font-family` CSS variable from `'Inter', sans-serif` to `'Manrope', sans-serif`.
3.  **Background Color:** Adjust the CSS variable for the main background from `#131313` to exactly `#0f1115`.
4.  **Expand Options:** Add the remaining 2 document types to the selector and ensure all 11 languages populate in the selection grid once the API is fixed.
5.  **Tagline:** Align the landing page hero text with the marketing requirements: "Extract Text from Any Indian Script Instantly".
6.  **Implement Error Screen:** Create the dedicated `/error` route and generic 404 page as per the design specs instead of defaulting to a redirect.
