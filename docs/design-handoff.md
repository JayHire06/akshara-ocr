# Akshara OCR - Design Handoff

**Designer:** Prince (UI/UX Design AI)  
**Developer:** John (Frontend Engineering AI)  
**Project:** Akshara OCR  

## 1. Design Tokens

### Colors
| Token Name | Hex Code | Usage |
|---|---|---|
| Background | `#0f1115` | App background, lowest z-index elements |
| Surface | `#181b21` | Modals, off-canvas menus, sidebars |
| Card | `#22262f` | Content cards, data tables, result displays |
| Border | `#333946` | Dividers, input borders, structural boundaries |
| Primary/Accent | `#c6f135` | Main CTAs, active states, progress indicators |
| Text Main | `#f3f4f6` | Primary reading text, headings |
| Text Muted | `#9ca3af` | Helper text, secondary labels, placeholders |
| Success | `#10b981` | Completed states, high confidence badges |
| Error | `#ef4444` | Invalid inputs, error alerts, failed states |
| Warning | `#f59e0b` | Low confidence badges, advisory alerts |

### Typography
**Font Families:**
- **Display/Heading:** `Outfit` (Modern, geometric, distinctive Sans-serif from Google Fonts)
- **Body:** `Manrope` (Clean, highly legible Sans-serif)
- **Mono/OCR Result:** `Noto Sans Devanagari` (Excellent Indian script support)

**Font Sizes:**
- `xs` (11px) - Badges, tiny helper text
- `sm` (12px) - Secondary labels, timestamps
- `base` (14px) - Default body text, button labels
- `md` (16px) - Input text, list item titles
- `lg` (20px) - Card headings, modal titles
- `xl` (28px) - Section headers, empty state titles
- `2xl` (36px) - Landing page hero text

**Weights & Line Heights:**
- Weights: Regular (400), Medium (500), SemiBold (600), Bold (700)
- Line Heights: Tight (1.2 - Headings), Normal (1.5 - Body text), Relaxed (1.75 - Long reading)

### Spacing (8px Grid)
- `4px` - Micro-spacing inside components
- `8px` - Between tight related items (icon + text)
- `12px` - Internal padding for small inputs/buttons
- `16px` - Default component padding, list item gaps
- `24px` - Card padding, section block gaps
- `32px` - Major component grouping gap
- `48px` - Layout structural gap
- `64px` - Section to section gap on desktop

---

## 2. Component Anatomy & States

### 2.1 Button
**Tokens Used:** `typography.base`, `colors.primary`, `colors.card`, `colors.border`, `spacing.12` (y), `spacing.24` (x)
- **Primary:**
  - *Default:* BG `Primary`, Text `#0f1115`, Font Weight 600
  - *Hover:* Brightness 110%
  - *Focus:* Outline 2px `Primary` + 2px offset
  - *Active:* Scale 0.98, Opacity 90%
  - *Disabled:* BG `Border`, Text `Text Muted`, Cursor not-allowed
  - *Error:* Shake animation, BG `Error`, Text `#fff`
- **Secondary:**
  - *Default:* BG `Card`, Text `Text Main`, Border 1px `Border`
  - *Hover:* BG `Border`
  - *Focus:* Outline 2px `Text Muted` + 2px offset
  - *Active:* Scale 0.98
  - *Disabled:* Opacity 50%, Cursor not-allowed
  - *Error:* Border `Error`
- **Ghost:**
  - *Default:* BG Transparent, Text `Text Main`
  - *Hover:* BG `Surface`
  - *Focus:* Outline 2px `Border`
  - *Active:* BG `Border`
  - *Disabled:* Opacity 50%
  - *Error:* Text `Error`, BG Error 10% opacity

### 2.2 Input Field (Text & File Upload Zone)
**Tokens Used:** `typography.md` (input), `typography.sm` (label), `colors.border`, `colors.card`, `colors.primary`, `spacing.32` (upload)
- **Text Input:**
  - *Default:* BG `Card`, Border 1px `Border`, Text `Text Main`, Placeholder `Text Muted`
  - *Hover:* Border `Text Muted`
  - *Focus:* Border `Primary`, Box-shadow 0 0 0 1px `Primary`
  - *Active:* (Same as Focus)
  - *Disabled:* BG `Surface`, Text `Text Muted`, Opacity 60%
  - *Error:* Border `Error`, Box-shadow 0 0 0 1px `Error`
- **File Upload Zone:**
  - *Default:* BG `Surface`, Border 2px dashed `Border`, Icon/Text `Text Muted`, Height 200px
  - *Hover:* BG `Card`, Border `Text Muted`
  - *Focus:* Outline 2px `Primary` + offset
  - *Active/Drag-over:* BG `Primary` (10% opacity), Border 2px dashed `Primary`, Icon/Text `Primary`
  - *Disabled:* Opacity 50%, dashed `Border` only
  - *Error:* Border 2px dashed `Error`, Text `Error`

### 2.3 Language Chip / Selector
**Tokens Used:** `typography.sm`, `colors.surface`, `colors.primary`
- *Default:* BG `Surface`, Text `Text Main`, Padding 8px 16px, Pill Radius
- *Hover:* BG `Card`, border 1px `Border`
- *Focus:* Outline 2px `Primary` + offset
- *Active/Selected:* BG `Primary` (15% opacity), Border 1px `Primary`, Text `Primary`
- *Disabled:* BG `Surface`, Text `Text Muted`, Opacity 50%
- *Error:* Red outline, Text `Error`

### 2.4 Progress Bar
**Tokens Used:** `colors.primary`, `colors.surface`, `spacing.8` (height)
- *Default (Track):* BG `Surface`, Height 8px, Pill radius
- *Active (Indicator):* BG `Primary`, width dynamic, Transition linearly
- *Hover:* Tooltip showing exact %
- *Focus:* None (non-interactive unless paused/canceled)
- *Disabled:* Track BG `Border`, Indicator `Border`
- *Error:* Indicator becomes BG `Error`

### 2.5 Confidence Badge
**Tokens Used:** `typography.xs`, `colors.success`/`warning`/`error`, `spacing.4` (padding)
- *Default (Medium 70-90%):* BG Warning 15%, Text `Warning`
- *Hover:* Slight brightness increase
- *Focus:* Outline 1px matching category color
- *Active:* None 
- *Disabled:* BG `Surface`, Text `Text Muted`
- *Error (Low <70%):* BG Error 15%, Text `Error`
- *Success (High >90%):* BG Success 15%, Text `Success`

### 2.6 Result Text Display Card
**Tokens Used:** `typography.mono` (Noto Sans Dev), `colors.card`, `colors.text`, `spacing.24`
- *Default:* BG `Card`, Border 1px `Border`, Radius 8px, Padding 24px, Overflow-Y Auto
- *Hover:* Border slightly lighter (`Text Muted`)
- *Focus:* Outline 2px `Primary` (for a11y keyboard scrolling)
- *Active:* Standard text selection highlighting (`Primary` BG, `Text Main`)
- *Disabled:* Opacity 60%, unselectable
- *Error:* Border `Error`, contains error placeholder text

### 2.7 Navigation Bar
**Tokens Used:** `colors.background`, `spacing.16`, `colors.border`
- *Default:* BG `Background`, Border-bottom 1px `Border`, Padding-y 16px
- *Hover/Focus/Active (Links inside):* See Ghost Button states
- *Disabled:* (N/A for container)
- *Error:* (N/A for container)

### 2.8 Toast / Notification
**Tokens Used:** `colors.card`, `colors.success` or `colors.error`, `spacing.16`
- *Default:* BG `Card`, Border-left 4px indicator color (`Success` info, `Error` alert), Padding 16px, Shadow lg
- *Hover:* Pause hide timer
- *Focus:* (If actionable item inside)
- *Active:* Click to dismiss (scale down, fade out)
- *Disabled:* (N/A)
- *Error:* Notification type strictly utilizing `Error` border + icon

---

## 3. Screen Designs & Interactions

### 1. Landing
- **Layout:** Centered hero section, followed by a 3-column features grid.
- **Components:**
  - Header: Logo (left), "Extract Text from Any Indian Script Instantly" `2xl` heading.
  - Subtitle: `lg` text in `Text Muted`.
  - Hero CTA: Primary Button ("Upload Image") - Click goes to Upload.
  - Features Grid: 3 `Card` components with 24px padding, icon, `lg` heading, `base` text.

### 2. Upload
- **Layout:** Contained card layout max-width 800px.
- **Components:**
  - Drag-and-drop zone (48px padding, dashed border), File Icon and instruction text.
  - Sub-section: Document Type Selector (4 options, Radio or Ghost Buttons).
  - Sub-section: Language Grid (11 languages, Grid layout of Language Chips in native script).
  - Action Footer: Primary Button ("Extract Text").
- **Interaction:** Drag over Upload Zone triggers `Active/Drag-over` state. File drop shows tiny preview + success icon.

### 3. Language Select (Modal)
- **Layout:** Overlay modal centered on screen with backdrop blur.
- **Components:**
  - Search Input: Top of modal. Type to filter list.
  - List/Grid: 11 Languages. Each item shows Native Script Name (`md`), English Name (`sm`, `Text Muted`), Script Family label (Badge `xs`).
  - Actions: "Apply" (Primary) / "Cancel" (Ghost).
- **Interaction:** Clicking a language triggers `Selected` state. Checkmark icon appears.

### 4. Processing
- **Layout:** Centered absolute layout across full screen.
- **Components:**
  - Centered Spinner: Custom SVG spinner in `Primary` color animating rotation.
  - Message Text: `lg` heading "Analyzing Document...".
  - Subtext: `base` text showing "Language: Hindi | Est. time: 4s remaining".
  - Progress Bar component below the text.
- **Interaction:** Auto-transitions to Results or Error screen when complete.

### 5. Results
- **Layout:** Two-column split layout (50/50 on desktop). Left = Image, Right = Text.
- **Components:**
  - **Left Col (Image preview):** Fitted image inside a `Card` container.
  - **Right Col (Extracted Text):** Scrollable `Result Text Display Card` using `Noto Sans Devanagari`.
  - **Bottom Row (Stats & Actions):**
    - Stats: Words, Chars, Lines in `Text Muted` `sm` text.
    - Actions: Ghost Button (New Upload), Secondary Button (Download), Primary Button (Copy).
- **Interaction:** 'Copy' button changes to "Copied!" and fires Success Toast.

### 6. History
- **Layout:** Full width list/table with max-width 1000px.
- **Components:**
  - Page Title: "Recent Scans" `xl`.
  - List View: Rows using `Card` styling with 16px padding.
  - Columns/Data: Date (`sm`), Language Badge, Text Snippet (`base`, Noto Sans Dev, truncated), Confidence Badge, Ghost Button (View).
- **Interaction:** Hovering a row changes Background to `Border` color. Click row goes to past Result.

### 7. Settings
- **Layout:** Left sidebar navigation (25%), right content area (75%).
- **Components:**
  - Sections: Language Preference (Dropdown), Font Size for results (Slider/Buttons: Default `base`, up to `xl`), Account Details.
  - Layout Blocks: standard `Input Field` and `Button` controls.
- **Interaction:** Sliders update live preview of OCR text. Changes auto-save with a Success Toast.

### 8. Error
- **Layout:** Centered empty-state layout.
- **Components:**
  - Large Icon: Alert/Warning icon (48x48) in `Error` color.
  - Heading: "We couldn't read that document" `xl` text.
  - Message: `base` plain language explanation ("The image was too blurry or the script is unsupported.").
  - Actions: Primary Button ("Try Again"), Ghost Button ("Help").
- **Interaction:** "Try Again" routes back to Upload screen with previous settings filled.

---

## 4. Responsive Notes
- **Mobile (375px):** 
  - Padding generally 16px screen edges.
  - 1-column layout everywhere. Features (Landing), Upload sections, Results split-view all stack vertically.
  - Result view: Image thumbnail top, tall text block bottom.
  - Touch targets minimum 44px height for interactive elements.
  - Modals become full-screen slide-up drawers.
- **Tablet (768px):** 
  - Padding 24px screen edges.
  - 2-column mixed grids enabled. Settings sidebar can exist.
- **Desktop (1440px):** 
  - Max container width 1200px for central readable constraints.
  - Main section paddings 64px.
  - True 50/50 split on Results screen. Hover states prominent and clearly distinguished.

## 5. Assets Needed (John's Checklist)
- **Icons (`Lucide` or `Phosphor` - Line style recommended):**
  - Upload / File Plus, Image, File Text
  - CheckCircle (Success), AlertCircle/Warning (Error/Warning)
  - Copy, Download, Refresh/Retry
  - Magnifying Glass, Settings/Gear, User/Account
  - X/Close, Chevron Down
- **Images/Illustrations:**
  - SVG Logo for Akshara OCR (Lime green motif on transparent background).
  - Empty state illustrations for History (document stack outline) and Error screens (broken file outline), lightly styled with `Primary` (#c6f135) and `Border` (#333946) tones.
