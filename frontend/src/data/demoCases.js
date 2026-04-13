export const demoCases = [
  {
    id: 'hindi-poster',
    title: 'Devanagari Poster',
    script: 'Devanagari',
    languageCode: 'hin',
    languageLabel: 'Hindi',
    assetPath: '/demo/hindi-poster.png',
    expectedText: 'नमस्ते दुनिया\nअक्षर ओसीआर परीक्षण',
    note: 'Conjunct-heavy greeting. Stresses क्ष, स्त, ण — the hardest classes for the v8 handwritten stage.',
    difficulty: 'Hard',
    testsFor: 'Shirorekha + conjuncts'
  },
  {
    id: 'hindi-proverb',
    title: 'Hindi Proverb',
    script: 'Devanagari',
    languageCode: 'hin',
    languageLabel: 'Hindi',
    assetPath: '/demo/hindi-proverb.png',
    expectedText: 'कर्म ही पूजा है\nएक पुरानी हिंदी कहावत',
    note: 'Display type in Eczar with half-form र्म and long ee matra. Clean printed baseline check.',
    difficulty: 'Medium',
    testsFor: 'Half-form + long matra'
  },
  {
    id: 'hindi-shopping-list',
    title: 'बाज़ार की सूची',
    script: 'Devanagari',
    languageCode: 'hin',
    languageLabel: 'Hindi',
    assetPath: '/demo/hindi-shopping-list.png',
    expectedText: 'बाज़ार की सूची\n1. चावल — २ किलो\n2. दूध — १ लीटर\n3. हल्दी और नमक\n4. हरी मिर्च',
    note: 'Multi-line list with mixed Devanagari numerals (१ २) and Latin digits. Exercises the line segmenter fix.',
    difficulty: 'Hard',
    testsFor: 'Line segmentation + mixed numerals'
  },
  {
    id: 'hindi-handwritten-note',
    title: 'प्रिय मित्र',
    script: 'Devanagari',
    languageCode: 'hin',
    languageLabel: 'Hindi',
    assetPath: '/demo/hindi-handwritten-note.png',
    expectedText: 'प्रिय मित्र\nआशा है तुम कुशल हो।\nजल्दी मिलते हैं।',
    note: 'Casual Kalam rendering — closer to the handwritten distribution in v8 Stage 3 (IIIT-HW-WORDS).',
    difficulty: 'Hard',
    testsFor: 'Handwritten-style + punctuation (।)'
  }
];
