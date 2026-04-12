export const demoCases = [
  {
    id: 'hello-world',
    title: 'English Signboard',
    script: 'Latin',
    languageCode: 'eng',
    languageLabel: 'English',
    assetPath: '/demo/hello-world.png',
    expectedText: 'Hello World',
    note: 'A tiny clean sample for checking the full upload to result loop.'
  },
  {
    id: 'hindi-poster',
    title: 'Hindi Poster',
    script: 'Devanagari',
    languageCode: 'hin',
    languageLabel: 'Hindi',
    assetPath: '/demo/hindi-poster.png',
    expectedText: 'नमस्ते दुनिया\nअक्षर ओसीआर परीक्षण',
    note: 'Bundled Devanagari text so visitors can preview the kind of script the project targets.'
  },
  {
    id: 'travel-list',
    title: 'Travel Notes',
    script: 'Latin',
    languageCode: 'eng',
    languageLabel: 'English',
    assetPath: '/demo/travel-list.png',
    expectedText: 'Double Decker Bridge\nDawki Lake\nLaitlum Canyons\nWei Sawdong Falls',
    note: 'A denser multi-line case that feels closer to a document snippet.'
  }
];
