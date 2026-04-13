/**
 * Local History Manager
 * Manages OCR results using Browser LocalStorage to ensure a completely offline experience.
 */

const STORAGE_KEY = 'akshara_ocr_history';

export const localHistory = {
  /**
   * Save a result to local storage
   */
  async saveResult(result, file, language) {
    try {
      const history = this.getHistory();
      
      const newEntry = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        text: result.text,
        language: language,
        processingTimeMs: result.processingTimeMs,
        fileName: file.name,
        // We don't store the full image blob in localStorage as it exceeds 5MB limits
        // In a real mobile app, we would store it in the filesystem via Capacitor
      };

      history.unshift(newEntry);
      
      // Keep only last 50 entries to avoid bloating localStorage
      const trimmedHistory = history.slice(0, 50);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmedHistory));
      
      return newEntry;
    } catch (e) {
      console.error("Failed to save local history:", e);
      return null;
    }
  },

  /**
   * Get all history entries
   */
  getHistory() {
    try {
      const history = localStorage.getItem(STORAGE_KEY);
      return history ? JSON.parse(history) : [];
    } catch (e) {
      console.error("Failed to read local history:", e);
      return [];
    }
  },

  /**
   * Clear all history
   */
  clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
  }
};
