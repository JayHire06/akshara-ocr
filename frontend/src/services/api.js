const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

let authToken = localStorage.getItem('access_token') || null;

export const setAuthToken = (token) => {
  authToken = token;
  if (token) {
    localStorage.setItem('access_token', token);
  } else {
    localStorage.removeItem('access_token');
  }
};

const getHeaders = (isFormData = false) => {
  const headers = {};
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  return headers;
};

export const api = {
  login: async (username, password) => {
    // Basic mock or real implementation depending on backend
    // Assuming backend takes regular JSON or Form Data
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData
    });
    if (!response.ok) throw new Error('Login failed');
    return response.json(); // {access_token, refresh_token}
  },

  register: async (username, password) => {
    const response = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Registration failed');
    return data;
  },

  getLanguages: async () => {
    const response = await fetch(`${API_URL}/languages`, {
      headers: getHeaders()
    });
    if (!response.ok) throw new Error('Failed to fetch languages');
    return response.json(); // [{code, name, native_name}]
  },

  uploadDocument: async (file, languageCode) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', languageCode);

    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      headers: getHeaders(true),
      body: formData
    });
    if (!response.ok) throw new Error('Upload failed');
    return response.json(); // {job_id}
  },

  pollResult: async (jobId) => {
    const response = await fetch(`${API_URL}/result/${jobId}`, {
      headers: getHeaders()
    });
    if (!response.ok) throw new Error('Failed to fetch result');
    return response.json();
  },

  // Assuming a history endpoint exists based on screen requirements
  getHistory: async (page = 1) => {
    const response = await fetch(`${API_URL}/history?page=${page}`, {
      headers: getHeaders()
    });
    if (!response.ok) throw new Error('Failed to fetch history');
    return response.json();
  }
};
