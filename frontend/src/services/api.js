const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const setAuthToken = (token) => {
  if (token) {
    localStorage.setItem('access_token', token);
  } else {
    localStorage.removeItem('access_token');
  }
};

const getHeaders = (isFormData = false) => {
  const headers = {};
  const token = localStorage.getItem('access_token');
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  return headers;
};

export const api = {
  login: async (username, password) => {
    // Backend uses OAuth2PasswordRequestForm which requires x-www-form-urlencoded
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString()
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    return response.json();
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
    const token = localStorage.getItem('access_token');
    if (!token) {
      localStorage.removeItem('access_token');
      window.location.href = '/';
      throw new Error('Not authenticated');
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', languageCode);

    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      headers: getHeaders(true),
      body: formData
    });

    if (response.status === 401 || response.status === 403) {
      localStorage.removeItem('access_token');
      window.location.href = '/';
      throw new Error('Session expired');
    }

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
