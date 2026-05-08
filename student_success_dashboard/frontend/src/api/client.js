const BASE = '/api';

async function request(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API Error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  // EDA
  getEdaSummary: () => request('/eda/summary'),
  getEdaPreview: () => request('/eda/preview'),
  getEdaDistribution: () => request('/eda/distribution'),
  getEdaBoxplot: () => request('/eda/boxplot'),
  getEdaCorrelation: () => request('/eda/correlation'),

  // Models
  getModelEvaluation: () => request('/models/evaluate'),

  // Explainability
  getShapGlobal: (modelName = 'XGBoost') =>
    request(`/xai/shap/global?model_name=${modelName}`),
  getShapLocal: (studentIndex, modelName = 'XGBoost') =>
    request('/xai/shap/local', {
      method: 'POST',
      body: JSON.stringify({ student_index: studentIndex, model_name: modelName }),
    }),
  getLimeLocal: (studentIndex, modelName = 'XGBoost') =>
    request('/xai/lime/local', {
      method: 'POST',
      body: JSON.stringify({ student_index: studentIndex, model_name: modelName }),
    }),

  // Bias
  getBiasAudit: () => request('/bias/audit'),

  // Predict
  predict: (data) =>
    request('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};
