import { Link } from 'react-router-dom';
import MetricCard from '../components/MetricCard';

const sections = [
  {
    to: '/eda',
    label: 'Data Insights',
    value: 'Explore Patterns',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)' }}>
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <path d="M3 9h18M9 21V9" />
      </svg>
    ),
  },
  {
    to: '/models',
    label: 'Model Bench',
    value: 'Compare Classifiers',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)' }}>
        <path d="M21.21 15.89A10 10 0 1 1 8 2.83M22 12A10 10 0 0 0 12 2v10z" />
      </svg>
    ),
  },
  {
    to: '/explainability',
    label: 'Interpret AI',
    value: 'SHAP & LIME',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)' }}>
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" />
      </svg>
    ),
  },
  {
    to: '/bias',
    label: 'Fairness Lab',
    value: 'Audit Subgroups',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-primary)' }}>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      </svg>
    ),
  },
];

export default function Home() {
  return (
    <div>
      <div className="hero">
        <h1>Student Success Platform</h1>
        <p>
          Explainable AI and Bias Auditing in Education. Predict student outcomes
          with transparent, interpretable machine learning models.
        </p>
      </div>

      <div className="grid-4 section">
        {sections.map((s) => (
          <Link to={s.to} key={s.to} style={{ textDecoration: 'none' }}>
            <MetricCard icon={s.icon} label={s.label} value={s.value} />
          </Link>
        ))}
      </div>

      <div className="glass-card">
        <h3 style={{ marginBottom: 16 }}>Platform Overview</h3>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: 1.8, fontSize: 'var(--font-base)' }}>
          The Student Success Platform leverages state-of-the-art Machine Learning models to analyze academic and socioeconomic indicators.
          It provides transparent, explainable predictions to help educators identify at-risk students early and intervene effectively.
        </p>
      </div>
    </div>
  );
}
