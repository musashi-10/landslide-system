/**
 * Landing page — public entry point for the Landslide Early Warning System.
 */

import { Link } from 'react-router-dom';
import {
  Mountain,
  Radar,
  CloudRain,
  Satellite,
  Brain,
  Bell,
  Shield,
  ChevronRight,
  ArrowRight,
  MapPin,
} from 'lucide-react';

export function Landing() {
  return (
    <div className="landing">
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero__bg-pattern" aria-hidden="true" />
        <div className="hero__content">
          <div className="hero__badge">
            <Radar size={14} />
            AI-POWERED EARLY WARNING
          </div>
          <h1 className="hero__title">
            Landslide Early
            <span className="hero__title-accent"> Warning System</span>
          </h1>
          <p className="hero__subtitle">
            Predict. Monitor. Warn.
          </p>
          <p className="hero__description">
            Location-specific landslide risk monitoring using terrain analysis,
            rainfall data, satellite imagery, and AI-based risk prediction.
          </p>
          <div className="hero__actions">
            <Link to="/register" className="btn btn--primary btn--lg">
              <MapPin size={18} />
              Check Risk
              <ArrowRight size={16} />
            </Link>
            <Link to="/login" className="btn btn--outline btn--lg">
              Sign In
            </Link>
          </div>
          <div className="hero__authority-link">
            <Shield size={14} />
            <Link to="/login" className="hero__auth-link">
              Authority Login →
            </Link>
          </div>
        </div>
      </section>

      {/* ── Pipeline ─────────────────────────────────────────────────── */}
      <section className="pipeline-section">
        <h2 className="section-title">How It Works</h2>
        <p className="section-subtitle">
          Multi-source data analysis powering real-time risk predictions
        </p>
        <div className="pipeline">
          <div className="pipeline__step">
            <div className="pipeline__icon-wrap pipeline__icon-wrap--terrain">
              <Mountain size={28} />
            </div>
            <h3>Terrain Analysis</h3>
            <p>GIS elevation, slope, and geological data</p>
          </div>
          <ChevronRight className="pipeline__arrow" size={24} />
          <div className="pipeline__step">
            <div className="pipeline__icon-wrap pipeline__icon-wrap--satellite">
              <Satellite size={28} />
            </div>
            <h3>Satellite Imagery</h3>
            <p>Land cover and change detection</p>
          </div>
          <ChevronRight className="pipeline__arrow" size={24} />
          <div className="pipeline__step">
            <div className="pipeline__icon-wrap pipeline__icon-wrap--rainfall">
              <CloudRain size={28} />
            </div>
            <h3>Rainfall Data</h3>
            <p>Real-time and forecast precipitation</p>
          </div>
          <ChevronRight className="pipeline__arrow" size={24} />
          <div className="pipeline__step">
            <div className="pipeline__icon-wrap pipeline__icon-wrap--ai">
              <Brain size={28} />
            </div>
            <h3>AI Risk Engine</h3>
            <p>Machine learning risk prediction</p>
          </div>
          <ChevronRight className="pipeline__arrow" size={24} />
          <div className="pipeline__step">
            <div className="pipeline__icon-wrap pipeline__icon-wrap--alert">
              <Bell size={28} />
            </div>
            <h3>Early Warning</h3>
            <p>SMS alerts and risk notifications</p>
          </div>
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────── */}
      <section className="features-section">
        <div className="features-grid">
          <div className="feature-card">
            <MapPin size={24} className="feature-card__icon" />
            <h3>Location-Specific</h3>
            <p>Monitor your exact location with precision risk analysis tailored to local terrain and conditions.</p>
          </div>
          <div className="feature-card">
            <Radar size={24} className="feature-card__icon" />
            <h3>Real-Time Monitoring</h3>
            <p>Continuous risk assessment updated with the latest rainfall, satellite, and environmental data.</p>
          </div>
          <div className="feature-card">
            <Bell size={24} className="feature-card__icon" />
            <h3>Instant Alerts</h3>
            <p>SMS and in-app warnings when risk levels reach HIGH or CRITICAL thresholds.</p>
          </div>
          <div className="feature-card">
            <Shield size={24} className="feature-card__icon" />
            <h3>Authority Dashboard</h3>
            <p>Regional command center for disaster management authorities with multi-location monitoring.</p>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer className="landing-footer">
        <p>Landslide Early Warning System — AI-powered risk prediction for safer communities.</p>
        <p className="landing-footer__disclaimer">
          Risk predictions are model-based estimates. They do not guarantee landslide occurrence.
          Always follow official guidance from local authorities.
        </p>
      </footer>
    </div>
  );
}
