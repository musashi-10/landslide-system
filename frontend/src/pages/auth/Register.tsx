/**
 * Register page — citizen user registration.
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { Mountain, Mail, Lock, User, Phone, MapPin, UserPlus, AlertTriangle } from 'lucide-react';

export function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: '',
    email: '',
    mobile: '',
    password: '',
    confirmPassword: '',
    location_id: 'LOC_001',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const update = (field: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (form.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    try {
      await register({
        name: form.name,
        email: form.email,
        mobile: form.mobile,
        password: form.password,
        location_id: form.location_id,
      });
      navigate('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card auth-card--wide">
        <div className="auth-card__header">
          <Mountain size={32} className="auth-card__logo" />
          <h1 className="auth-card__title">Create Account</h1>
          <p className="auth-card__subtitle">
            Register to monitor landslide risk for your location
          </p>
        </div>

        {error && (
          <div className="auth-error" role="alert">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="reg-name" className="form-label">
                <User size={14} /> Full Name
              </label>
              <input id="reg-name" type="text" className="form-input" value={form.name} onChange={update('name')} placeholder="Your full name" required />
            </div>
            <div className="form-group">
              <label htmlFor="reg-email" className="form-label">
                <Mail size={14} /> Email
              </label>
              <input id="reg-email" type="email" className="form-input" value={form.email} onChange={update('email')} placeholder="you@example.com" required autoComplete="email" />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="reg-mobile" className="form-label">
              <Phone size={14} /> Mobile Number
            </label>
            <input id="reg-mobile" type="tel" className="form-input" value={form.mobile} onChange={update('mobile')} placeholder="+91-9876543210" required />
          </div>

          <div className="form-group">
            <label htmlFor="reg-location" className="form-label">
              <MapPin size={14} /> Monitored Location
            </label>
            <select id="reg-location" className="form-input" value={form.location_id} onChange={update('location_id')}>
              <option value="LOC_001">LOC_001 — Darjeeling North</option>
              <option value="LOC_002">LOC_002 — Darjeeling South</option>
              <option value="LOC_003">LOC_003 — Kalimpong Ridge</option>
              <option value="LOC_004">LOC_004 — Mirik Valley</option>
              <option value="LOC_005">LOC_005 — Kurseong Hills</option>
              <option value="LOC_006">LOC_006 — Teesta Valley</option>
              <option value="LOC_007">LOC_007 — Rangeet Basin</option>
              <option value="LOC_008">LOC_008 — Siliguri Foothills</option>
            </select>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="reg-password" className="form-label">
                <Lock size={14} /> Password
              </label>
              <input id="reg-password" type="password" className="form-input" value={form.password} onChange={update('password')} placeholder="Min. 6 characters" required autoComplete="new-password" />
            </div>
            <div className="form-group">
              <label htmlFor="reg-confirm" className="form-label">
                <Lock size={14} /> Confirm Password
              </label>
              <input id="reg-confirm" type="password" className="form-input" value={form.confirmPassword} onChange={update('confirmPassword')} placeholder="Re-enter password" required autoComplete="new-password" />
            </div>
          </div>

          <button type="submit" className="btn btn--primary btn--full" disabled={loading}>
            {loading ? (
              <span className="btn-spinner" />
            ) : (
              <>
                <UserPlus size={16} /> Create Account
              </>
            )}
          </button>
        </form>

        <div className="auth-card__footer">
          <p>Already have an account? <Link to="/login" className="auth-link">Sign In</Link></p>
        </div>
      </div>
    </div>
  );
}
