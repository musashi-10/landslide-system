/**
 * Login page — supports both citizen and authority login.
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { Mountain, Mail, Lock, LogIn, AlertTriangle } from 'lucide-react';

export function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login({ email, password });
      // Navigate based on role
      const session = JSON.parse(sessionStorage.getItem('lews_session') ?? '{}');
      if (session?.user?.role === 'authority') {
        navigate('/authority/command');
      } else {
        navigate('/dashboard');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-card__header">
          <Mountain size={32} className="auth-card__logo" />
          <h1 className="auth-card__title">Sign In</h1>
          <p className="auth-card__subtitle">
            Access your landslide monitoring dashboard
          </p>
        </div>

        {error && (
          <div className="auth-error" role="alert">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="login-email" className="form-label">
              <Mail size={14} /> Email
            </label>
            <input
              id="login-email"
              type="email"
              className="form-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoComplete="email"
            />
          </div>

          <div className="form-group">
            <label htmlFor="login-password" className="form-label">
              <Lock size={14} /> Password
            </label>
            <input
              id="login-password"
              type="password"
              className="form-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoComplete="current-password"
            />
          </div>

          <button type="submit" className="btn btn--primary btn--full" disabled={loading}>
            {loading ? (
              <span className="btn-spinner" />
            ) : (
              <>
                <LogIn size={16} /> Sign In
              </>
            )}
          </button>
        </form>

        <div className="auth-card__footer">
          <p>
            Don't have an account?{' '}
            <Link to="/register" className="auth-link">Register</Link>
          </p>
        </div>

        <div className="auth-card__demo">
          <p className="auth-card__demo-title">Demo Credentials</p>
          <div className="demo-creds">
            <div className="demo-cred">
              <span className="demo-cred__role">Citizen</span>
              <code>user@example.com / password</code>
            </div>
            <div className="demo-cred">
              <span className="demo-cred__role">Authority</span>
              <code>authority@example.com / password</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
