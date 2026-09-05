/**
 * Top navigation bar — adapts for public, user, and authority contexts.
 */

import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import {
  Mountain,
  LogOut,
  Shield,
  User,
  LayoutDashboard,
  Bell,
  Settings,
  Map,
  AlertTriangle,
  History,
  MonitorCheck,
} from 'lucide-react';

export function Navbar() {
  const { user, isAuthenticated, logout, hasRole } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="navbar" role="banner">
      <div className="navbar__left">
        <Link to="/" className="navbar__brand">
          <Mountain size={22} className="navbar__brand-icon" />
          <span className="navbar__brand-text">LEWS</span>
        </Link>

        {isAuthenticated && hasRole('user') && (
          <nav className="navbar__links" aria-label="User navigation">
            <Link to="/dashboard" className={`navbar__link ${isActive('/dashboard') ? 'navbar__link--active' : ''}`}>
              <LayoutDashboard size={16} /> Dashboard
            </Link>
            <Link to="/alerts" className={`navbar__link ${isActive('/alerts') ? 'navbar__link--active' : ''}`}>
              <Bell size={16} /> Alerts
            </Link>
            <Link to="/settings" className={`navbar__link ${isActive('/settings') ? 'navbar__link--active' : ''}`}>
              <Settings size={16} /> Settings
            </Link>
          </nav>
        )}

        {isAuthenticated && hasRole('authority') && (
          <nav className="navbar__links" aria-label="Authority navigation">
            <Link to="/authority/command" className={`navbar__link ${isActive('/authority/command') ? 'navbar__link--active' : ''}`}>
              <Shield size={16} /> Command Center
            </Link>
            <Link to="/authority/map" className={`navbar__link ${isActive('/authority/map') ? 'navbar__link--active' : ''}`}>
              <Map size={16} /> Risk Map
            </Link>
            <Link to="/authority/alerts" className={`navbar__link ${isActive('/authority/alerts') ? 'navbar__link--active' : ''}`}>
              <AlertTriangle size={16} /> Active Alerts
            </Link>
            <Link to="/authority/history" className={`navbar__link ${isActive('/authority/history') ? 'navbar__link--active' : ''}`}>
              <History size={16} /> Alert History
            </Link>
            <Link to="/authority/system" className={`navbar__link ${isActive('/authority/system') ? 'navbar__link--active' : ''}`}>
              <MonitorCheck size={16} /> System
            </Link>
          </nav>
        )}
      </div>

      <div className="navbar__right">
        {isAuthenticated ? (
          <>
            <div className="navbar__user">
              {hasRole('authority') ? (
                <Shield size={14} className="navbar__role-icon" />
              ) : (
                <User size={14} className="navbar__role-icon" />
              )}
              <span className="navbar__username">{user?.name}</span>
            </div>
            <button className="btn btn--ghost btn--sm" onClick={handleLogout} title="Sign out">
              <LogOut size={16} />
            </button>
          </>
        ) : (
          <div className="navbar__auth-links">
            <Link to="/login" className="btn btn--ghost btn--sm">Sign In</Link>
            <Link to="/register" className="btn btn--primary btn--sm">Register</Link>
          </div>
        )}
      </div>
    </header>
  );
}
