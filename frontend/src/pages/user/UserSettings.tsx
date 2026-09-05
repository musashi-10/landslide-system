/**
 * User Settings — notification preferences management.
 *
 * Frontend only manages notification display preferences.
 * SMS sending is handled entirely by the backend alert engine.
 */

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { getNotificationPreferences, saveNotificationPreferences } from '../../services/userService';
import { LoadingState } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import type { NotificationPreferences } from '../../types';
import { Bell, Phone, Save, CheckCircle, AlertTriangle, Info } from 'lucide-react';

export function UserSettings() {
  const { user } = useAuth();
  const [prefs, setPrefs] = useState<NotificationPreferences | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    getNotificationPreferences()
      .then((p) => { setPrefs(p); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, []);

  const toggle = (key: keyof Omit<NotificationPreferences, 'mobile_number'>) => {
    if (!prefs) return;
    setPrefs((prev) => prev ? { ...prev, [key]: !prev[key] } : prev);
  };

  const handleSave = async () => {
    if (!prefs) return;
    setSaving(true);
    setSaveSuccess(false);
    try {
      await saveNotificationPreferences(prefs);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save preferences');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="page-container"><LoadingState message="Loading settings…" /></div>;
  if (error && !prefs) return <div className="page-container"><ErrorState message={error} /></div>;
  if (!prefs) return null;

  return (
    <div className="page-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">Notification Settings</h1>
        <p className="dashboard-subtitle">Configure your alert preferences for location {user?.location_id}</p>
      </div>

      <div className="settings-layout">
        <div className="settings-card">
          <div className="settings-card__header">
            <Bell size={18} />
            <h2>Alert Preferences</h2>
          </div>

          <div className="settings-info">
            <Info size={14} />
            <p>SMS alerts are sent by the backend alert engine. Your preferences determine which risk levels trigger notifications.</p>
          </div>

          <div className="settings-toggles">
            <div className="settings-toggle">
              <div className="settings-toggle__info">
                <span className="settings-toggle__label">Critical Risk SMS</span>
                <span className="settings-toggle__desc">Receive SMS when risk reaches CRITICAL level</span>
              </div>
              <button
                className={`toggle ${prefs.critical_sms ? 'toggle--on' : ''}`}
                onClick={() => toggle('critical_sms')}
                role="switch"
                aria-checked={prefs.critical_sms}
                id="toggle-critical"
              >
                <span className="toggle__thumb" />
              </button>
            </div>

            <div className="settings-toggle">
              <div className="settings-toggle__info">
                <span className="settings-toggle__label">High Risk SMS</span>
                <span className="settings-toggle__desc">Receive SMS when risk reaches HIGH level</span>
              </div>
              <button
                className={`toggle ${prefs.high_sms ? 'toggle--on' : ''}`}
                onClick={() => toggle('high_sms')}
                role="switch"
                aria-checked={prefs.high_sms}
                id="toggle-high"
              >
                <span className="toggle__thumb" />
              </button>
            </div>

            <div className="settings-toggle">
              <div className="settings-toggle__info">
                <span className="settings-toggle__label">Risk Increase Notifications</span>
                <span className="settings-toggle__desc">Alert when risk probability increases significantly</span>
              </div>
              <button
                className={`toggle ${prefs.risk_increase ? 'toggle--on' : ''}`}
                onClick={() => toggle('risk_increase')}
                role="switch"
                aria-checked={prefs.risk_increase}
                id="toggle-increase"
              >
                <span className="toggle__thumb" />
              </button>
            </div>

            <div className="settings-toggle">
              <div className="settings-toggle__info">
                <span className="settings-toggle__label">Emergency Alerts</span>
                <span className="settings-toggle__desc">Receive all emergency-level notifications</span>
              </div>
              <button
                className={`toggle ${prefs.emergency_alerts ? 'toggle--on' : ''}`}
                onClick={() => toggle('emergency_alerts')}
                role="switch"
                aria-checked={prefs.emergency_alerts}
                id="toggle-emergency"
              >
                <span className="toggle__thumb" />
              </button>
            </div>
          </div>
        </div>

        <div className="settings-card">
          <div className="settings-card__header">
            <Phone size={18} />
            <h2>Contact Information</h2>
          </div>

          <div className="form-group">
            <label htmlFor="mobile-number" className="form-label">
              <Phone size={14} /> Mobile Number for SMS
            </label>
            <input
              id="mobile-number"
              type="tel"
              className="form-input"
              value={prefs.mobile_number}
              onChange={(e) => setPrefs((prev) => prev ? { ...prev, mobile_number: e.target.value } : prev)}
              placeholder="+91-9876543210"
            />
            <p className="form-hint">Used by the backend alert engine for SMS notifications. Never shared.</p>
          </div>

          {error && (
            <div className="auth-error" role="alert">
              <AlertTriangle size={16} />
              {error}
            </div>
          )}

          {saveSuccess && (
            <div className="save-success" role="status">
              <CheckCircle size={16} />
              Preferences saved successfully
            </div>
          )}

          <button className="btn btn--primary" onClick={handleSave} disabled={saving}>
            {saving ? <span className="btn-spinner" /> : <><Save size={16} /> Save Preferences</>}
          </button>
        </div>
      </div>
    </div>
  );
}
