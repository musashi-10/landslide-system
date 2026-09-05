/**
 * App.tsx — root application with full routing.
 *
 * Routes:
 *   / → Landing
 *   /login → Login
 *   /register → Register
 *   /dashboard → UserDashboard (user only)
 *   /alerts → UserAlerts (user only)
 *   /settings → UserSettings (user only)
 *   /authority/command → CommandCenter (authority only)
 *   /authority/map → AuthRiskMap (authority only)
 *   /authority/alerts → ActiveAlerts (authority only)
 *   /authority/history → AlertHistory (authority only)
 *   /authority/system → SystemStatus (authority only)
 */

import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import { Navbar } from './components/layout/Navbar';

// Pages
import { Landing } from './pages/Landing';
import { Login } from './pages/auth/Login';
import { Register } from './pages/auth/Register';
import { UserDashboard } from './pages/user/UserDashboard';
import { UserAlerts } from './pages/user/UserAlerts';
import { UserSettings } from './pages/user/UserSettings';
import { CommandCenter } from './pages/authority/CommandCenter';
import { AuthRiskMap } from './pages/authority/AuthRiskMap';
import { ActiveAlerts } from './pages/authority/ActiveAlerts';
import { AlertHistory } from './pages/authority/AlertHistory';
import { SystemStatus } from './pages/authority/SystemStatus';

const IS_MOCK = import.meta.env.VITE_USE_MOCK_API === 'true';

export default function App() {
  return (
    <AuthProvider>
      <div className="app-shell">
        <Navbar />
        {IS_MOCK && <div className="global-mock-bar">MOCK DATA MODE — backend not connected</div>}

        <main className="app-main" role="main">
          <Routes>
            {/* Public */}
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* User (citizen) routes */}
            <Route path="/dashboard" element={
              <ProtectedRoute requiredRole="user">
                <UserDashboard />
              </ProtectedRoute>
            } />
            <Route path="/alerts" element={
              <ProtectedRoute requiredRole="user">
                <UserAlerts />
              </ProtectedRoute>
            } />
            <Route path="/settings" element={
              <ProtectedRoute requiredRole="user">
                <UserSettings />
              </ProtectedRoute>
            } />

            {/* Authority routes */}
            <Route path="/authority/command" element={
              <ProtectedRoute requiredRole="authority">
                <CommandCenter />
              </ProtectedRoute>
            } />
            <Route path="/authority/map" element={
              <ProtectedRoute requiredRole="authority">
                <AuthRiskMap />
              </ProtectedRoute>
            } />
            <Route path="/authority/alerts" element={
              <ProtectedRoute requiredRole="authority">
                <ActiveAlerts />
              </ProtectedRoute>
            } />
            <Route path="/authority/history" element={
              <ProtectedRoute requiredRole="authority">
                <AlertHistory />
              </ProtectedRoute>
            } />
            <Route path="/authority/system" element={
              <ProtectedRoute requiredRole="authority">
                <SystemStatus />
              </ProtectedRoute>
            } />

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </AuthProvider>
  );
}
