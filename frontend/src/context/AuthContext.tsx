/**
 * AuthContext — provides authentication state across the entire app.
 *
 * Wraps login/register/logout from authService and provides
 * the current user + role to any consuming component.
 */

import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { UserProfile, UserRole, LoginRequest, RegisterRequest } from '../types';
import * as authService from '../services/authService';

interface AuthContextValue {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (req: LoginRequest) => Promise<void>;
  register: (req: RegisterRequest) => Promise<void>;
  logout: () => void;
  hasRole: (role: UserRole) => boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Restore session on mount
  useEffect(() => {
    const existing = authService.getCurrentUser();
    if (existing) setUser(existing);
    setIsLoading(false);
  }, []);

  const login = useCallback(async (req: LoginRequest) => {
    const resp = await authService.login(req);
    setUser(resp.user);
  }, []);

  const register = useCallback(async (req: RegisterRequest) => {
    const resp = await authService.register(req);
    setUser(resp.user);
  }, []);

  const logout = useCallback(() => {
    authService.logout();
    setUser(null);
  }, []);

  const hasRole = useCallback(
    (role: UserRole) => user?.role === role,
    [user]
  );

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: user !== null,
        isLoading,
        login,
        register,
        logout,
        hasRole,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within <AuthProvider>');
  return ctx;
}
