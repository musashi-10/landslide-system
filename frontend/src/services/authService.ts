/**
 * Auth service — handles login, register, and session management.
 * 
 * Uses mock backend until auth endpoints are implemented.
 * Token is stored in sessionStorage (never localStorage for security).
 */

import type { LoginRequest, RegisterRequest, AuthResponse, UserProfile, UserRole } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const USE_MOCK = import.meta.env.VITE_USE_MOCK_API === 'true';

const SESSION_KEY = 'lews_session';

// ── Mock users ──────────────────────────────────────────────────────────────

const MOCK_USERS: Record<string, { password: string; profile: UserProfile }> = {
  'user@example.com': {
    password: 'password',
    profile: {
      id: 'USR_001',
      name: 'Tenzing Sherpa',
      email: 'user@example.com',
      mobile: '+91-9876543210',
      role: 'user',
      location_id: 'LOC_001',
    },
  },
  'authority@example.com': {
    password: 'password',
    profile: {
      id: 'AUTH_001',
      name: 'District Collector',
      email: 'authority@example.com',
      mobile: '+91-9876543211',
      role: 'authority',
      location_id: null,
    },
  },
};

// ── Helpers ─────────────────────────────────────────────────────────────────

function delay(ms = 400): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function mockToken(): string {
  return `mock_jwt_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

// ── Public API ──────────────────────────────────────────────────────────────

export async function login(req: LoginRequest): Promise<AuthResponse> {
  if (USE_MOCK) {
    await delay();
    const entry = MOCK_USERS[req.email];
    if (!entry || entry.password !== req.password) {
      throw new Error('Invalid email or password');
    }
    const resp: AuthResponse = { token: mockToken(), user: entry.profile };
    saveSession(resp);
    return resp;
  }

  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.error?.message ?? 'Login failed');
  }
  const resp: AuthResponse = await res.json();
  saveSession(resp);
  return resp;
}

export async function register(req: RegisterRequest): Promise<AuthResponse> {
  if (USE_MOCK) {
    await delay(600);
    if (MOCK_USERS[req.email]) {
      throw new Error('Email already registered');
    }
    const profile: UserProfile = {
      id: `USR_${Date.now()}`,
      name: req.name,
      email: req.email,
      mobile: req.mobile,
      role: 'user',
      location_id: req.location_id ?? 'LOC_001',
    };
    const resp: AuthResponse = { token: mockToken(), user: profile };
    saveSession(resp);
    return resp;
  }

  const res = await fetch(`${BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.error?.message ?? 'Registration failed');
  }
  const resp: AuthResponse = await res.json();
  saveSession(resp);
  return resp;
}

export function logout(): void {
  sessionStorage.removeItem(SESSION_KEY);
}

export function getSession(): AuthResponse | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as AuthResponse;
  } catch {
    return null;
  }
}

export function getToken(): string | null {
  return getSession()?.token ?? null;
}

export function getCurrentUser(): UserProfile | null {
  return getSession()?.user ?? null;
}

export function isAuthenticated(): boolean {
  return getSession() !== null;
}

export function hasRole(role: UserRole): boolean {
  return getCurrentUser()?.role === role;
}

// ── Internal ────────────────────────────────────────────────────────────────

function saveSession(resp: AuthResponse): void {
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(resp));
}
