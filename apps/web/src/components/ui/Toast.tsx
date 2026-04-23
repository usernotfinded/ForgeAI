/**
 * ForgeAI — Toast Notification System
 *
 * Queue-based toast notifications with auto-dismiss, action buttons,
 * and variants (success, error, warning, info).
 */

'use client';

import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import { clsx } from 'clsx';

// ─── Types ──────────────────────────────────────────────────────────────────

type ToastVariant = 'success' | 'error' | 'warning' | 'info';

interface Toast {
    id: string;
    message: string;
    variant: ToastVariant;
    duration?: number;
    action?: { label: string; onClick: () => void };
}

interface ToastContextType {
    toasts: Toast[];
    addToast: (toast: Omit<Toast, 'id'>) => void;
    removeToast: (id: string) => void;
}

// ─── Context ────────────────────────────────────────────────────────────────

const ToastContext = createContext<ToastContextType | null>(null);

export function useToast() {
    const ctx = useContext(ToastContext);
    if (!ctx) throw new Error('useToast must be used within ToastProvider');
    return ctx;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

const variantStyles: Record<ToastVariant, string> = {
    success: 'bg-success/10 border-success/30 text-success-dark dark:text-success',
    error: 'bg-danger/10 border-danger/30 text-danger-dark dark:text-danger',
    warning: 'bg-warning/10 border-warning/30 text-warning-dark dark:text-warning',
    info: 'bg-info/10 border-info/30 text-info-dark dark:text-info',
};

const variantIcons: Record<ToastVariant, string> = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ',
};

// ─── Provider ───────────────────────────────────────────────────────────────

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
        const id = Math.random().toString(36).slice(2);
        const newToast: Toast = { ...toast, id };
        setToasts((prev) => [...prev, newToast]);

        // Auto-dismiss
        const duration = toast.duration ?? 5000;
        if (duration > 0) {
            setTimeout(() => {
                setToasts((prev) => prev.filter((t) => t.id !== id));
            }, duration);
        }
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    }, []);

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </ToastContext.Provider>
    );
}

// ─── Container ──────────────────────────────────────────────────────────────

function ToastContainer({
    toasts,
    removeToast,
}: {
    toasts: Toast[];
    removeToast: (id: string) => void;
}) {
    if (toasts.length === 0) return null;

    return (
        <div
            className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm w-full"
            aria-live="polite"
            aria-label="Notifications"
        >
            {toasts.map((toast) => (
                <div
                    key={toast.id}
                    className={clsx(
                        'flex items-start gap-3 px-4 py-3 rounded-xl border shadow-lg animate-slide-up',
                        'bg-surface-0 dark:bg-surface-800',
                        variantStyles[toast.variant]
                    )}
                    role="alert"
                >
                    <span className="text-lg mt-0.5" aria-hidden="true">
                        {variantIcons[toast.variant]}
                    </span>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium">{toast.message}</p>
                        {toast.action && (
                            <button
                                onClick={toast.action.onClick}
                                className="mt-1 text-xs font-semibold underline underline-offset-2 hover:no-underline"
                            >
                                {toast.action.label}
                            </button>
                        )}
                    </div>
                    <button
                        onClick={() => removeToast(toast.id)}
                        className="text-surface-400 hover:text-surface-600 dark:hover:text-surface-300 mt-0.5"
                        aria-label="Dismiss notification"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
            ))}
        </div>
    );
}
