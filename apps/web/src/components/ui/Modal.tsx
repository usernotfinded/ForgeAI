/**
 * ForgeAI — Modal Component
 *
 * Accessible overlay modal with focus trap, escape-to-close,
 * click-outside-to-close, and portal rendering.
 */

'use client';

import { useCallback, useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';

export interface ModalProps {
    open: boolean;
    onClose: () => void;
    title?: string;
    description?: string;
    size?: 'sm' | 'md' | 'lg';
    children: ReactNode;
}

const sizeStyles = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
};

export function Modal({
    open,
    onClose,
    title,
    description,
    size = 'md',
    children,
}: ModalProps) {
    const dialogRef = useRef<HTMLDivElement>(null);
    const previousFocusRef = useRef<HTMLElement | null>(null);

    // Focus trap & escape handler
    useEffect(() => {
        if (!open) return;

        previousFocusRef.current = document.activeElement as HTMLElement;

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
                return;
            }
            if (e.key !== 'Tab') return;

            const dialog = dialogRef.current;
            if (!dialog) return;

            const focusable = dialog.querySelectorAll<HTMLElement>(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            if (focusable.length === 0) return;

            const first = focusable[0];
            const last = focusable[focusable.length - 1];

            if (e.shiftKey && document.activeElement === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        document.body.style.overflow = 'hidden';

        // Auto-focus first focusable element
        requestAnimationFrame(() => {
            const firstFocusable = dialogRef.current?.querySelector<HTMLElement>(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            firstFocusable?.focus();
        });

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
            previousFocusRef.current?.focus();
        };
    }, [open, onClose]);

    const handleBackdropClick = useCallback(
        (e: React.MouseEvent) => {
            if (e.target === e.currentTarget) onClose();
        },
        [onClose]
    );

    if (!open) return null;

    const modal = (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
            onClick={handleBackdropClick}
            role="dialog"
            aria-modal="true"
            aria-label={title}
            aria-describedby={description ? 'modal-description' : undefined}
        >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

            {/* Panel */}
            <div
                ref={dialogRef}
                className={clsx(
                    'relative w-full bg-surface-0 dark:bg-surface-800 rounded-2xl shadow-2xl',
                    'animate-scale-in',
                    sizeStyles[size]
                )}
            >
                {title && (
                    <div className="flex items-center justify-between px-6 pt-6 pb-2">
                        <h2 className="text-lg font-semibold text-surface-900 dark:text-surface-50">
                            {title}
                        </h2>
                        <button
                            onClick={onClose}
                            className="p-1 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 text-surface-400 hover:text-surface-600 transition-colors"
                            aria-label="Close modal"
                        >
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                )}
                {description && (
                    <p id="modal-description" className="px-6 text-sm text-surface-500">
                        {description}
                    </p>
                )}
                <div className="p-6">{children}</div>
            </div>
        </div>
    );

    if (typeof window === 'undefined') return null;
    return createPortal(modal, document.body);
}
