/**
 * ForgeAI — Button Component
 *
 * Variants: primary, secondary, ghost, danger
 * States: loading, disabled
 * Accessible: uses <button> element, aria-disabled, focus ring
 */

'use client';

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { clsx } from 'clsx';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    loading?: boolean;
    icon?: ReactNode;
    fullWidth?: boolean;
}

const variantStyles: Record<string, string> = {
    primary:
        'bg-brand-500 hover:bg-brand-600 text-white shadow-md hover:shadow-lg active:bg-brand-700 disabled:bg-brand-300',
    secondary:
        'bg-surface-100 hover:bg-surface-200 text-surface-900 border border-surface-300 dark:bg-surface-800 dark:hover:bg-surface-700 dark:text-surface-100 dark:border-surface-600',
    ghost:
        'bg-transparent hover:bg-surface-100 text-surface-700 dark:text-surface-300 dark:hover:bg-surface-800',
    danger:
        'bg-danger hover:bg-red-600 text-white shadow-md active:bg-red-700 disabled:bg-red-300',
};

const sizeStyles: Record<string, string> = {
    sm: 'h-8 px-3 text-sm gap-1.5 rounded-lg',
    md: 'h-10 px-4 text-sm gap-2 rounded-xl',
    lg: 'h-12 px-6 text-base gap-2.5 rounded-xl',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    (
        {
            variant = 'primary',
            size = 'md',
            loading = false,
            disabled,
            icon,
            fullWidth,
            className,
            children,
            ...props
        },
        ref
    ) => {
        const isDisabled = disabled || loading;

        return (
            <button
                ref={ref}
                disabled={isDisabled}
                aria-disabled={isDisabled}
                className={clsx(
                    'inline-flex items-center justify-center font-medium transition-all duration-200 focus-ring',
                    variantStyles[variant],
                    sizeStyles[size],
                    fullWidth && 'w-full',
                    isDisabled && 'opacity-50 cursor-not-allowed',
                    className
                )}
                {...props}
            >
                {loading ? (
                    <svg
                        className="animate-spin h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        aria-hidden="true"
                    >
                        <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                        />
                        <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                    </svg>
                ) : (
                    icon
                )}
                {children}
            </button>
        );
    }
);

Button.displayName = 'Button';
