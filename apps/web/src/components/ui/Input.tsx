/**
 * ForgeAI — Input Component
 *
 * Text input with label, error state, password strength meter,
 * and full accessibility support.
 */

'use client';

import { forwardRef, useState, type InputHTMLAttributes } from 'react';
import { clsx } from 'clsx';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    hint?: string;
    showPasswordStrength?: boolean;
}

function getPasswordStrength(password: string): {
    level: 'weak' | 'fair' | 'strong';
    score: number;
    label: string;
} {
    let score = 0;
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[a-z]/.test(password)) score++;
    if (/\d/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;

    if (score <= 2) return { level: 'weak', score, label: 'Weak' };
    if (score <= 4) return { level: 'fair', score, label: 'Fair' };
    return { level: 'strong', score, label: 'Strong' };
}

const strengthColors = {
    weak: 'bg-danger',
    fair: 'bg-warning',
    strong: 'bg-success',
};

export const Input = forwardRef<HTMLInputElement, InputProps>(
    (
        {
            label,
            error,
            hint,
            showPasswordStrength,
            className,
            type,
            id,
            ...props
        },
        ref
    ) => {
        const [showPassword, setShowPassword] = useState(false);
        const [value, setValue] = useState('');
        const inputId = id ?? label?.toLowerCase().replace(/\s+/g, '-');
        const isPassword = type === 'password';
        const strength =
            showPasswordStrength && isPassword ? getPasswordStrength(value) : null;

        return (
            <div className="space-y-1.5">
                {label && (
                    <label
                        htmlFor={inputId}
                        className="block text-sm font-medium text-surface-700 dark:text-surface-300"
                    >
                        {label}
                    </label>
                )}

                <div className="relative">
                    <input
                        ref={ref}
                        id={inputId}
                        type={isPassword && showPassword ? 'text' : type}
                        className={clsx(
                            'w-full h-10 px-3 rounded-xl border text-sm',
                            'bg-surface-0 dark:bg-surface-800',
                            'placeholder:text-surface-400 dark:placeholder:text-surface-500',
                            'transition-colors duration-200 focus-ring',
                            error
                                ? 'border-danger text-danger-dark'
                                : 'border-surface-300 dark:border-surface-600 hover:border-surface-400 dark:hover:border-surface-500',
                            className
                        )}
                        aria-invalid={!!error}
                        aria-describedby={error ? `${inputId}-error` : hint ? `${inputId}-hint` : undefined}
                        onChange={(e) => {
                            setValue(e.target.value);
                            props.onChange?.(e);
                        }}
                        {...props}
                    />

                    {isPassword && (
                        <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600 dark:hover:text-surface-300 text-xs font-medium"
                            tabIndex={-1}
                            aria-label={showPassword ? 'Hide password' : 'Show password'}
                        >
                            {showPassword ? 'Hide' : 'Show'}
                        </button>
                    )}
                </div>

                {/* Password Strength Meter */}
                {strength && value.length > 0 && (
                    <div className="space-y-1">
                        <div className="flex gap-1">
                            {[1, 2, 3].map((i) => (
                                <div
                                    key={i}
                                    className={clsx(
                                        'h-1 flex-1 rounded-full transition-colors',
                                        i <= (strength.level === 'weak' ? 1 : strength.level === 'fair' ? 2 : 3)
                                            ? strengthColors[strength.level]
                                            : 'bg-surface-200 dark:bg-surface-700'
                                    )}
                                />
                            ))}
                        </div>
                        <p className={clsx('text-xs', {
                            'text-danger': strength.level === 'weak',
                            'text-warning': strength.level === 'fair',
                            'text-success': strength.level === 'strong',
                        })}>
                            {strength.label}
                        </p>
                    </div>
                )}

                {error && (
                    <p id={`${inputId}-error`} className="text-xs text-danger" role="alert">
                        {error}
                    </p>
                )}
                {hint && !error && (
                    <p id={`${inputId}-hint`} className="text-xs text-surface-500">
                        {hint}
                    </p>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';
