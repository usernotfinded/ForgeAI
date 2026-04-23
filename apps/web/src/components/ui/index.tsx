/**
 * ForgeAI — Skeleton Loader Component
 * Animated placeholder for loading states.
 */

import { clsx } from 'clsx';

export interface SkeletonProps {
    className?: string;
    variant?: 'text' | 'circular' | 'rectangular';
    width?: string | number;
    height?: string | number;
}

export function Skeleton({
    className,
    variant = 'text',
    width,
    height,
}: SkeletonProps) {
    return (
        <div
            className={clsx(
                'animate-shimmer bg-gradient-to-r from-surface-200 via-surface-100 to-surface-200 dark:from-surface-700 dark:via-surface-600 dark:to-surface-700 bg-[length:400%_100%]',
                variant === 'circular' && 'rounded-full',
                variant === 'text' && 'rounded-md h-4',
                variant === 'rectangular' && 'rounded-xl',
                className
            )}
            style={{ width, height }}
            aria-hidden="true"
        />
    );
}

/**
 * ForgeAI — Badge Component
 */

export interface BadgeProps {
    children: React.ReactNode;
    variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
    size?: 'sm' | 'md';
    className?: string;
}

const badgeVariants: Record<string, string> = {
    default: 'bg-surface-100 text-surface-700 dark:bg-surface-700 dark:text-surface-300',
    success: 'bg-success-light text-success-dark dark:bg-success/20 dark:text-success',
    warning: 'bg-warning-light text-warning-dark dark:bg-warning/20 dark:text-warning',
    danger: 'bg-danger-light text-danger-dark dark:bg-danger/20 dark:text-danger',
    info: 'bg-info-light text-info-dark dark:bg-info/20 dark:text-info',
};

export function Badge({ children, variant = 'default', size = 'sm', className }: BadgeProps) {
    return (
        <span
            className={clsx(
                'inline-flex items-center font-medium rounded-full',
                size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm',
                badgeVariants[variant],
                className
            )}
        >
            {children}
        </span>
    );
}

/**
 * ForgeAI — Card Component
 */

export interface CardProps {
    children: React.ReactNode;
    className?: string;
    hover?: boolean;
    padding?: 'sm' | 'md' | 'lg';
}

const paddingStyles = { sm: 'p-4', md: 'p-6', lg: 'p-8' };

export function Card({ children, className, hover, padding = 'md' }: CardProps) {
    return (
        <div
            className={clsx(
                'bg-surface-0 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 rounded-2xl',
                paddingStyles[padding],
                hover && 'transition-all duration-200 hover:shadow-lg hover:border-surface-300 dark:hover:border-surface-600 cursor-pointer',
                className
            )}
        >
            {children}
        </div>
    );
}

/**
 * ForgeAI — Accordion Component (used in FAQ)
 */

'use client';

import { useState } from 'react';

export interface AccordionItem {
    trigger: string;
    content: string;
}

export interface AccordionProps {
    items: AccordionItem[];
    className?: string;
}

export function Accordion({ items, className }: AccordionProps) {
    const [openIndex, setOpenIndex] = useState<number | null>(null);

    return (
        <div className={clsx('divide-y divide-surface-200 dark:divide-surface-700', className)}>
            {items.map((item, i) => (
                <div key={i}>
                    <button
                        onClick={() => setOpenIndex(openIndex === i ? null : i)}
                        className="flex items-center justify-between w-full py-4 text-left text-surface-900 dark:text-surface-50 hover:text-brand-500 transition-colors"
                        aria-expanded={openIndex === i}
                    >
                        <span className="text-sm font-medium pr-4">{item.trigger}</span>
                        <svg
                            className={clsx(
                                'w-5 h-5 flex-shrink-0 text-surface-400 transition-transform duration-200',
                                openIndex === i && 'rotate-180'
                            )}
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>
                    {openIndex === i && (
                        <div className="pb-4 text-sm text-surface-600 dark:text-surface-400 animate-slide-down">
                            {item.content}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}
