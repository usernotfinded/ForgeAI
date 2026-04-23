/**
 * ForgeAI — Button Component Unit Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '@/components/ui/Button';

describe('Button', () => {
    it('renders children text', () => {
        render(<Button>Click me</Button>);
        expect(screen.getByText('Click me')).toBeInTheDocument();
    });

    it('calls onClick when clicked', () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick}>Click</Button>);
        fireEvent.click(screen.getByText('Click'));
        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('is disabled when disabled prop is true', () => {
        render(<Button disabled>Disabled</Button>);
        const button = screen.getByText('Disabled');
        expect(button).toBeDisabled();
        expect(button).toHaveAttribute('aria-disabled', 'true');
    });

    it('is disabled when loading is true', () => {
        render(<Button loading>Loading</Button>);
        const button = screen.getByRole('button');
        expect(button).toBeDisabled();
    });

    it('shows spinner when loading', () => {
        render(<Button loading>Loading</Button>);
        const svg = screen.getByRole('button').querySelector('svg');
        expect(svg).toBeInTheDocument();
        expect(svg).toHaveClass('animate-spin');
    });

    it('applies variant styles', () => {
        const { rerender } = render(<Button variant="primary">Primary</Button>);
        expect(screen.getByText('Primary')).toHaveClass('bg-brand-500');

        rerender(<Button variant="danger">Danger</Button>);
        expect(screen.getByText('Danger')).toHaveClass('bg-danger');
    });

    it('applies fullWidth class', () => {
        render(<Button fullWidth>Full</Button>);
        expect(screen.getByText('Full')).toHaveClass('w-full');
    });
});
