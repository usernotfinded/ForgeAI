/**
 * ForgeAI — Structured Frontend Logger
 *
 * Wraps console.* methods with structured JSON output in production
 * and readable format in development.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';
type LogContext = Record<string, unknown>;

const LOG_LEVELS: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
};

const currentLevel: LogLevel =
    (process.env.NEXT_PUBLIC_LOG_LEVEL as LogLevel) ?? 'info';
const isProduction = process.env.NODE_ENV === 'production';

function shouldLog(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= LOG_LEVELS[currentLevel];
}

function formatMessage(
    level: LogLevel,
    message: string,
    context?: LogContext
): string | object {
    if (isProduction) {
        return {
            timestamp: new Date().toISOString(),
            level,
            message,
            service: 'web',
            ...context,
        };
    }
    return message;
}

export const logger = {
    debug(message: string, context?: LogContext): void {
        if (!shouldLog('debug')) return;
        const formatted = formatMessage('debug', message, context);
        if (isProduction) {
            console.debug(JSON.stringify(formatted));
        } else {
            console.debug(`🐛 ${message}`, context ?? '');
        }
    },

    info(message: string, context?: LogContext): void {
        if (!shouldLog('info')) return;
        const formatted = formatMessage('info', message, context);
        if (isProduction) {
            console.info(JSON.stringify(formatted));
        } else {
            console.info(`ℹ️ ${message}`, context ?? '');
        }
    },

    warn(message: string, context?: LogContext): void {
        if (!shouldLog('warn')) return;
        const formatted = formatMessage('warn', message, context);
        if (isProduction) {
            console.warn(JSON.stringify(formatted));
        } else {
            console.warn(`⚠️ ${message}`, context ?? '');
        }
    },

    error(message: string, error?: unknown, context?: LogContext): void {
        if (!shouldLog('error')) return;
        const formatted = formatMessage('error', message, {
            ...context,
            error: error instanceof Error ? error.message : String(error),
        });
        if (isProduction) {
            console.error(JSON.stringify(formatted));
        } else {
            console.error(`❌ ${message}`, error, context ?? '');
        }
    },
};
