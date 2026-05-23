const defaultHttpBase = `${window.location.protocol}//${window.location.hostname}:8000`;

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? defaultHttpBase;

const envWsBase = (import.meta.env.VITE_WS_BASE as string | undefined)?.replace(/\/$/, "");
export const WS_BASE = envWsBase ?? API_BASE.replace(/^http/, "ws");
