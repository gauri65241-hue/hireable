"""
Outbound-network safety helpers (SSRF guard).

Any URL that comes from a user — job-posting URLs, custom LLM base URLs — must
pass through here before the server connects to it.

What this enforces
  * scheme must be http/https; no embedded credentials; only common web ports
  * hostname must not be localhost / *.local / *.internal / cloud-metadata names
  * EVERY address the name resolves to must be globally routable (blocks
    loopback, RFC1918, link-local incl. 169.254.169.254, CGNAT, multicast,
    reserved, IPv4-mapped / NAT64 / 6to4 wrappers around those)
  * numeric tricks (2130706433, 0x7f.1, 017700000001) are covered because we
    validate the *resolved* addresses, not the string
  * redirects are followed manually and re-validated on every hop
  * the TCP connection is pinned to the validated IP (Host header + TLS SNI are
    kept), which closes the DNS-rebinding window between check and use
  * response bodies are streamed and capped

Residual risk: a host that is legitimately public but serves attacker-chosen
content (that is the nature of "fetch a URL"). Output is treated as untrusted
text downstream (see analysers.base.prepare_user_text).
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

ALLOWED_PORTS = {None, 80, 443, 8080, 8443}
MAX_REDIRECTS = 3
DEFAULT_MAX_BYTES = 2_000_000

_BLOCKED_NAMES = {"localhost", "metadata", "metadata.google.internal", "instance-data"}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_6TO4 = ipaddress.ip_network("2002::/16")


class UnsafeURLError(ValueError):
    """Raised when a URL must not be fetched. Message is safe to show to users."""


def _embedded_v4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip.ipv4_mapped:
        return ip.ipv4_mapped
    if ip in _NAT64:
        return ipaddress.IPv4Address(ip.packed[-4:])
    if ip in _6TO4:
        return ipaddress.IPv4Address(ip.packed[2:6])
    return None


def ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        inner = _embedded_v4(ip)
        if inner is not None:
            return ip_is_public(inner)
    return bool(ip.is_global) and not ip.is_multicast


def _check_url_shape(url: str, allowed_schemes: tuple[str, ...]):
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL is empty")
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in allowed_schemes:
        raise UnsafeURLError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeURLError("URL has no hostname")
    try:
        port = parts.port
    except ValueError:
        raise UnsafeURLError("URL has an invalid port") from None
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError("URL port is not allowed")
    if host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        raise UnsafeURLError("URL host is not allowed")
    return parts, host, port


def _validate_addrinfo(infos) -> list[str]:
    ips: list[str] = []
    for info in infos:
        raw = info[4][0].split("%")[0]  # strip IPv6 zone id
        ip = ipaddress.ip_address(raw)
        if not ip_is_public(ip):
            raise UnsafeURLError("URL resolves to a non-public address")
        ips.append(str(ip))
    if not ips:
        raise UnsafeURLError("URL host did not resolve")
    return list(dict.fromkeys(ips))


def resolve_public_sync(url: str, allowed_schemes=("http", "https")) -> tuple[str, list[str]]:
    """Blocking variant — use only outside the event loop. Returns (host, [ips])."""
    parts, host, port = _check_url_shape(url, allowed_schemes)
    try:
        infos = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise UnsafeURLError("URL host did not resolve") from None
    return host, _validate_addrinfo(infos)


async def resolve_public(url: str, allowed_schemes=("http", "https")) -> tuple[str, list[str]]:
    parts, host, port = _check_url_shape(url, allowed_schemes)
    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                             type=socket.SOCK_STREAM),
            timeout=5,
        )
    except (socket.gaierror, asyncio.TimeoutError):
        raise UnsafeURLError("URL host did not resolve") from None
    return host, _validate_addrinfo(infos)


def validate_public_url(url: str, allowed_schemes=("http", "https")) -> str:
    """Blocking validation (shape + DNS). Returns the normalised URL."""
    resolve_public_sync(url, allowed_schemes)
    return url.strip()


@dataclass
class SafeResponse:
    status_code: int
    headers: httpx.Headers
    content: bytes
    url: str
    truncated: bool

    @property
    def text(self) -> str:
        ctype = self.headers.get("content-type", "")
        enc = "utf-8"
        if "charset=" in ctype.lower():
            enc = ctype.lower().split("charset=")[-1].split(";")[0].strip() or "utf-8"
        try:
            return self.content.decode(enc, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")


def _pinned(url: str, ip: str) -> tuple[str, dict]:
    parts = urlsplit(url)
    host_hdr = parts.netloc.rsplit("@", 1)[-1]
    netloc_ip = f"[{ip}]" if ":" in ip else ip
    if parts.port:
        netloc_ip += f":{parts.port}"
    pinned = parts._replace(netloc=netloc_ip).geturl()
    return pinned, {"Host": host_hdr}


async def safe_get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: float = 15.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = MAX_REDIRECTS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SafeResponse:
    """GET with SSRF protection, pinned IPs, manual redirects and a body cap."""
    current = url.strip()
    for _ in range(max_redirects + 1):
        host, ips = await resolve_public(current)
        pinned_url, host_hdr = _pinned(current, ips[0])
        req_headers = {**(headers or {}), **host_hdr}
        ext = {"sni_hostname": host} if urlsplit(current).scheme == "https" else {}
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout,
                                     trust_env=False, transport=transport) as client:
            async with client.stream("GET", pinned_url, headers=req_headers, extensions=ext) as resp:
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                buf = bytearray()
                truncated = False
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        del buf[max_bytes:]
                        truncated = True
                        break
                return SafeResponse(resp.status_code, resp.headers, bytes(buf), current, truncated)
    raise UnsafeURLError("Too many redirects")
