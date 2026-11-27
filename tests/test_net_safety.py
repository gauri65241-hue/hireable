"""SSRF guard: URL shape, address classification, pinned fetch, redirects, body cap."""
import asyncio
import ipaddress

import httpx
import pytest

import net_safety
from net_safety import UnsafeURLError, ip_is_public, safe_get, validate_public_url


@pytest.mark.parametrize("ip,public", [
    ("8.8.8.8", True), ("93.184.216.34", True), ("2606:4700:4700::1111", True),
    ("127.0.0.1", False), ("10.1.2.3", False), ("172.16.0.9", False), ("192.168.1.1", False),
    ("169.254.169.254", False),          # cloud metadata
    ("100.64.0.1", False),               # CGNAT
    ("0.0.0.0", False), ("224.0.0.1", False),
    ("::1", False), ("fe80::1", False), ("fc00::1", False),
    ("::ffff:127.0.0.1", False),         # IPv4-mapped loopback
    ("64:ff9b::7f00:1", False),          # NAT64 wrapping 127.0.0.1
    ("2002:7f00:0001::1", False),        # 6to4 wrapping 127.0.0.1
])
def test_ip_classification(ip, public):
    assert ip_is_public(ipaddress.ip_address(ip)) is public


@pytest.mark.parametrize("url", [
    "", "   ", "file:///etc/passwd", "ftp://example.com/x", "gopher://example.com",
    "http://user:pw@example.com/", "http://localhost/", "http://LOCALHOST./",
    "http://foo.localhost/", "http://metadata.google.internal/computeMetadata/v1/",
    "http://printer.local/", "http://example.com:22/", "http://example.com:6379/",
    "http://127.0.0.1/", "http://[::1]/", "http://169.254.169.254/latest/meta-data/",
    "http://2130706433/",                 # decimal 127.0.0.1
    "http://0x7f.0.0.1/", "http://017700000001/",
])
def test_rejected_urls(url):
    with pytest.raises(UnsafeURLError):
        validate_public_url(url)


def test_unresolvable_host_is_rejected(monkeypatch):
    import socket
    def boom(*a, **k): raise socket.gaierror("nope")
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", boom)
    with pytest.raises(UnsafeURLError):
        validate_public_url("http://does-not-exist.example/")


def test_mixed_public_and_private_resolution_is_rejected():
    # A name with ANY private A/AAAA record must fail (DNS rebinding / split-horizon).
    infos = [(2, 1, 6, "", ("93.184.216.34", 80)), (2, 1, 6, "", ("10.0.0.5", 80))]
    with pytest.raises(UnsafeURLError):
        net_safety._validate_addrinfo(infos)


@pytest.fixture
def fake_dns(monkeypatch):
    table = {"good.example": ["93.184.216.34"], "other.example": ["93.184.216.35"]}
    async def fake_resolve(url, allowed_schemes=("http", "https")):
        parts, host, _ = net_safety._check_url_shape(url, allowed_schemes)
        if host not in table:
            raise UnsafeURLError("URL resolves to a non-public address")
        return host, table[host]
    monkeypatch.setattr(net_safety, "resolve_public", fake_resolve)


def test_fetch_is_pinned_to_validated_ip_and_keeps_host_header(fake_dns):
    seen = {}
    def handler(request: httpx.Request):
        seen["host"], seen["hdr"] = request.url.host, request.headers["host"]
        return httpx.Response(200, text="hello")
    r = asyncio.run(safe_get("http://good.example/jobs/1", transport=httpx.MockTransport(handler)))
    assert r.status_code == 200 and r.text == "hello"
    assert seen == {"host": "93.184.216.34", "hdr": "good.example"}


def test_redirect_to_private_host_is_blocked(fake_dns):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://internal.corp/admin"})
    with pytest.raises(UnsafeURLError):
        asyncio.run(safe_get("http://good.example/", transport=httpx.MockTransport(handler)))


def test_redirect_chain_to_public_host_is_followed(fake_dns):
    def handler(request):
        if request.headers["host"] == "good.example":
            return httpx.Response(301, headers={"location": "http://other.example/final"})
        return httpx.Response(200, text="done")
    r = asyncio.run(safe_get("http://good.example/", transport=httpx.MockTransport(handler)))
    assert r.text == "done" and r.url == "http://other.example/final"


def test_redirect_loop_is_capped(fake_dns):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://good.example/again"})
    with pytest.raises(UnsafeURLError, match="redirect"):
        asyncio.run(safe_get("http://good.example/", transport=httpx.MockTransport(handler)))


def test_body_is_capped(fake_dns):
    def handler(request):
        return httpx.Response(200, content=b"x" * 5_000_000)
    r = asyncio.run(safe_get("http://good.example/", max_bytes=10_000,
                             transport=httpx.MockTransport(handler)))
    assert r.truncated and len(r.content) == 10_000
