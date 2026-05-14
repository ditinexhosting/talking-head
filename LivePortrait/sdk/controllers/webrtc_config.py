from __future__ import annotations

import asyncio
import os
import re

WEBRTC_UDP_MIN_PORT = int(os.environ.get("WEBRTC_UDP_MIN_PORT", "50000"))
WEBRTC_UDP_MAX_PORT = int(os.environ.get("WEBRTC_UDP_MAX_PORT", "50010"))
WEBRTC_PUBLIC_IP = os.environ.get("WEBRTC_PUBLIC_IP")

_patched = False


def apply_aiortc_patches() -> None:
    """Force aioice STUN sockets into a fixed UDP port range.

    aioice calls `loop.create_datagram_endpoint(..., local_addr=(addr, 0))`,
    which lets the kernel pick any free port. That makes firewall rules
    impossible. We wrap the call: when port==0 is requested we iterate
    WEBRTC_UDP_MIN_PORT..MAX_PORT and use the first one that binds.
    """
    global _patched
    if _patched:
        return
    _patched = True

    loop_cls = asyncio.BaseEventLoop
    original = loop_cls.create_datagram_endpoint

    async def bounded(self, protocol_factory, *args, local_addr=None, **kwargs):
        if local_addr is not None and local_addr[1] == 0:
            host = local_addr[0]
            last_err: Exception | None = None
            for port in range(WEBRTC_UDP_MIN_PORT, WEBRTC_UDP_MAX_PORT + 1):
                try:
                    return await original(
                        self, protocol_factory, *args,
                        local_addr=(host, port), **kwargs,
                    )
                except OSError as exc:
                    last_err = exc
                    continue
            raise OSError(
                f"no free UDP port in {WEBRTC_UDP_MIN_PORT}-{WEBRTC_UDP_MAX_PORT} "
                f"on {host} ({last_err})"
            )
        return await original(
            self, protocol_factory, *args, local_addr=local_addr, **kwargs,
        )

    loop_cls.create_datagram_endpoint = bounded
    print(
        f"[webrtc-config] UDP port range pinned to "
        f"{WEBRTC_UDP_MIN_PORT}-{WEBRTC_UDP_MAX_PORT}",
        flush=True,
    )
    if WEBRTC_PUBLIC_IP:
        print(
            f"[webrtc-config] SDP candidates will be rewritten to "
            f"{WEBRTC_PUBLIC_IP}",
            flush=True,
        )


_CANDIDATE_IP_RE = re.compile(
    r"(a=candidate:\S+ \S+ \S+ \S+ )(\S+)( \d+ typ host)"
)


def rewrite_sdp_public_ip(sdp: str) -> str:
    """Replace the host candidate's IP with WEBRTC_PUBLIC_IP if configured.

    Used when the server binds to an internal interface (Docker bridge, NAT'd
    VM) but the browser must reach it on a public address. The socket still
    binds locally — only the IP advertised in SDP is rewritten.
    """
    if not WEBRTC_PUBLIC_IP:
        return sdp
    return _CANDIDATE_IP_RE.sub(rf"\g<1>{WEBRTC_PUBLIC_IP}\g<3>", sdp)
