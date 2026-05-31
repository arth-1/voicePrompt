"""
Smoke test — verifies the server is running and the WebSocket works.

Run with:
    python scripts/smoke_test.py

Checks:
1. Health endpoint returns 200 OK
2. WebSocket connection succeeds
3. Server handles a reset message
"""

import asyncio
import json
import sys

try:
    import httpx
except ImportError:
    print("Install httpx for smoke test: pip install httpx")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Install websockets for smoke test: pip install websockets")
    sys.exit(1)

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/voice"


async def test_health():
    """Test the health endpoint."""
    print("1. Testing health endpoint...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok", f"Expected 'ok', got {data['status']}"
        print(f"   ✓ Health OK: {data}")


async def test_websocket():
    """Test WebSocket connection and reset."""
    print("2. Testing WebSocket connection...")
    async with websockets.connect(WS_URL) as ws:
        print("   ✓ WebSocket connected")

        # Send a reset message
        print("3. Testing reset message...")
        await ws.send(json.dumps({"type": "reset"}))

        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(response)
            print(f"   ✓ Got response: {msg}")
        except asyncio.TimeoutError:
            print("   ✓ No response (acceptable for reset)")

        # Send stop
        print("4. Testing stop message...")
        await ws.send(json.dumps({"type": "stop"}))
        print("   ✓ Stop sent")

    print("   ✓ WebSocket closed cleanly")


async def main():
    print("=" * 50)
    print("  Voice-Claude Bridge — Smoke Test")
    print("=" * 50)
    print()

    try:
        await test_health()
        print()
        await test_websocket()
        print()
        print("=" * 50)
        print("  All smoke tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n  ✗ Test failed: {e}")
        print("  Make sure the server is running: uvicorn backend.app.main:app")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
