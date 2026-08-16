"""Regression: Cloudflare detection must fire on a REAL SDK error object.

`_is_cloudflare_challenge` guards on ``error.body``. The rest of the suite
constructs the error with ``body=None`` by hand -- but the real Anthropic SDK
never does that for an HTML page: when it cannot parse the body as JSON it
stores the RAW TEXT in ``error.body`` (a str, not None). A "body is not None"
guard therefore bailed on exactly the challenge pages this detector exists to
catch, and the hand-built fixtures encoded the wrong assumption.

This builds the error the way the SDK actually does -- via
``client._make_status_error_from_response(response)`` -- so it fails if anyone
reintroduces the body-is-None premise.
"""

import httpx
import anthropic

from amplifier_module_provider_anthropic import AnthropicProvider


def _sdk_error(status: int, content_type: str, body: bytes) -> anthropic.APIStatusError:
    client = anthropic.Anthropic(api_key="x")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        status, headers={"content-type": content_type}, content=body, request=request
    )
    return client._make_status_error_from_response(response)


def test_real_html_error_body_is_str_not_none():
    err = _sdk_error(403, "text/html", b"<html>Just a moment...</html>")
    assert isinstance(err.body, str)


def test_real_json_error_body_is_dict():
    err = _sdk_error(
        400, "application/json", b'{"type":"error","error":{"message":"bad"}}'
    )
    assert isinstance(err.body, dict)


def test_real_html_challenge_is_detected():
    err = _sdk_error(
        403, "text/html", b"<html><title>Just a moment...</title>Cloudflare</html>"
    )
    assert AnthropicProvider._is_cloudflare_challenge(err) is True


def test_real_json_error_is_not_a_challenge():
    err = _sdk_error(
        400, "application/json", b'{"type":"error","error":{"message":"bad"}}'
    )
    assert AnthropicProvider._is_cloudflare_challenge(err) is False
