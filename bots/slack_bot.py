"""NeuroGuard Slack bot (Slack Bolt for Python, Socket Mode - no public URL needed).

    export SLACK_BOT_TOKEN=xoxb-PASTE_YOUR_TOKEN_HERE SLACK_APP_TOKEN=xapp-PASTE_YOUR_TOKEN_HERE OPENAI_API_KEY=PASTE_YOUR_KEY_HERE
    python -m bots.slack_bot

Slash command ``/neuroguard``:
  /neuroguard row 9046                                   -> run a real row of the public dataset
  /neuroguard age=67 gender=Male hypertension=1 glucose=228.7 bmi=36.6 smoking=former
  /neuroguard what should we discuss about blood pressure?   -> follow-up on your last case (conversation memory)
  /neuroguard help

The handler acks within 3 s (Slack's limit), runs the multi-agent pipeline in a worker thread and responds with the
report summary, citations and the signed receipt prefix (``response_type`` ephemeral: only the requester sees it).
Tokens come from the environment only. Session = ``slack-<team>-<user>`` (pseudonymous); consent is off, so chat turns
stay in process memory. App manifest + setup: docs/BUYER_SETUP.md#slack-bot.
"""
from __future__ import annotations

import os
import threading

from core.chat import handle_message

HELP = ("*NeuroGuard* (decision support, not a medical device)\n"
        "• `/neuroguard row 9046` - assess a real row of the public stroke dataset\n"
        "• `/neuroguard age=67 gender=Male hypertension=1 glucose=228.7 bmi=36.6 smoking=former`\n"
        "• `/neuroguard <follow-up question>` - asks about your last case\n"
        "Never paste real patient identifiers.")

_last_case: dict[str, dict] = {}


def session_for(body: dict) -> str:
    return f"slack-{body.get('team_id', 't')}-{body.get('user_id', 'u')}"


def build_reply(text: str, session_id: str, run=None, follow=None) -> dict:
    """Pure handler logic (unit-tested): returns a Slack message payload."""
    text = (text or "").strip()
    if not text or text.lower() in ("help", "?"):
        return {"response_type": "ephemeral", "text": HELP}
    out = handle_message(text, session_id, case=_last_case.get(session_id), consent=False, run=run, follow=follow)
    if out["kind"] == "case":
        _last_case[session_id] = out["case"]
    reply = out["reply"].replace("**", "*")
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": reply[:2900]}}]
    rc = (out.get("followup") or {}).get("receipt") or (out.get("result") or {}).get("receipt")
    if rc:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text":
                       f"Signed receipt `{rc['signature'][:16]}...` · {rc['algorithm']} · contract `{rc.get('contract_sha256', '')[:12]}`"}]})
    return {"response_type": "ephemeral", "text": reply[:3000], "blocks": blocks}


def handle_command(ack, body, respond, run=None, follow=None, background: bool = True) -> None:
    """Bolt listener for /neuroguard: ack immediately, work in a thread, then respond()."""
    ack({"response_type": "ephemeral", "text": "NeuroGuard is working on it (about 1-2 minutes)..."})
    sid = session_for(body)

    def work():
        try:
            respond(build_reply(body.get("text", ""), sid, run=run, follow=follow))
        except Exception as exc:
            respond({"response_type": "ephemeral", "text": f"NeuroGuard error: {type(exc).__name__}: {str(exc)[:200]}"})
    if background:
        threading.Thread(target=work, daemon=True).start()
    else:
        work()


def create_app():
    from slack_bolt import App
    app = App(token=os.environ["SLACK_BOT_TOKEN"], signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))
    app.command("/neuroguard")(lambda ack, body, respond: handle_command(ack, body, respond))
    return app


def main() -> None:
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY"):
        if not os.environ.get(k):
            raise SystemExit(f"{k} is not set")
    SocketModeHandler(create_app(), os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
