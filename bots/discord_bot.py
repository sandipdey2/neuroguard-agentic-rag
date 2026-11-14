"""Optional Discord bot (discord.py): same handler as the Slack bot, slash command /neuroguard.

    pip install -r requirements-bots.txt ; export DISCORD_BOT_TOKEN=PASTE_YOUR_TOKEN_HERE ; python -m bots.discord_bot
Not deployed or tested end-to-end against Discord (no bot account in this project); the shared handler
(bots/slack_bot.py::build_reply) is unit-tested.
"""
from __future__ import annotations

import asyncio
import os

from bots.slack_bot import build_reply


def main() -> None:
    import discord
    from discord import app_commands
    client = discord.Client(intents=discord.Intents.default())
    tree = app_commands.CommandTree(client)

    @tree.command(name="neuroguard", description="Stroke-risk decision support (not a medical device)")
    async def neuroguard(interaction: discord.Interaction, text: str):
        await interaction.response.defer(ephemeral=True, thinking=True)
        out = await asyncio.to_thread(build_reply, text, f"discord-{interaction.user.id}")
        await interaction.followup.send(out["text"][:1900], ephemeral=True)

    @client.event
    async def on_ready():
        await tree.sync()

    client.run(os.environ["DISCORD_BOT_TOKEN"])


if __name__ == "__main__":
    main()
