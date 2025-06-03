import os
import discord
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Load model and tokenizer
model_path = "gpt2-medium"  # or "gpt2-medium" if no fine-tune
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

chat_history = []

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")
    await client.change_presence(activity=discord.Game(name="made by potatoguy0910"))

@client.event
async def on_message(message):
    if message.author == client.user or message.author.bot:
        return

    if client.user in message.mentions:
        user_input = message.content.replace(f"<@{client.user.id}>", "").strip()
        chat_history.append(f"You: {user_input}")

        system_prompt = (
            "You are Broly, a chill Gen-Z roaster bot. "
            "Keep replies SHORT (1–2 lines), casual, funny, sarcastic, human, cool. "
            "Never mention anything fake (no reddit, no girlfriend, no links, no stories, no cartoon characters). "
            "Sound like a bestie. Use emojis like 😭💀🔥. No weird stuff. Just be natural & fun.\n"
        )

        context = "\n".join(chat_history[-4:])
        prompt = f"{system_prompt}{context}\nBroly:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        async with message.channel.typing():
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.75,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.2,
                    eos_token_id=tokenizer.eos_token_id
                )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = decoded.split("Broly:")[-1].strip()

        # Cut to first full sentence only
        for end in [".", "!", "?", "\n"]:
            if end in reply:
                reply = reply.split(end)[0] + end
                break

        reply = reply.strip()

        # Block hallucinations or garbage
        banned_words = [
            "futurama", "reddit", "subreddit", "customer", "shop", "store", "girlfriend",
            "aliens", "http", "blog", "santa", "wikipedia", "username", "robot"
        ]

        if any(bad in reply.lower() for bad in banned_words) or len(reply) < 3:
            reply = "bro retry that 😭 I glitched out"

        chat_history.append(f"Broly: {reply}")
        await message.reply(reply)

# Run bot
if DISCORD_BOT_TOKEN:
    client.run(DISCORD_BOT_TOKEN)
else:
    print("❌ Missing DISCORD_BOT_TOKEN in .env")
