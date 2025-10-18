# 🧠 AI StoryForge — Content Pipeline

## 1. 📝 Text Generation
- Prompt engineered children’s story generated using OpenAI or other LLM API.
- Context: age range, theme, moral of the story.
- Length: short (300–600 words).
- Output: structured story with title, chapters, and illustration cues.

## 2. 🎨 Image Generation
- Illustration prompts extracted from story text.
- Uses image generation API (e.g., OpenAI DALL·E, Midjourney, or Stable Diffusion).
- One image per scene or chapter.
- Automatically stored in cloud storage.

## 3. 🧹 Review & Edit Loop
- Human review twice a week (you).
- Text and images adjusted using feedback prompts.
- Final version approved for publishing.

## 4. 📦 Storage
- Approved stories and illustrations stored in database + cloud bucket.
- Metadata saved for indexing, publishing, and search.

## 5. 🚀 Publishing
- Export to EPUB/PDF.
- Push to Gumroad, Kindle Direct Publishing, or your own store.

## 🧰 Tech Stack
- **Backend**: FastAPI, Python
- **Frontend**: HTML / JS (upgrade to React later)
- **Image Gen**: OpenAI DALL·E / SD API
- **DB**: SQLite / Postgres
- **Storage**: MinIO / S3
