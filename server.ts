import express from 'express';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isProduction = process.env.NODE_ENV === 'production';
const PORT = Number(process.env.PORT) || 3000;

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Primary and fallback models for lightning-fast high availability
const PRIMARY_MODEL = 'gemini-3.1-flash-lite';
const FALLBACK_MODEL = 'gemini-flash-latest';

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '25mb' }));

  // Static manifest and PWA assets for crawlers & PWABuilder
  const manifestPath = path.resolve(__dirname, 'public', 'manifest.json');
  app.get(['/manifest.json', '/manifest.webmanifest'], (_req, res) => {
    res.setHeader('Content-Type', 'application/manifest+json; charset=utf-8');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.sendFile(manifestPath);
  });

  // Health and connectivity check
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
      model: PRIMARY_MODEL,
      timestamp: Date.now(),
    });
  });

  // Chat completion endpoint (with SSE streaming support)
  app.post('/api/chat', async (req, res) => {
    console.log('Received /api/chat request. stream:', req.body?.stream);
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: 'Gemini API key is not configured on the server. Please check Settings > Secrets.',
      });
    }

    const { messages, settings, stream = true } = req.body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }

    const assistantName = settings?.assistantName?.trim() || 'Your Persnal AI';
    const language = settings?.language?.trim() || 'English';
    const studyMode = settings?.studyMode || 'step-by-step';
    const detailLevel = settings?.detailLevel || 'detailed';

    let modeInstruction = '';
    switch (studyMode) {
      case 'socratic':
        modeInstruction =
          'You are a Socratic tutor: Rather than immediately dumping full answers, provide guided hints, ask probing questions, and lead the student to discover the answers on their own.';
        break;
      case 'exam-prep':
        modeInstruction =
          'Focus on exam readiness: High-yield concepts, mnemonic memory tricks, common exam pitfalls, summary bullet points, and quick practice test checks.';
        break;
      case 'code-mentor':
        modeInstruction =
          'You are an expert programming mentor: Provide clean, idiomatic code with syntax highlighting tags, break down algorithms line-by-line, explain time/space complexity, and point out edge cases.';
        break;
      case 'quick-summary':
        modeInstruction =
          'You specialize in ultra-clear study summaries: Structure your response with an Executive Summary, Key Takeaways, Core Formulas/Definitions, and Actionable Revision Notes.';
        break;
      case 'language-tutor':
        modeInstruction =
          'You are a supportive language coach: Explain grammar rules clearly, provide vocabulary tables with pronunciation cues, sample dialogues, and contextual examples.';
        break;
      case 'step-by-step':
      default:
        modeInstruction =
          'Provide clear, sequential, step-by-step explanations. Number each major step, show intermediate reasoning (especially for mathematics, physics, and science), and conclude with a quick recap or verification check.';
        break;
    }

    let detailInstruction = '';
    if (detailLevel === 'concise') {
      detailInstruction = 'Keep explanations tight, direct, and focused without unnecessary fluff.';
    } else if (detailLevel === 'bullet-points') {
      detailInstruction = 'Structure major sections with organized bullet points and bold headers.';
    } else {
      detailInstruction = 'Provide comprehensive, thorough explanations with examples and clarity.';
    }

    const systemInstruction = `You are ${assistantName}, an encouraging, patient, and highly articulate personal AI study assistant.
Primary Language: Respond in ${language} (unless the student explicitly asks to practice another language or translates a specific passage).
Study Pedagogical Style:
${modeInstruction}
${detailInstruction}

Guidelines:
- Explain complex concepts simply without being condescending.
- When dealing with Mathematics or Science, format formulas clearly (show each algebraic or arithmetic transformation step-by-step).
- When formatting code, always include language identifiers in markdown code fences (e.g. \`\`\`python).
- Use tables for comparing terms, concepts, formulas, or vocabulary.
- If the user provides uploaded notes, diagrams, or documents, ground your answer directly in the provided material.
- Maintain a warm, encouraging, student-friendly tone. Never judge honest mistakes.`;

    // Build Gemini contents array from conversation messages
    const contents = messages.map((msg: any) => {
      const role = msg.role === 'assistant' || msg.role === 'model' ? 'model' : 'user';
      const parts: any[] = [];

      // Attachments (images or text files)
      if (Array.isArray(msg.attachments)) {
        for (const att of msg.attachments) {
          if (att.data && att.mimeType && att.mimeType.startsWith('image/')) {
            parts.push({
              inlineData: {
                mimeType: att.mimeType,
                data: att.data.replace(/^data:[^;]+;base64,/, ''),
              },
            });
          } else if (att.text) {
            parts.push({
              text: `[Attached Document/Notes: ${att.name || 'Study File'}]\n${att.text}\n[End of Document]\n`,
            });
          }
        }
      }

      if (msg.content) {
        parts.push({ text: msg.content });
      } else if (parts.length === 0) {
        parts.push({ text: ' ' });
      }

      return { role, parts };
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let activeModel = PRIMARY_MODEL;
      let streamSuccess = false;

      // Helper to attempt streaming
      const tryStream = async (model: string) => {
        const responseStream = await ai.models.generateContentStream({
          model,
          contents,
          config: {
            systemInstruction,
            temperature: 0.7,
          },
        });

        for await (const chunk of responseStream) {
          if (chunk.text) {
            res.write(`data: ${JSON.stringify({ text: chunk.text })}\n\n`);
          }
        }
      };

      try {
        await tryStream(activeModel);
        streamSuccess = true;
      } catch (primaryErr: any) {
        console.warn(`Primary model ${activeModel} failed, trying fallback ${FALLBACK_MODEL}:`, primaryErr.message);
        try {
          await tryStream(FALLBACK_MODEL);
          streamSuccess = true;
        } catch (fallbackErr: any) {
          console.error('All streaming models failed:', fallbackErr);
          res.write(`data: ${JSON.stringify({ error: fallbackErr.message || 'Error generating AI response.' })}\n\n`);
        }
      }

      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      try {
        const response = await ai.models.generateContent({
          model: PRIMARY_MODEL,
          contents,
          config: { systemInstruction, temperature: 0.7 },
        });
        res.json({ text: response.text || '' });
      } catch (err: any) {
        console.warn('Primary model failed for unary call, trying fallback:', err.message);
        try {
          const response = await ai.models.generateContent({
            model: FALLBACK_MODEL,
            contents,
            config: { systemInstruction, temperature: 0.7 },
          });
          res.json({ text: response.text || '' });
        } catch (fallbackErr: any) {
          res.status(500).json({ error: fallbackErr.message || 'Failed to generate response' });
        }
      }
    }
  });

  // Quick action: Generate a study quiz / flashcards based on topic or text
  app.post('/api/quiz-generate', async (req, res) => {
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ error: 'Gemini API key is missing on the server.' });
    }

    const { topic, context, language = 'English' } = req.body;
    if (!topic && !context) {
      return res.status(400).json({ error: 'Topic or study context is required.' });
    }

    const promptText = `Create 3 interactive study quiz questions based on the following topic or text.
Language: ${language}
Topic: ${topic || 'Study material'}
Context notes: ${context || 'None'}

Return ONLY a valid JSON object in this exact shape:
{
  "title": "Quiz Title",
  "questions": [
    {
      "question": "Clear question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctIndex": 0,
      "explanation": "Step-by-step explanation why this is correct."
    }
  ]
}`;

    const generateQuizWithModel = async (modelName: string) => {
      return await ai.models.generateContent({
        model: modelName,
        contents: promptText,
        config: {
          responseMimeType: 'application/json',
        },
      });
    };

    try {
      let response;
      try {
        response = await generateQuizWithModel(PRIMARY_MODEL);
      } catch {
        response = await generateQuizWithModel(FALLBACK_MODEL);
      }

      const parsed = JSON.parse(response.text || '{}');
      res.json(parsed);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to generate quiz.' });
    }
  });

  // Mount Vite middleware in development or serve static in production
  if (!isProduction) {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Your Persnal AI server is running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
