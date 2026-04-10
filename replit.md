# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.
Also includes a standalone Python Flask ML application for Handwritten Character Recognition.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Handwritten Character Recognition App

A Python Flask ML web app at `artifacts/hcr-app/`.

- **Backend**: Flask + TensorFlow + OpenCV
- **Frontend**: Vanilla HTML/CSS/JS with dark-themed drawing canvas
- **Model**: CNN trained on EMNIST Balanced (47 classes: A-Z, 0-9, + mixed case)
- **Port**: 8008

### File Structure

```
artifacts/hcr-app/
├── main.py              # Flask server with /predict endpoint
├── model_trainer.py     # CNN training script using EMNIST dataset
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Dark-themed drawing UI
├── static/
│   ├── css/style.css    # Modern dark theme styles
│   └── js/script.js     # Canvas drawing + fetch API logic
```

### Usage

1. **Train the model first**: Open a terminal and run:
   ```bash
   cd artifacts/hcr-app && python model_trainer.py
   ```
   This downloads EMNIST Balanced dataset and trains a CNN for ~5 epochs (takes a few minutes).

2. **Start the server**: The "Start application" workflow runs `python main.py` automatically.

3. **Use the app**: Draw a character in the canvas and click "Predict".

### Model Details

- Architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(47, softmax)
- Dataset: EMNIST Balanced — 47 classes (0-9, A-Z, plus lowercase a, b, d, e, f, g, h, n, q, r, t)
- EMNIST orientation fix applied: 90° CW rotation + horizontal flip before training
- Preprocessing: bounding box crop, 2px padding, resize to 28×28, invert, normalize [0,1]
