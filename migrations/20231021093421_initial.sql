CREATE TABLE "sessions" (
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
    "channel" text NOT NULL UNIQUE,
    "model_state" blob,
    "created_at" integer NOT NULL,
    "updated_at" integer NOT NULL
);
PRAGMA journal_mode=WAL;
