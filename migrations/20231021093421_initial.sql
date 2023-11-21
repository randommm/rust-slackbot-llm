CREATE TABLE "sessions" (
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
    "channel" text NOT NULL,
    "thread_ts" text NOT NULL,
    "model_state" blob,
    "created_at" integer NOT NULL,
    "updated_at" integer NOT NULL,
    UNIQUE(channel, thread_ts)
);
CREATE TABLE "queue" (
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
    "text" text NOT NULL,
    "channel" text NOT NULL,
    "thread_ts" text NOT NULL,
    "created_at" integer NOT NULL,
    "leased_at" integer NOT NULL
);
PRAGMA journal_mode=WAL;
