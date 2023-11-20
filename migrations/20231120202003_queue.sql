CREATE TABLE "queue" (
    "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
    "text" text NOT NULL,
    "channel" text NOT NULL,
    "created_at" integer NOT NULL,
    "leased_at" integer NOT NULL
);
