import "dotenv/config";

import express, {
    Request,
    Response,
    NextFunction,
} from "express";

import axios from "axios";

import {
    create,
    decryptMedia,
    Client,
    Message,
} from "@open-wa/wa-automate";


// ============================================================
// Configuration
// ============================================================

const BACKEND_URL =
    process.env.BACKEND_URL ??
    "http://backend:8000";

const GATEWAY_SECRET =
    process.env.WHATSAPP_GATEWAY_SECRET ??
    "";

const PORT =
    Number(
        process.env.PORT ?? 3001,
    );

const SESSION_ID =
    process.env.WHATSAPP_SESSION_ID ??
    "aqar";


if (!GATEWAY_SECRET) {
    throw new Error(
        "WHATSAPP_GATEWAY_SECRET is required",
    );
}


// ============================================================
// State
// ============================================================

let whatsappClient:
    Client | null = null;


// Basic duplicate protection.
//
// If you later run multiple WhatsApp containers,
// move this to Redis.
const processedMessages =
    new Map<string, number>();

const MESSAGE_TTL =
    10 * 60 * 1000;


// ============================================================
// Backend client
// ============================================================

const backend = axios.create({
    baseURL: BACKEND_URL,

    timeout: 60_000,

    headers: {
        Authorization:
            `Bearer ${GATEWAY_SECRET}`,
    },
});


// ============================================================
// Duplicate protection
// ============================================================

function alreadyProcessed(
    messageId: string,
): boolean {

    const now =
        Date.now();


    for (
        const [id, timestamp]
        of processedMessages
    ) {

        if (
            now - timestamp >
            MESSAGE_TTL
        ) {
            processedMessages.delete(id);
        }
    }


    if (
        processedMessages.has(
            messageId,
        )
    ) {
        return true;
    }


    processedMessages.set(
        messageId,
        now,
    );


    return false;
}


// ============================================================
// Send text to FastAPI
// ============================================================

async function sendTextToBackend(
    message: Message,
): Promise<string | null> {

    const response =
        await backend.post(
            "/internal/whatsapp/message",
            {
                message_id:
                    message.id,

                sender:
                    message.from,

                type:
                    "text",

                text:
                    message.body,
            },
        );


    return (
        response.data?.reply ??
        null
    );
}


// ============================================================
// Send voice note to FastAPI
// ============================================================

async function sendAudioToBackend(
    message: Message,
): Promise<string | null> {

    /*
     * OpenWA decrypts the WhatsApp media
     * for us.
     */
    const media =
        await decryptMedia(
            message,
        );


    const audioBuffer =
        Buffer.isBuffer(media)
            ? media
            : Buffer.from(media);


    const mimeType =
        message.mimetype ??
        "audio/ogg";


    /*
     * Convert to base64.
     *
     * This avoids needing multipart/form-data
     * in the WhatsApp gateway.
     *
     * FastAPI can decode it and pass the bytes
     * to your voice worker.
     */
    const audioBase64 =
        audioBuffer.toString(
            "base64",
        );


    const response =
        await backend.post(
            "/internal/whatsapp/audio",
            {
                message_id:
                    message.id,

                sender:
                    message.from,

                mime_type:
                    mimeType,

                audio_base64:
                    audioBase64,
            },
            {
                timeout:
                    120_000,
            },
        );


    return (
        response.data?.reply ??
        null
    );
}


// ============================================================
// Send WhatsApp message
// ============================================================

async function sendWhatsAppText(
    recipient: string,
    text: string,
): Promise<void> {

    if (!whatsappClient) {

        throw new Error(
            "WhatsApp client is not connected",
        );
    }


    await whatsappClient.sendText(
        recipient,
        text,
    );
}


// ============================================================
// Incoming message handler
// ============================================================

async function handleMessage(
    message: Message,
): Promise<void> {

    try {

        // ----------------------------------------------------
        // Ignore our own messages
        // ----------------------------------------------------

        if (message.fromMe) {
            return;
        }


        // ----------------------------------------------------
        // Ignore groups
        // ----------------------------------------------------

        if (message.isGroupMsg) {
            return;
        }


        // ----------------------------------------------------
        // Ignore status
        // ----------------------------------------------------

        if (
            message.from?.includes(
                "status@broadcast",
            )
        ) {
            return;
        }


        // ----------------------------------------------------
        // Duplicate protection
        // ----------------------------------------------------

        if (
            alreadyProcessed(
                message.id,
            )
        ) {
            return;
        }


        console.log(
            `[WhatsApp] ${message.type} from ${message.from}`,
        );


        // ====================================================
        // TEXT
        // ====================================================

        if (
            message.type === "chat" &&
            message.body
        ) {

            const reply =
                await sendTextToBackend(
                    message,
                );


            if (reply) {

                await sendWhatsAppText(
                    message.from,
                    reply,
                );
            }


            return;
        }


        // ====================================================
        // VOICE / AUDIO
        // ====================================================

        if (
            message.type === "ptt" ||
            message.type === "audio"
        ) {

            const reply =
                await sendAudioToBackend(
                    message,
                );


            /*
             * Normally audio processing should
             * be asynchronous.
             *
             * If FastAPI immediately produced
             * a response, send it here.
             *
             * Otherwise it returns reply=null
             * and the worker later calls:
             *
             * POST /send/text
             */
            if (reply) {

                await sendWhatsAppText(
                    message.from,
                    reply,
                );
            }


            return;
        }


        console.log(
            `[WhatsApp] Ignored message type: ${message.type}`,
        );

    } catch (error) {

        console.error(
            "[WhatsApp] Message processing failed",
            error,
        );
    }
}


// ============================================================
// Internal HTTP authentication
// ============================================================

function authenticate(
    req: Request,
    res: Response,
    next: NextFunction,
): void {

    const authorization =
        req.headers.authorization;


    if (
        authorization !==
        `Bearer ${GATEWAY_SECRET}`
    ) {

        res.status(401).json({
            detail:
                "Unauthorized",
        });

        return;
    }


    next();
}


// ============================================================
// Internal HTTP server
// ============================================================
//
// This lets FastAPI/Celery send messages later.
//
// Example:
//
// POST http://whatsapp:3001/send/text
//
// {
//     "recipient": "2010xxxxxxx@c.us",
//     "text": "Your appointment is confirmed."
// }
// ============================================================

function startInternalServer(): void {

    const app =
        express();


    app.use(
        express.json({
            limit:
                "2mb",
        }),
    );


    // --------------------------------------------------------
    // Health
    // --------------------------------------------------------

    app.get(
        "/health",

        (
            _req: Request,
            res: Response,
        ) => {

            res.json({
                status:
                    "ok",

                whatsapp_connected:
                    whatsappClient !== null,

                session:
                    SESSION_ID,
            });
        },
    );


    // --------------------------------------------------------
    // Send text
    // --------------------------------------------------------

    app.post(
        "/send/text",

        authenticate,

        async (
            req: Request,
            res: Response,
        ) => {

            const {
                recipient,
                text,
            } =
                req.body;


            if (
                !recipient ||
                !text
            ) {

                res.status(422).json({
                    detail:
                        "recipient and text are required",
                });

                return;
            }


            try {

                await sendWhatsAppText(
                    recipient,
                    text,
                );


                res.json({
                    sent:
                        true,
                });

            } catch (error) {

                console.error(
                    "[WhatsApp] Send failed",
                    error,
                );


                res.status(503).json({
                    detail:
                        "WhatsApp unavailable",
                });
            }
        },
    );


    // --------------------------------------------------------
    // Start server
    // --------------------------------------------------------

    app.listen(
        PORT,
        "0.0.0.0",

        () => {

            console.log(
                `[WhatsApp] Internal server running on port ${PORT}`,
            );
        },
    );
}


// ============================================================
// Start OpenWA
// ============================================================

async function startWhatsApp(): Promise<void> {

    console.log(
        "[WhatsApp] Starting OpenWA...",
    );


    whatsappClient =
        await create({
            sessionId:
                SESSION_ID,

            headless:
                true,

            qrTimeout:
                0,

            authTimeout:
                0,

            restartOnCrash:
                startWhatsApp,

            useChrome:
                true,

            killProcessOnBrowserClose:
                true,
        });


    console.log(
        "[WhatsApp] Connected.",
    );


    /*
     * onMessage receives incoming WhatsApp
     * messages only.
     */
    await whatsappClient.onMessage(
        async (
            message: Message,
        ) => {

            /*
             * Don't block OpenWA's event listener
             * while LangGraph / voice processing
             * runs.
             */
            void handleMessage(
                message,
            );
        },
    );
}


// ============================================================
// Graceful shutdown
// ============================================================

async function shutdown(
    signal: string,
): Promise<void> {

    console.log(
        `[WhatsApp] Received ${signal}`,
    );


    if (whatsappClient) {

        try {

            await whatsappClient.kill(
                signal,
            );

        } catch (error) {

            console.error(
                "[WhatsApp] Shutdown error",
                error,
            );
        }
    }


    process.exit(0);
}


process.on(
    "SIGTERM",
    () => {
        void shutdown(
            "SIGTERM",
        );
    },
);


process.on(
    "SIGINT",
    () => {
        void shutdown(
            "SIGINT",
        );
    },
);


// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {

    startInternalServer();

    await startWhatsApp();
}


main().catch(
    (error) => {

        console.error(
            "[WhatsApp] Fatal error",
            error,
        );

        process.exit(1);
    },
);